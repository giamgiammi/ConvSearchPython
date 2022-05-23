"""
# Sub Index Search

## Idea

Inside a conversation, the first utterance generally set the main arguments of the
entire conversation. So it seems reasonable to think that the answer to subsequent
utterance (in the same conversation) should appear among the results of the first
one.

Exploiting this idea, this module provides an implementation for pipelines that
use the results of the first utterance for constructing a temporary index and
utilize that for the subsequent ones.

## Usage

Construct a `SubIndexPipelineFactory` with the intended options.

The temp index creation support two different implementation:

- in-memory index:

    this is implemented using pyterrier DFIndexer. Unfortunately RM3
    (and possibly other rewriters that use terrier query language) do not work with this implementation.

- standard index on temp folder:

    for this to be used the environment variable `SUB_INDEX_DIR` needs
    to be set and refer to a writable path where the temp indexes will be created. To keep everything in
    memory you can set this variable to /dev/shm (on Linux) as it's guaranteed to be a ramdisk (if exists).

    Note that, if en error occur, the temp indexes may not be automatically deleted.
"""
import gc
import logging
from os import environ
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Iterable, Type, NamedTuple

import pandas as pd
from multiprocess.pool import Pool

from convSearchPython.basics import pyterrier as pt
from convSearchPython.dataset import Conversations
from convSearchPython.pipelines import IndexConf, Step, Pipeline, ChainPipeline, StepType, Model, Rewriter, RM3
from convSearchPython.pipelines.factory.base_factory import BasePipelineFactory
from convSearchPython.utils import recursive_rm, mkdir, parse_bool, mktmpdir
from convSearchPython.utils.imports import instantiate_class, import_object

sub_index_dir_info = [False, False]


def create_sub_index(results: pd.DataFrame, properties: dict, metadata: list) -> Tuple[Any, Optional[Path]]:
    """
    Create a sub index from search results.

    If "SUB_INDEX_DIR" environment variable is set, an index is created at that location.
    Otherwise, an in-memory index will be used (but RM3 will not work).

    Args:
        results: search results to use for documents (must include 'text' column)
        properties: indexing properties
        metadata: retrieve metadata

    Returns:
        A tuple of (index, directory), where index is the pyterrier index and directory is a Path to the index directory, if any, or None
    """
    sub_index_dir = environ.get('SUB_INDEX_DIR')
    if sub_index_dir is None:
        if not sub_index_dir_info[1]:
            logging.getLogger(__name__).warning('using in-memory sub-index implementation: RM3 will not work')
            sub_index_dir_info[1] = True
        import pyterrier.index as pt_index
        indexer = pt.DFIndexer('', type=pt_index.IndexingType.MEMORY)
        for k, v in properties.items():
            indexer.setProperty(k, v)
        index_ref = indexer.index(results['text'], results[metadata])
        return pt.IndexFactory.of(index_ref), None
    else:
        if not sub_index_dir_info[0]:
            logging.getLogger(__name__).info(f'Using {sub_index_dir} to store temp indexes')
            sub_index_dir_info[0] = True
        directory = Path(sub_index_dir)
        if not (directory.exists() and directory.is_dir()):
            raise Exception(f'invalid SUB_INDEX_DIR {sub_index_dir}')
        directory = mktmpdir(directory)
        mkdir(directory)
        indexer = pt.IterDictIndexer(str(directory))
        for k, v in properties.items():
            indexer.setProperty(k, v)
        index_ref = indexer.index(({'docno': d['docno'], 'text': d['text']} for _, d in results.iterrows()),
                                  meta=['docno', 'text'],
                                  meta_lengths=[44, 4096],
                                  fields=['text']
                                  )
        return pt.IndexFactory.of(index_ref), directory


class _SubIndexJob(NamedTuple):
    """
    Internal class.

    Callable for a sub-index job.
    """
    props: dict
    first_results: pd.DataFrame
    conversations: Conversations
    metadata: list
    sub_steps_args: Tuple[Tuple[Type[Any], Tuple[Any], Dict[str, Any]]]
    kwargs: dict
    rerun_first: bool
    queries: pd.DataFrame

    def __call__(self, conv_list: list) -> List[pd.DataFrame]:
        parts = []
        conv_set = set(conv_list)
        index, directory = create_sub_index(self.first_results[self.first_results['qid'].isin(conv_set)], self.props,
                                            ['docno', 'text'])
        index_name = IndexConf.add_index(index, self.props)
        steps = list(t[0](*t[1], **{
            **self.kwargs,
            **t[2],
            'conversations': self.conversations,
            'index': index_name,
            'metadata': self.metadata,
        }) for t in self.sub_steps_args)
        if self.rerun_first:
            parts.append(ChainPipeline(steps, '', self.conversations)(self.queries[self.queries['qid'].isin(conv_set)]))
        else:
            parts.append(self.first_results[self.first_results['qid'].isin(conv_set)])
            r = ChainPipeline(steps, '', self.conversations)(self.queries[self.queries['qid'].isin(conv_set)])
            parts.append(r[(r['qid'].isin(conv_set)) & (r['qid'] != conv_list[0])])
            # ChainPipeline(steps, '', self.conversations)(
            #     self.queries[(self.queries['qid'].isin(conv_set)) & (self.queries['qid'] != conv_list[0])]
            # )
        IndexConf.remove_index(index_name)
        if directory is not None:
            recursive_rm(directory)

        return parts


class SubIndexConversationSearcher(Pipeline):
    """
    Implementation of a sub-index searcher.

    Even if this class implements the `Pipeline` interface, it should not
    be instantiated directly. Instead, use `SubIndexPipelineFactory`
    """
    def __init__(self,
                 base_steps: Iterable[Step],
                 sub_steps_args: Iterable[Tuple[Type[Any], Tuple[Any], Dict[str, Any]]],
                 conversations: Conversations, index: str, metadata: List[str],
                 rerun_first_query: bool,
                 name: str,
                 **kwargs):
        """
        Args:
            base_steps: step for first-utterance-search
            sub_steps_args: iterable of tuples (class, *args, **kwargs) for instantiating
            steps to be used with subsequent utterance
            conversations: conversations structure
            index: index name
            metadata: metadata list
            rerun_first_query: if True, the first utterances results are re-run on the temp index
            name: pipeline parsable name
            **kwargs: extra arguments, can be used in step initialization
        """
        super().__init__(**kwargs)
        self._base_passes = tuple(base_steps)
        self._sub_steps_args = tuple(sub_steps_args)
        self._conversations = conversations
        self._index = index
        self._metadata = metadata
        self._rerun_first = rerun_first_query
        self._kwargs = kwargs
        self._name = name

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:
        return self._name

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        first_qids = set(x[0] for x in self._conversations.values())
        first_queries = queries[queries['qid'].isin(first_qids)]
        first_results = ChainPipeline(self._base_passes, '', self._conversations)(first_queries, parallel_pool)

        props = IndexConf.load_index(self._index).properties
        if props is None:
            props = {}
        parts = []
        is_sequential: bool = parallel_pool is None
        if not is_sequential:
            # temporary instantiate object, so it's possible to check
            # if there is a sequential one
            for t in self._sub_steps_args:
                step = t[0](*t[1], **{
                        **self._kwargs,
                        **t[2],
                        'conversations': self._conversations,
                        'index': self._index,
                        'metadata': self._metadata,
                    })
                if step.type == StepType.SEQUENTIAL:
                    self._logger.warning('fallback on sequential execution cause a sequential step was found: "%s"', t[0])
                    is_sequential = True
                    break
            gc.collect()  # clean objects
        job = _SubIndexJob(props, first_results, self._conversations, self._metadata, self._sub_steps_args,
                           self._kwargs, self._rerun_first, queries)
        if is_sequential:
            for conv_list in self._conversations.values():
                parts.extend(job(conv_list))
        else:
            for x in parallel_pool.imap(job, self._conversations.values()):
                parts.extend(x)

        results = pd.concat(parts)

        # metadata adjustment
        if 'text' not in self._metadata:
            results.drop('text', 1, inplace=True)

        return results


class SubIndexPipelineFactory(BasePipelineFactory):
    """
    Factory for building a pipeline that uses a sub-index for subsequent utterance search.

    You can set:

    - base model (required)
    - base rewriter
    - base rm3
    - model (required)
    - rewriter
    - rm3
    - reranker
    - if the first utterance should be re-ran on the sub index

    Base settings refer to first utterance search. Settings for subsequent utterance
    are the same as `convSearchPython.pipelines.factory.base_factory.BasePipelineFactory`.

    By default, the number of results on the base model is set to 10000.
    You can override it by setting `base.model.num_results = <value>`.

    Example config:

    ```
    subFactory = convSearchPython.pipelines.factory.sub_index.SubIndexPipelineFactory
    subFactory.base.model = BM25
    subFactory.base.rm3.lambda = 0.6
    subFactory.base.rerun_first = false
    subFactory.model = DirichletLM
    subFactory.rewriter = full.path.to.rewriter
    ```
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: common parameters for step initialization
        """
        super().__init__(**kwargs)

        self._base_model: Optional[Dict[str, Any]] = None
        self._base_rm3: Optional[Dict[str, Any]] = None
        self._base_rewriter: Optional[Tuple[str, tuple, Dict[str, Any]]] = None

        self._rerun_first = False

    @property
    def rerun_first(self) -> bool:
        """Rerun first option"""
        return self._rerun_first

    @property
    def base_model(self) -> Optional[Model]:
        """Base model"""
        return Model(**self._base_model) \
            if self._base_model is not None else None

    @property
    def base_rewriter(self) -> Optional[Rewriter]:
        """Base rewriter"""
        return instantiate_class(self._base_rewriter[0], *self._base_rewriter[1], **{**self._kwargs, **self._base_rewriter[2]}) \
            if self._base_rewriter is not None else None

    @property
    def base_rm3(self) -> Optional[RM3]:
        """Base RM3"""
        return RM3(**self._base_rm3) if self._base_rm3 is not None else None

    def set_base_model(self, wmodel: str, **controls):
        """
        Set the base pipeline model
        Args:
            wmodel: name of the model (pyterrier wmodel BatchRetrieve parameter)
            **controls: model configuration parameters
        """
        self._base_model = {'wmodel': wmodel, 'controls': controls,
                            'index': self._kwargs['index'], 'metadata': self._kwargs['metadata']}

    def set_base_rewriter(self, classname: str, *args, **kwargs):
        """
        Set pipeline base rewriter
        Args:
            classname: rewriter classname
            *args: positional arguments
            **kwargs: dict arguments
        """
        self._base_rewriter = (classname, args, kwargs)

    def set_base_rm3(self, fb_terms: int, fb_docs: int, fb_lambda: float):
        """
        Set pipeline base RM3
        Args:
            fb_terms: number of best terms to consider
            fb_docs: number of docs to consider
            fb_lambda: lambda factor
        """
        self._base_rm3 = {'index': self._kwargs['index'], 'fb_terms': fb_terms,
                          'fb_docs': fb_docs, 'fb_lambda': fb_lambda}

    def set_rerun_first(self, rerun_first: bool):
        """
        Set if the first utterance search should be repeated on the sub-index

        Args:
            rerun_first: True if the first utterance search should be repeated, False otherwise
        """
        self._rerun_first = rerun_first

    def set(self, config: Dict[str, Any]):
        base_config = {k: v for k, v in config.items() if k.startswith('base.')}
        super().set({k: v for k, v in config.items() if k not in base_config})

        model_wmodel = None
        model = {'num_results': 10000}
        rewriter_class = None
        rewriter = {}
        rm3 = {}

        for key, value in base_config.items():
            if key.startswith('base.model.'):
                key = key[11:]
                model[key] = value
            elif key.startswith('base.rewriter.'):
                key = key[14:]
                rewriter[key] = value
            elif key == 'base.rm3.terms' or key == 'base.rm3.fb_terms':
                rm3['fb_terms'] = value
            elif key == 'base.rm3.docs' or key == 'base.rm3.fb_docs':
                rm3['fb_docs'] = value
            elif key == 'base.rm3.lambda' or key == 'base.rm3.fb_lambda':
                rm3['fb_lambda'] = value
            elif key == 'base.model':
                model_wmodel = value
            elif key == 'base.rewriter':
                rewriter_class = value
            elif key == 'rerun_first':
                self._rerun_first = parse_bool(value)
            else:
                self._logger.warning('unknown parameter "%s", ignoring...', key)

        if model_wmodel is not None:
            self.set_base_model(model_wmodel, **model)
        elif len(model) > 0:
            self._logger.warning('ignoring base.model.* parameters because base.model is not set')

        if rewriter_class is not None:
            self.set_base_rewriter(instantiate_class(rewriter_class, **rewriter))
        elif len(rewriter) > 0:
            self._logger.warning('ignoring base.rewriter.* parameters because base.rewriter is not set')

        if len(rm3) > 0:
            self.set_base_rm3(**rm3)

    def _get_name(self):
        base = ['none', 'none', 'none']
        sub = ['none', 'none', 'none', 'none']

        base[0] = self.base_model.name
        sub[0] = self.model.name

        base_rw = self.base_rewriter
        if base_rw is not None:
            base[1] = base_rw.name
        sub_rw = self.rewriter
        if sub_rw is not None:
            sub[1] = sub_rw.name

        base_prf = self.base_rm3
        if base_prf is not None:
            base[2] = base_prf.name
        sub_prf = self.rm3
        if sub_prf is not None:
            sub[2] = sub_prf.name

        sub_rr = self.reranker
        if sub_rr is not None:
            sub[3] = sub_rr.name

        parts = []
        for i in range(len(base)):
            parts.append(f'{base[i]}--{sub[i]}')
        parts.append(sub[3])

        return '_'.join(parts)

    def build(self) -> SubIndexConversationSearcher:
        base_model = self.base_model
        if base_model is None:
            raise TypeError('missing base model')
        base_rw = self.base_rewriter
        base_prf = self.base_rm3
        if base_rw is None:
            if base_prf is None:
                base_steps = (base_model, )
            else:
                base_steps = (base_model, base_prf, base_model)
        else:
            if base_prf is None:
                base_steps = (base_rw, base_model)
            else:
                base_steps = (base_rw, base_model, base_prf, base_model)

        sub_model = self._model
        if sub_model is None:
            raise TypeError('missing sub model')
        sub_rw = self._rewriter
        sub_prf = self._rm3
        sub_rr = self._reranker
        sub_steps_args = []
        if sub_rw is None:
            if sub_prf is None:
                sub_steps_args.append((Model, tuple(), sub_model))
            else:
                sub_steps_args.extend(((Model, tuple(), sub_model),
                                       (RM3, tuple(), sub_prf),
                                       (Model, tuple(), sub_model)))
        else:
            if sub_prf is None:
                sub_steps_args.extend(((import_object(sub_rw[0]), sub_rw[1], sub_rw[2]),
                                       (Model, tuple(), sub_model)))
            else:
                sub_steps_args.extend(((import_object(sub_rw[0]), sub_rw[1], sub_rw[2]),
                                       (Model, tuple(), sub_model),
                                       (RM3, tuple(), sub_prf),
                                       (Model, tuple(), sub_model)))
        if sub_rr is not None:
            sub_steps_args.append((import_object(sub_rr[0]), sub_rr[1], sub_rr[2]))

        searcher = SubIndexConversationSearcher(base_steps=base_steps, sub_steps_args=sub_steps_args,
                                                rerun_first_query=self._rerun_first,
                                                name=self._get_name(), **self._kwargs)
        return searcher

