"""
This module contains a Pipeline Factory implementation for the common case
of rewriter -> model -> RM3 -> reranker.
"""
import logging
from typing import Optional, Dict, Any, Tuple

from convSearchPython.pipelines import Model, Rewriter, Reranker, RM3, ChainPipeline
from convSearchPython.pipelines.factory import PipelineFactory
from convSearchPython.utils.imports import instantiate_class


class BasePipelineFactory(PipelineFactory):
    """
    Factory to build a pipeline composed of:

    rewriter -> model -> rm3 -> reranker

    Only model is mandatory.

    Example of configuration:

    ```
    factory1 = convSearchPython.pipelines.factory.base_factory.BasePipelineFactory
    factory1.model = DirichletLM
    factory1.model.num_results = 1000
    factory1.model.c = 2500
    factory1.rewriter = full.path.to.rewriter
    factory1.rewriter.p1 = 10
    factory1.rm3.terms = 20
    factory1.rm3.docs = 20
    factory1.rm3.lambda = 0.5
    factory1.reranker = full.path.to.reranker
    factory1.reranker.p2 = true
    ```
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: common arguments for pipelines
        """
        super().__init__(**kwargs)

        self._rm3: Optional[Dict[str, Any]] = None
        self._reranker: Optional[Tuple[str, tuple, Dict[str, Any]]] = None
        self._rewriter: Optional[Tuple[str, tuple, Dict[str, Any]]] = None
        self._model: Optional[Dict[str, Any]] = None
        self._kwargs = kwargs

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def model(self) -> Optional[Model]:
        """Model (wrapper) object"""
        return Model(**self._model) \
            if self._model is not None else None

    @property
    def rewriter(self) -> Optional[Rewriter]:
        """Rewriter object"""
        return instantiate_class(self._rewriter[0], *self._rewriter[1], **{**self._kwargs, **self._rewriter[2]}) \
            if self._rewriter is not None else None

    @property
    def reranker(self) -> Optional[Reranker]:
        """Reranker object"""
        return instantiate_class(self._reranker[0], *self._reranker[1], **{**self._kwargs, **self._reranker[2]}) \
            if self._reranker is not None else None

    @property
    def rm3(self) -> Optional[RM3]:
        """RM3 (wrapper) object"""
        return RM3(**self._rm3) if self._rm3 is not None else None

    def set(self, config: Dict[str, Any]):
        model_wmodel = None
        model = {}
        rewriter_class = None
        rewriter = {}
        reranker_class = None
        reranker = {}
        rm3 = {}

        for key, value in config.items():
            if key.startswith('model.'):
                key = key[6:]
                model[key] = value
            elif key.startswith('rewriter.'):
                key = key[9:]
                rewriter[key] = value
            elif key.startswith('reranker.'):
                key = key[9:]
                reranker[key] = value
            elif key == 'rm3.terms' or key == 'rm3.fb_terms':
                rm3['fb_terms'] = value
            elif key == 'rm3.docs' or key == 'rm3.fb_docs':
                rm3['fb_docs'] = value
            elif key == 'rm3.lambda' or key == 'rm3.fb_lambda':
                rm3['fb_lambda'] = value
            elif key == 'model':
                model_wmodel = value
            elif key == 'rewriter':
                rewriter_class = value
            elif key == 'reranker':
                reranker_class = value
            else:
                self._logger.warning('unknown parameter "%s", ignoring...', key)

        if model_wmodel is not None:
            self.set_model(model_wmodel, **model)
        elif len(model) > 0:
            self._logger.warning('ignoring model.* parameters because model is not set')

        if rewriter_class is not None:
            self.set_rewriter(rewriter_class, **rewriter)
        elif len(rewriter) > 0:
            self._logger.warning('ignoring rewriter.* parameters because rewriter is not set')

        if reranker_class is not None:
            self.set_reranker(reranker_class, **reranker)
        elif len(reranker) > 0:
            self._logger.warning('ignoring reranker.* parameters because reranker is not set')

        if len(rm3) > 0:
            self.set_rm3(**rm3)

    def set_model(self, wmodel: str, **controls):
        """
        Set the pipeline model
        Args:
            wmodel: name of the model (pyterrier wmodel BatchRetrieve parameter)
            **controls: model configuration parameters
        """
        self._model = {'wmodel': wmodel, 'controls': controls,
                       'index': self._kwargs['index'], 'metadata': self._kwargs['metadata']}

    def set_rewriter(self, classname: str, *args, **kwargs):
        """
        Set pipeline rewriter
        Args:
            classname: rewriter classname
            *args: positional arguments
            **kwargs: dict arguments
        """
        self._rewriter = (classname, args, kwargs)

    def set_reranker(self, classname: str, *args, **kwargs):
        """
        Set pipeline reranker
        Args:
            classname: reranker classname
            *args: positional arguments
            **kwargs: dict arguments
        """
        self._reranker = (classname, args, kwargs)

    def set_rm3(self, fb_terms: int, fb_docs: int, fb_lambda: float):
        """
        Set pipeline RM3
        Args:
            fb_terms: number of best terms to consider
            fb_docs: number of docs to consider
            fb_lambda: lambda factor
        """
        self._rm3 = {
            'index': self._kwargs['index'], 'fb_terms': fb_terms,
            'fb_docs': fb_docs, 'fb_lambda': fb_lambda
        }

    def build(self) -> ChainPipeline:
        model = self.model
        rewriter = self.rewriter
        rm3 = self.rm3
        reranker = self.reranker
        if model is None:
            raise TypeError('missing model')
        name_parts = [f'{model.name}', 'none', 'none', 'none']
        steps = []
        if rewriter is not None:
            steps.append(rewriter)
            name_parts[1] = getattr(rewriter, 'name', rewriter.__class__.__name__)
        steps.append(model)
        if rm3 is not None:
            steps.append(rm3)
            steps.append(model)
            name_parts[2] = rm3.name
        if reranker is not None:
            steps.append(reranker)
            name_parts[3] = getattr(reranker, 'name', reranker.__class__.__name__)

        return ChainPipeline(steps, '_'.join(name_parts), **self._kwargs)


if __name__ == '__main__':
    from convSearchPython.dataset.CAsT import cast_dataset

    queries, qrels, conversations, query_map = cast_dataset(2019)
    commons = {
        'queries': queries,
        'conversations': conversations,
        'query_map': query_map,
        'metadata': ['docno'],
        'index': 'custom',
    }
    factory = BasePipelineFactory(**commons)
    factory.set({
        'model': 'BM25',
        'model.c': 0.75,
        'rm3.terms': 15,
        'rm3.docs': 5,
        'rm3.lambda': 0.4,
    })

    pipeline: ChainPipeline = factory.build()
    print(pipeline)
    print(pipeline.steps)
    print(pipeline.name)
