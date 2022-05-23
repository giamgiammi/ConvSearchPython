"""Implements co-reference query rewriter"""
import gzip
import logging
import pickle
from os import environ
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch.cuda as cuda
from allennlp_models import pretrained
from pandas import DataFrame, Series

from convSearchPython.basics import conf
from convSearchPython.pipelines import Rewriter, StepType
from convSearchPython.utils.data_utils import replace_col_with_history

logging.getLogger('allennlp.common.params').setLevel(logging.ERROR)
logging.getLogger('allennlp.nn.initializers').setLevel(logging.ERROR)
logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

if conf.getboolean('GENERAL', 'quiet', fallback=True):
    logging.getLogger('allennlp.common.plugins').setLevel(logging.ERROR)
    logging.getLogger('allennlp.common.model_card').setLevel(logging.ERROR)
    logging.getLogger('cached_path').setLevel(logging.ERROR)
    logging.getLogger('allennlp.models.archival').setLevel(logging.ERROR)
    logging.getLogger('allennlp.data.vocabulary').setLevel(logging.ERROR)
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.ERROR)
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if 'NO_PRELOAD' not in environ:
    # make sure required data is downloaded when this module is loaded
    pretrained.load_predictor("coref-spanbert")


def get_default_predictor(no_cuda=False):
    """Get the default predictor for coref query rewriting

    :param no_cuda: if True cuda is not used even if available (default False)
    :returns: the default predictor for the co-reference"""
    if cuda.is_available() and not no_cuda:
        logger.info('Using cuda device "%s" for Coref1 (allennlp)', str(cuda.get_device_name(cuda.current_device())))
        return pretrained.load_predictor("coref-spanbert", cuda_device=cuda.current_device())
    return pretrained.load_predictor("coref-spanbert")


class AllennlpCoreferenceQueryRewriter(Rewriter):
    """
    Co-reference query rewriter that uses allennlp.

    If a cache file is provided resolved queries are automatically loaded from it
    """
    def __init__(self, conversations: Dict[str, List[str]],
                 query_map: Dict[str, Tuple[str, int]], allow_cuda=True,
                 cache_dir: Union[str, Path] = None, autosave=True,
                 **kwargs):
        """
        Args:
            conversations: the conversations structure
            query_map: the query map structure
            allow_cuda: if True cuda is used if available
            cache_dir: the cache dir path (None disable cache)
            autosave: if True the cache is automatically saved
            kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self._allow_cuda = allow_cuda
        self._autosave = autosave
        self.__logger = logging.getLogger(self.__class__.__name__)

        self._cache = None
        self._cache_file = None
        self._cache_modified = None
        if cache_dir is not None:
            self._cache_file = Path(cache_dir, f'{self.__module__}.{self.__class__.__name__}.pkl.gz')

        self._cuda = allow_cuda and cuda.is_available()
        self._predictor = None

    def write_cache(self):
        """Update the cache if it was modified"""
        if not self._cache_modified:
            return
        with self._cache_file.open('wb') as file_stream:
            with gzip.open(file_stream, 'wb') as stream:
                pickle.dump(self._cache, stream)
        self._cache_modified = False

    def load_cache(self):
        """Load cache from file (if exists)

        No need to call it is using rewrite or __call__ method"""
        if self._cache_file.exists():
            try:
                with self._cache_file.open('rb') as file_stream:
                    with gzip.open(file_stream, 'rb') as stream:
                        self._cache = pickle.load(stream)
                self._cache_modified = False
            except Exception as e:
                self.__logger.error(e)
                self._cache = {}
        else:
            self._cache = {}

    def _coref(self, query: Series, queries: DataFrame) -> str:
        if self._predictor is None:
            self._predictor = get_default_predictor(no_cuda=not self._allow_cuda)
        qid = query['qid']
        conv_id, conv_index = self._query_map[qid]
        current_conv_ids = self._conversations[conv_id][:conv_index + 1]
        current_conv_queries = queries[queries['qid'].isin(current_conv_ids)]
        full_concat = ' | '.join(current_conv_queries['query'])
        full_coref: str = self._predictor.coref_resolved(full_concat)
        rewritten_query = full_coref.split(' | ')[-1]

        # terrier fails if encounter char '
        rewritten_query = rewritten_query.replace("'s", "").replace("'", "")

        return rewritten_query

    def _rewrite_single(self, query: Series, queries: DataFrame):
        if self._cache is not None:
            qid = query['qid']
            rewritten_query = self._cache.get(qid)
            if rewritten_query is None:
                rewritten_query = self._coref(query, queries)
                self._cache[qid] = rewritten_query
                self._cache_modified = True
        else:
            rewritten_query = self._coref(query, queries)

        return rewritten_query

    def rewrite(self, queries: DataFrame) -> DataFrame:
        if self._cache is None and self._cache_file is not None:
            self.load_cache()
        values = queries.apply(self._rewrite_single, axis=1, args=(queries, ))
        return replace_col_with_history('query', values, queries)

    def cleanup(self):
        self._predictor = None
        self._cache = None

    @property
    def name(self) -> str:
        return 'Coref1'

    @property
    def type(self) -> StepType:
        if self._cuda:
            return StepType.SEQUENTIAL
        else:
            return StepType.CONVERSATIONALLY_PARALLEL


if __name__ == '__main__':
    def main():
        """For testing purpose"""
        # noinspection PyUnresolvedReferences
        import convSearchPython.basics
        from convSearchPython.dataset.CAsT import CastDataset

        queries, qrels, conversations, query_map = CastDataset.cast2019.get_dataset()
        # queries, qrels, conversations, query_map = CastDataset.cast2020.get_dataset()

        rewriter = AllennlpCoreferenceQueryRewriter(queries,
                                                    conversations,
                                                    query_map,
                                                    cache_dir='./workdir/cache/cast2019',
                                                    )
        # for i, row in queries.iterrows():
        #     orig = row['query']
        #     rw = rewrite(row)
        #     print('Orig: {}\nRewr: {}\n'.format(orig, rw))
        # rewrite.write_cache()
        queries['orign'] = queries['query']
        queries = rewriter(queries)
        print(queries[['orign', 'query']])


    main()
