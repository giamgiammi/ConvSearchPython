"""
This module contains a collection of pipelines that add
a bottom up reranking filter at the end of the searching.

The purpose of the filter is to assign relevant documents to utterance
in a conversation, starting with the last one, and
removing (or penalizing) the selected document from the others utterances.

The filter work is guided by 2 parameter:

- multiplier: this is the multiplicative factor used to reduce the score of the
              already seen documents. $$ 0 \le multiplier < 1 $$
- max_rank: this indicates the maximum rank to consider when keeping seen docs in memory.
            Given that a normal search pull 1000 documents, it's useless to consider
            **all** the seen documents. Instead, we want to focus on the top-k documents
            that our system considered relevant

The filter is implemented in `ranking.bottom_up_filter`
"""

from convSearchPython.pipelines.baselines import *
from convSearchPython.ranking.bottom_up_filter import BottomUpReranker


class BottomUpUamPipeline(AbstractParallelPipeline):
    """Pipeline that tries to reproduce the BottomUp approach of UAmsterdam"""
    def __init__(self, index, metadata, conversations: Conversations, multiplier=0.0, max_rank=1000, mu=2500, c=0.75, fb_terms=10, fb_docs=10,
                 fb_lambda=0.5, **kwargs):
        if 'rm3' in kwargs:
            kwargs.pop('rm3')
        super().__init__(rm3=True, fb_terms=fb_terms, fb_docs=fb_docs, fb_lambda=fb_lambda, **kwargs)
        self.__multiplier = multiplier
        self.__max_rank = max_rank
        self._conversations = conversations
        self._qlm = Model('DirichletLM', {'c': mu}, index, metadata)
        self._bm25 = Model('BM25', {'c': c}, index, metadata)
        self._btm = BottomUpReranker(multiplier=multiplier, max_rank=max_rank, conversations=conversations, **kwargs)
        self._rm3 = RM3(fb_terms=fb_terms, fb_docs=fb_docs, fb_lambda=fb_lambda, index=index, **kwargs)

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None, **kwargs) -> pd.DataFrame:
        if parallel_pool is None:
            data = self._bm25(queries)
            data = self._rm3(data)
            data = self._qlm(data)
            data = self._btm(data)
        else:
            data = self._conv_parallel_run(self._bm25, parallel_pool, queries, self._conversations)
            data = self._conv_parallel_run(self._rm3, parallel_pool, data, self._conversations)
            data = self._conv_parallel_run(self._qlm, parallel_pool, data, self._conversations)
            data = self._conv_parallel_run(self._btm, parallel_pool, data, self._conversations)
        return data

    @property
    def name(self):
        """BM25-DLM_none_rm3_bottomUp"""
        return f'{self._bm25.name}-{self._qlm.name}_none_{self._rm3.name}' \
               f'_{self._btm.name}'

