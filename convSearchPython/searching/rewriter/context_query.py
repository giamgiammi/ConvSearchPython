"""Implement context query"""

from pandas import DataFrame, Series

from convSearchPython.dataset import Conversations, QueryMap
from convSearchPython.pipelines import Rewriter, StepType
from convSearchPython.searching.rewriter.utils import get_query
from convSearchPython.utils.data_utils import replace_col_with_history


def _repeat_query_zero(q):
    return f'{q} {q} {q}'


def _repeat_query_one(fq, q):
    return f'{fq} {fq} {q}'


def _no_repeat_query_zero(q):
    return q


def _no_repeat_query_one(fq, q):
    return f'{fq} {q}'


class ContextQueryRewriter(Rewriter):
    """
    Rewriter that concatenate the current query with the previous and the first query
    in the same conversation.

    I.E. \\( q_{r_{i}} = q_1 + q_{i-1} + q_i \\)
    """
    def __init__(self, conversations: Conversations, query_map: QueryMap,
                 variant='no-repeat',
                 **kwargs):
        """
        Args:
            conversations: conversations structure
            query_map: query map structure
            variant: can be 'no-repeat' or 'repeat'.

                With 'no-repeat' no repetition will be made for 1° and 2° query in a conversation.

                With 'repeat' the 1° and 2° query will be repeated, respectively, 3 and 2 times.
            **kwargs: extra unused arguments

        Raises:
            ValueError: if variant is not 'no-repeat' or 'repeat'
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self._variant = variant
        if variant not in ('repeat', 'no-repeat'):
            raise ValueError(f'invalid variant {variant}')

    def _rewrite_single(self, query: Series, queries: DataFrame, zero, one):
        qid = query['qid']
        str_query = query['query']
        conv_id, index = self._query_map[qid]
        if index == 0:
            return zero(str_query)
        f_query = get_query(queries, self._conversations, conv_id, 0)
        if index == 1:
            return one(f_query, str_query)
        c_query = get_query(queries, self._conversations, conv_id, index - 1)
        return f'{f_query} {c_query} {str_query}'

    def rewrite(self, queries: DataFrame) -> DataFrame:
        if self._variant == 'no-repeat':
            values = queries.apply(self._rewrite_single, axis=1,
                                   args=(queries, _no_repeat_query_zero, _no_repeat_query_one))
        elif self._variant == 'repeat':
            values = queries.apply(self._rewrite_single, axis=1,
                                   args=(queries, _repeat_query_zero, _repeat_query_one))
        else:
            raise ValueError(f'invalid variant {self._variant}')

        return replace_col_with_history('query', values, queries)

    @property
    def name(self) -> str:
        if self._variant == 'no-repeat':
            return 'ContextQ'
        elif self._variant == 'repeat':
            return 'ContextQ-repeat'
        else:
            raise ValueError(f'invalid variant {self._variant}')

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL
