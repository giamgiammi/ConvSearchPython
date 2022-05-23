"""Implement first query rewriter"""
from typing import Dict, List, Tuple
from pandas import DataFrame, Series

from convSearchPython.dataset import Conversations, QueryMap
from convSearchPython.pipelines import Rewriter, StepType
from convSearchPython.searching.rewriter.utils import get_query
from convSearchPython.utils.data_utils import replace_col_with_history


def _repeat_query(q):
    return f'{q} {q}'


def _no_repeat_query(q):
    return q


class FirstQueryRewriter(Rewriter):
    """
    Rewriter that concatenate the current query with the first one, in the same conversation.

    I.E. \\( q_{r_{i}} = q_1 + q_i \\)
    """
    def __init__(self, conversations: Conversations, query_map: QueryMap,
                 variant='no-repeat', **kwargs):
        """
        Args:
            conversations: conversations structure
            query_map: query map structure
            variant: can be 'no-repeat' or 'repeat'.

                With 'no-repeat' no repetition will be made for 1° and 2° query in a conversation.

                With 'repeat' the 1° and 2° query will be repeated, respectively, 3 and 2 times.
            **kwargs:

        Raises:
            ValueError: if variant is not 'no-repeat' or 'repeat'
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self._variant = variant
        if variant not in ('repeat', 'no-repeat'):
            raise ValueError(f'invalid variant {variant}')

    @property
    def name(self) -> str:
        if self._variant == 'no-repeat':
            return 'FirstQ'
        elif self._variant == 'repeat':
            return 'FirstQ-repeat'
        else:
            raise ValueError(f'invalid variant {self._variant}')

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def _rewrite_single(self, query: Series, queries: DataFrame, zero):
        qid = query['qid']
        str_query = query['query']
        conv_id, index = self._query_map[qid]
        if index == 0:
            return zero(str_query)
        f_query = get_query(queries, self._conversations, conv_id, 0)
        return f'{f_query} {str_query}'

    def rewrite(self, queries: DataFrame) -> DataFrame:
        if self._variant == 'no-repeat':
            values = queries.apply(self._rewrite_single, axis=1, args=(queries, _no_repeat_query))
        elif self._variant == 'repeat':
            values = queries.apply(self._rewrite_single, axis=1, args=(queries, _repeat_query))
        else:
            raise ValueError(f'invalid variant {self._variant}')

        return replace_col_with_history('query', values, queries)
