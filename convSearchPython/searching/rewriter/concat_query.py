"""Query rewriter that concatenate all the conversation utterance up to that point"""

from pandas import DataFrame, Series

from convSearchPython.dataset import Conversations, QueryMap
from convSearchPython.pipelines import Rewriter, StepType
from convSearchPython.utils.data_utils import replace_col_with_history


class ConcatQueryRewriter(Rewriter):
    def __init__(self, conversations: Conversations,
                 query_map: QueryMap, **kwargs):
        """
        Query rewriter that concatenate all the conversation utterance up to that point.

        Args:
            conversations: conversations dict
            query_map: query_map dict
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map

    def _rewrite_single(self, query: Series, queries: DataFrame):
        qid = query['qid']
        conv_id, index = self._query_map[qid]
        return ' '.join(queries[queries['qid'] == q].iloc[0]['query']
                        for q in self._conversations[conv_id][0:index+1])

    def rewrite(self, queries: DataFrame) -> DataFrame:
        values = queries.apply(self._rewrite_single, axis=1, args=(queries, ))
        return replace_col_with_history('query', values, queries)

    @property
    def name(self) -> str:
        return 'ConcatQ'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL


if __name__ == '__main__':
    from convSearchPython.dataset.CAsT import cast_dataset
    _queries, _, _conv, _map = cast_dataset(2019)
    _queries2 = ConcatQueryRewriter(_conv, _map)(_queries)

    print(_queries[['qid', 'query']])

    print(_queries2[['qid', 'query']])
