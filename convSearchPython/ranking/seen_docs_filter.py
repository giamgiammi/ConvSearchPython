"""
This module contains a rewriter that apply
a reranking filtering that remove (or penalize) relevant docs found for
an utterance from the following utterances' results (inside
a conversation).

The filter work is guided by 2 parameter:

- multiplier: this is the multiplicative factor used to reduce the score of the
              already seen documents. $$ 0 \\le multiplier < 1 $$
- max_rank: this indicates the maximum rank to consider when keeping seen docs in memory.
            Given that a normal search pull 1000 documents, it's useless to consider
            **all** the seen documents. Instead, we want to focus on the top-k documents
            that our system considered relevant
"""

import pandas as pd

from convSearchPython.dataset import QueryMap, Conversations
from convSearchPython.pipelines import Reranker, StepType
from convSearchPython.utils.data_utils import replace_col_with_history, sort_results_update_rank


class SeenFilterReranker(Reranker):
    """
    Reranker that filter already seen docs in the same conversation
    """

    def __init__(self, conversations: Conversations, query_map: QueryMap, multiplier: float, max_rank: int, **kwargs):
        """
        Args:
            conversations: conversations structure
            query_map: query map structure
            multiplier: factor to apply to already seen docs' score
            max_rank: max rank to save a doc as seen
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self._multiplier = multiplier
        self._max_rank = max_rank

    @property
    def name(self) -> str:
        return f'SeenFilter-m{self._multiplier}-r{self._max_rank}'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def _rerank_single(self, data: pd.Series, seen: dict):
        score = data['score']
        rank = data['rank']
        qid = data['qid']
        conversation = self._query_map[qid][0]
        doc = data['docno']
        conv_seen = seen.get(conversation)
        if conv_seen is None:
            conv_seen = set()
            seen[conversation] = conv_seen
        if doc in conv_seen:
            score = score * self._multiplier
        if rank <= self._max_rank:
            conv_seen.add(doc)
        return score

    def rerank(self, results: pd.DataFrame) -> pd.DataFrame:
        seen = {}
        results = replace_col_with_history(
            'score',
            results.apply(self._rerank_single, axis=1, args=(seen, )),
            results
        )
        # todo check if something has changed now that we manually sort
        return sort_results_update_rank(results, self._conversations)
