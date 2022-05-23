"""Bottom up filtering re-ranking strategy"""

import pandas as pd

from convSearchPython.dataset import Conversations
from convSearchPython.pipelines import Reranker, StepType
from convSearchPython.utils.data_utils import sort_results_update_rank


class BottomUpReranker(Reranker):
    """
    Bottom up reranker
    """
    def __init__(self, conversations: Conversations,
                 multiplier: float, max_rank: int,
                 **kwargs):
        """
        Args:
            conversations: conversations structure
            multiplier: factor to apply to seen docs' score
            max_rank: max rank to save a doc as seen
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._multiplier = multiplier
        self._max_rank = max_rank

    @property
    def name(self) -> str:
        return f'BottomUp-m{self._multiplier}-r{self._max_rank}'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def rerank(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        docs_to_change = {}

        for conv_id, conv_qid_list in self._conversations.items():
            seen = set()
            if conv_qid_list[0] not in data['qid'].unique():
                continue
            for qid in reversed(conv_qid_list):
                query_data = data[data['qid'] == qid].sort_values('rank')
                top_k = query_data.iloc[:self._max_rank]
                top_docs = top_k['docno'].unique()
                seen_docs = set(query_data['docno'].unique()).intersection(seen)
                docs_to_change[qid] = seen_docs
                seen.update(top_docs)

        scores = []
        for _, row in data.iterrows():
            if row['docno'] in docs_to_change[row['qid']]:
                scores.append(row['score'] * self._multiplier)
            else:
                scores.append(row['score'])
        data['score'] = scores
        return sort_results_update_rank(data, self._conversations)

    # def rerank(self, data: pd.DataFrame) -> pd.DataFrame:
    #     seen = collections.defaultdict(set)
    #     docno_to_be_removed = {}
    #
    #     for conv_id, conv_qid_list in self._conversations.items():
    #         conv_seen = seen[conv_id]
    #         for qid in reversed(conv_qid_list):
    #             query_data = data[data['qid'] == qid]
    #             query_data.sort_values('rank', inplace=True)
    #             top_k = query_data.iloc[:self._max_rank]
    #             top_docs = set(top_k['docno'].unique())
    #             docno_to_be_removed[qid] = set(query_data['docno'].unique()).difference(conv_seen)
    #             conv_seen.update(top_docs)
    #
    #     parts = []
    #     for conv_id, conv_qid_list in self._conversations.items():
    #         conv_data = data[data['qid'].isin(conv_qid_list)].copy()
    #         scores = []
    #         for i, row in conv_data.iterrows():
    #             qid = row['qid']
    #             docno = row['docno']
    #             score = row['score']
    #             if docno in docno_to_be_removed[qid]:
    #                 score = score * self._multiplier
    #             scores.append(score)
    #         # conv_data['score'] = scores
    #         replace_col_with_history('score', scores, conv_data, inplace=True)
    #         conv_data.sort_values(['qid', 'score'], key=evaluation_key, inplace=True)
    #         conv_data['rank'] = range(len(conv_data))
    #         parts.append(conv_data)
    #
    #     new_data = pd.concat(parts)
    #     new_data.reset_index(drop=True)
    #     return new_data
