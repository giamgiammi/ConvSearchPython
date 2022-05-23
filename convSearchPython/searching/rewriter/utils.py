"""Utils for rewriter"""
from typing import Dict, List
from pandas import DataFrame


def get_query(queries: DataFrame, conversations: Dict[str, List[str]], conv_id: str, index: int) -> str:
    """return a query from conversation id an index"""
    qid = conversations[conv_id][index]
    return queries[queries['qid'] == qid].iloc[0]['query']


def get_map_query_rewriter(rewritten_queries: Dict[str, str]) -> callable:
    """Obtain a query rewriter that remap queries with the given map"""

    def rewriter(q):
        return rewritten_queries[q['qid']]
    return rewriter

