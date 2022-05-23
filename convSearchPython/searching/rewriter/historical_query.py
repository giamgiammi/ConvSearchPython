"""Historical query expansion (HQExp) from
Query and Answer Expansion from Conversation History(Jheng-Hong Yang et al)"""

from typing import Iterable, List

import pandas as pd
import pyterrier as pt

from convSearchPython.dataset import QueryMap
from convSearchPython.pipelines import IndexConf, Rewriter, StepType
from convSearchPython.utils.data_utils import replace_col_with_history


def _first_element(data: pd.DataFrame):
    return data.iloc[0]


def _hq_expansion(query: str,
                  transformer, session_relevant_threshold: float,
                  query_relevant_threshold: float, non_ambiguous_threshold: float, is_first_query: bool,
                  queries_keywords: List[Iterable[str]], session_keywords: List[str]) -> str:
    """
    Expand query with historical keywords

    Args:
        query: the query string
        transformer: the pyterrier transformer to calculate the scores
        session_relevant_threshold: threshold for a session-relevant keyword
        query_relevant_threshold: threshold for a query-relevant keyword
        non_ambiguous_threshold: threshold for a non-ambiguous query
        is_first_query: if the query is the first in a conversation
        queries_keywords: list of iterable with previous queries keywords
        session_keywords: list of session keywords

    Returns:
        the expanded query
    """
    tokens = query.split()
    data = pd.DataFrame({'query': tokens, 'qid': list(range(1, len(tokens) + 1))})
    res = transformer(data)[['qid', 'score']].groupby('qid').aggregate(_first_element)
    res = res.merge(data, how='left', on='qid')[['query', 'score']]
    query_relevant = res[res['score'] > query_relevant_threshold]['query']
    session_relevant = res[res['score'] > session_relevant_threshold]['query']

    queries_keywords.append(query_relevant)
    session_keywords.extend(session_relevant)

    if is_first_query:
        new_query = query
    else:
        parts = tokens + session_keywords
        query_search = transformer.search(query)
        query_score = query_search.iloc[0]['score'] if len(query_search) > 0 else 0
        if query_score < non_ambiguous_threshold:
            for old in queries_keywords[-4:]:
                parts.extend(old)
        new_query = ' '.join(parts)

    # queries_keywords.append(query_relevant)
    # session_keywords.extend(session_relevant)

    return new_query


class HistoricalQueryRewriter(Rewriter):
    """
    HQE Rewriter
    """

    def __init__(self, query_map: QueryMap,
                 session_relevant_threshold: float, query_relevant_threshold: float, non_ambiguous_threshold: float,
                 index: str, model: str = 'BM25', c: float = '0.75', **kwargs):
        """
        Args:
            query_map: query map structure
            session_relevant_threshold: threshold for session keywords (\\(r_s\\) in paper)
            query_relevant_threshold: threshold for query keyword (\\(r_q\\) in paper)
            non_ambiguous_threshold: threshold for non-ambiguous query (\\(\\theta\\) in paper)
            index: index name
            model: model to use
            c: model tuning
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._query_map = query_map
        self._session_relevant_threshold = session_relevant_threshold
        self._query_relevant_threshold = query_relevant_threshold
        self._non_ambiguous_threshold = non_ambiguous_threshold
        self._model = model
        self._index = index
        self._c = c

        self._transformer = None

    @property
    def name(self) -> str:
        return f'HQE-{self._model}-{self._c}' \
               f'-rs{self._session_relevant_threshold}' \
               f'-rq{self._query_relevant_threshold}' \
               f'-theta{self._non_ambiguous_threshold}'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def _get_tr(self):
        if self._transformer is not None:
            return self._transformer
        index_conf = IndexConf.load_index(self._index)
        self._transformer = pt.BatchRetrieve(index_conf.index,
                                             wmodel=self._model,
                                             controls={'c': self._c},
                                             properties=index_conf.properties,
                                             metadata=['docno'],
                                             num_results=1)
        return self._transformer

    def _rewrite_single(self, data: pd.Series, conversation: list,
                        queries_keywords: List[Iterable[str]],
                        session_keywords: List[str]):
        query = data['query']
        conv = self._query_map[data['qid']][0]
        if conv != conversation[0]:
            conversation[0] = conv
            queries_keywords.clear()
            session_keywords.clear()
            return _hq_expansion(query, self._get_tr(), self._session_relevant_threshold,
                                 self._query_relevant_threshold,
                                 self._non_ambiguous_threshold, True, queries_keywords, session_keywords)
        else:
            return _hq_expansion(query, self._get_tr(), self._session_relevant_threshold,
                                 self._query_relevant_threshold,
                                 self._non_ambiguous_threshold, False, queries_keywords, session_keywords)

    def rewrite(self, queries: pd.DataFrame) -> pd.DataFrame:
        queries_keywords: List[Iterable[str]] = []
        session_keywords: List[str] = []
        conversation = [None]
        return replace_col_with_history(
            'query',
            queries.apply(self._rewrite_single, axis=1, args=(conversation, queries_keywords, session_keywords)),
            queries
        )
