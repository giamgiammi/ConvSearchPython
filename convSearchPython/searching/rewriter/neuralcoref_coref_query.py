"""Implements co-reference query rewriter with  neuralcoref library"""
from sys import stderr
from typing import Dict, List, Tuple, Union, Callable

import neuralcoref
import spacy
from pandas import DataFrame, Series

from convSearchPython.dataset import Conversations, QueryMap
from convSearchPython.pipelines import Rewriter, StepType
from convSearchPython.utils.data_utils import replace_col_with_history


def get_default_nlp() -> Callable:
    """Get the default nlp object for coreference resolution"""
    try:
        nlp = spacy.load('en')
    except Exception as e:
        print('Model needs to be downloaded with: python -m spacy download en', file=stderr)
        raise e
    neuralcoref.add_to_pipe(nlp)

    return nlp


def get_large_nlp() -> Callable:
    """Get a large nlp object for coreference resolution"""
    try:
        nlp = spacy.load('en_core_web_lg')
    except Exception as e:
        print('Model needs to be downloaded with: python -m spacy download en_core_web_lg', file=stderr)
        raise e
    neuralcoref.add_to_pipe(nlp)

    return nlp


class NeuralCorefRewriter(Rewriter):
    """
    Co-reference query rewriter that uses neuralcoref
    """
    def __init__(self, conversations: Conversations, query_map: QueryMap,
                 nlp: str = 'default', **kwargs):
        """
        Args:
            conversations: conversations structure
            query_map: query map structure
            nlp: a string ('default' or 'large) for the wanted spacy variant
            **kwargs: extra unused arguments

        Raises:
            ValueError: if nlp is not valid
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self._variant = nlp
        self._nlp = None
        if nlp not in ('default', 'large'):
            raise ValueError(f'invalid nlp argument: {nlp}')

    @property
    def name(self) -> str:
        if self._variant == 'default':
            return 'Coref2'
        return f'Coref2-{self._variant}'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def cleanup(self):
        self._nlp = None

    def _rewrite_single(self, query: Series, queries: DataFrame):
        qid = query['qid']
        conv_id, conv_index = self._query_map[qid]
        current_conv_ids = self._conversations[conv_id][:conv_index+1]
        current_conv_queries = queries[queries['qid'].isin(current_conv_ids)]
        full_concat = ' .; '.join(current_conv_queries['query'])
        full_coref: str = self._nlp(full_concat)._.coref_resolved
        rewritten_query = full_coref.split(' .; ')[-1]
        return rewritten_query

    def rewrite(self, queries: DataFrame) -> DataFrame:
        if self._nlp is None:
            if self._variant == 'default':
                self._nlp = get_default_nlp()
            elif self._variant == 'large':
                self._nlp = get_large_nlp()
            else:
                raise ValueError(f'invalid nlp variant: {self._variant}')
        return replace_col_with_history(
            'query',
            queries.apply(self._rewrite_single, axis=1, args=(queries, )),
            queries
        )


if __name__ == '__main__':
    from convSearchPython.dataset.CAsT import CastDataset
    _queries, _qrels, _conversations, _query_map = CastDataset.cast2019.get_dataset()

    difference = 0
    different = []
    rw = NeuralCorefRewriter(_conversations, _query_map)(_queries)
    rw_lg = NeuralCorefRewriter(_conversations, _query_map, 'large')(_queries)
    for i in range(len(_queries)):
        print(f'Original    : {_queries.iloc[i]["query"]}')
        print(f'Rewritten   : {rw.iloc[i]["query"]}')
        print(f'Rewritten lg: {rw_lg.iloc[i]["query"]}')
        print('--------------------------------------')
        if rw.iloc[i]['query'] != rw_lg.iloc[i]['query']:
            difference += 1
            different.append((rw.iloc[i]["query"], rw_lg.iloc[i]["query"]))
    print('\n######################################')
    print(f'The two rewrite are different in {difference} utterance of {len(_queries)}')
    print('######################################\n')
    for a, b in different:
        print(f'df: {a}\nlg: {b}\n----------------------------')
