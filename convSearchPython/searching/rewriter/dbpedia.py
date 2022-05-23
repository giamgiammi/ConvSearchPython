"""
This module contains a class for rewriting queries using wikipedia and conceptnet.

It's based on paper [Incoporating Query Context into a BERT Re-ranker](https://trec.nist.gov/pubs/trec28/papers/mpii.C.pdf)
"""
import collections
import logging
import re
from math import log
from time import sleep
from typing import List

import requests
from pandas import DataFrame, Series

from convSearchPython.dataset import Conversations, QueryMap
from convSearchPython.pipelines import Rewriter, StepType, IndexConf
from convSearchPython.utils.data_utils import replace_col_with_history


class DbPediaRewriter(Rewriter):
    """
    Rewriter that use Wikipedia and ConceptNet to do query expansion.
    """
    def __init__(self,  conversations: Conversations,
                 query_map: QueryMap, index: str, num_snippets,
                 snippets_doc_freq: bool, **kwargs):
        """
        Args:
            conversations: conversations map
            query_map: queries map
            index: index name
            num_snippets: number of wikipedia snippets to retrieve
            snippets_doc_freq: if True the frequency of a term inside snippets will
            be used instead of frequency inside the index
            restrict_to_ascii_letters_and_numbers: if True characted that match '[^a-zA-Z0-9]'
            will be removed. This is done because pyterrier fail to parse queries that contain
            certain symbols.
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._query_map = query_map
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__tag_re = re.compile('<.*?>')
        self.__symbols_re = re.compile('[.,;:\\-_"]')
        self.__restrict_re = re.compile('[^a-zA-Z0-9]')
        self._retry = 5
        self._wait_time = 60
        self._num_snippets = num_snippets
        self._snippets_doc_freq = snippets_doc_freq
        self._index = index

        import spacy
        if spacy.__version__.startswith('2.'):
            self._nlp = spacy.load('en')
        else:
            self._nlp = spacy.load("en_core_web_sm")

    @property
    def name(self) -> str:
        freq = 'snipFreq' if self._snippets_doc_freq else 'docFreq'
        return f'DbPedia-{self._num_snippets}-{freq}'

    @property
    def type(self) -> StepType:
        return StepType.CONVERSATIONALLY_PARALLEL

    def _concat(self, query: Series, queries: DataFrame) -> str:
        qid = query['qid']
        str_query = query['query']
        conv_id, index = self._query_map[qid]
        parts = []
        for oq_id in self._conversations[conv_id][0:index]:
            q = queries[queries['qid'] == oq_id].iloc[0]['query']
            parts.append(q)
        parts.append(str_query)
        return ' '.join(parts)

    def _make_req(self, url, params=None, headers=None):
        count = 0
        err = None
        while count < self._retry:
            if err is not None:
                self.__logger.error(err)
            if count > 0:
                sleep(self._wait_time)
            try:
                resp = requests.get(url,
                                    params=params,
                                    headers=headers)
                if not resp.ok:
                    err = Exception(f'response code: {resp.status_code}')
                    continue
                return resp.json()
            except Exception as ex:
                err = ex
        raise err

    def _spotlight(self, text: str):
        result = self._make_req('https://api.dbpedia-spotlight.org/en/annotate',
                                params={'text': text},
                                headers={'accept': 'application/json'})
        resources = result.get('Resources')
        if resources is None:
            return []
        _list = []
        for res in resources:
            _list.append(res['@surfaceForm'])
        return _list

    def _wiki(self, term: str):
        data = self._make_req('https://en.wikipedia.org/w/api.php',
                              params={
                                  "action": "query",
                                  "format": "json",
                                  "list": "search",
                                  'srsearch': term,
                                  'srlimit': self._num_snippets
                              })
        orig_snippets = data['query']['search']
        snippets = []
        for snip in orig_snippets:
            raw = snip['snippet']
            words = self.__tag_re.sub(' ', raw).split()
            words = list(self.__symbols_re.sub('', w) for w in words)
            snippets.append(words)
        return snippets

    def _score_terms(self, snippets: List[List[str]]):
        lexicon = IndexConf.load_index(self._index).index.getLexicon()
        tot_snippets_len = 0
        all_terms = set()
        for x in snippets:
            all_terms.update(x)
            tot_snippets_len += len(x)

        terms_freq_per_snip = {}
        for i in range(len(snippets)):
            terms_freq_per_snip[i] = collections.Counter(snippets[i])

        scores = []
        for term in all_terms:
            score = 0.0
            for i in range(len(snippets)):
                snip = snippets[i]
                _len = len(snip)
                v = terms_freq_per_snip[i][term] / _len  # freq in i-th snippet
                if v == 0:
                    continue
                v *= (1 / (i + 1))  # RR
                score += v
            try:
                if self._snippets_doc_freq:
                    doc_freq = 0
                    for counter in terms_freq_per_snip.values():
                        doc_freq += counter[term]
                    doc_freq = doc_freq / tot_snippets_len
                else:
                    doc_freq = lexicon[term].getDocumentFrequency()
            except KeyError:
                continue
            score *= log(1 / doc_freq)
            scores.append((term, score))
        scores.sort(key=lambda _x: _x[1], reverse=True)
        return scores

    def _concept_net(self, term: str):
        url_term = term.replace(' ', '_')
        data = self._make_req(f'https://api.conceptnet.io/c/en/{url_term}',
                              headers={'accept': 'application/json'})
        expansion_terms = []
        for edge in data['edges']:
            node = edge['start']
            label = node['label']
            lang = node['language']
            if lang == 'en':
                expansion_terms.append(label)
        return expansion_terms

    def _find_noun_adj_grams(self, text: str):
        tagged = self._nlp(text)
        ngrams = []
        current = []  # todo considerare tutti i possibili n-grams (adiacenti) e fare statistiche
        for w in tagged:
            tag = w.tag_
            if tag == 'NN' or tag == 'JJ':
                current.append(w.lemma_)
            elif len(current) > 0:
                ngrams.append('_'.join(current))
                current.clear()
        if len(current) > 0:
            ngrams.append('_'.join(current))
        return ngrams

    def _rewrite_single(self, query: Series, queries: DataFrame) -> str:
        concatenated_query = self._concat(query, queries)
        entities = self._spotlight(concatenated_query)
        expand_parts = [concatenated_query]
        if len(entities) > 0:
            for entity in entities:
                snippets = self._wiki(entity)
                scored_terms = self._score_terms(snippets)
                expand_parts.extend(x[0] for x in scored_terms[0:10])
        else:
            ngrams = self._find_noun_adj_grams(concatenated_query)
            for n in ngrams:
                expand_parts.extend(self._concept_net(n))
        rewritten = ' '.join(expand_parts)
        rewritten = self.__restrict_re.sub(' ', rewritten)
        return rewritten

    def rewrite(self, queries: DataFrame) -> DataFrame:
        values = queries.apply(self._rewrite_single, axis=1, args=(queries, ))
        return replace_col_with_history('query', values, queries)


if __name__ == '__main__':
    from convSearchPython.dataset.CAsT import cast_dataset

    _queries, _qrels, _conversations, _query_map = cast_dataset(2019)
    rw = DbPediaRewriter(_conversations, _query_map, 'custom', 10, False)
    for _, _query in _queries.iterrows():
        print(f'original:  {_query["query"]}')
        print(f'rewritten: {rw._rewrite_single(_query, _queries)}')
    # wiki = rw._wiki('lung cancer')
    # print(wiki)
    # scores = rw._score_terms(wiki)
    # print(scores)
    # nj = rw._find_noun_adj_grams('tell me about lung cancer. How it spread?')
    # print(nj)
    # cn = rw._concept_net(nj[0])
    # print(cn)
