"""Historical query expansion (HQExp) Query and Answer Expansion from Conversation History(Jheng-Hong Yang et al)
based on code at https://github.com/castorini/chatty-goose/blob/ac9066c0aa54b9d0b6c0fb3e0cd5a46d8b64f4c1/chatty_goose/cqr/hqe.py"""
import collections
import re
from typing import Dict, Tuple, Optional

import pandas as pd
import pyterrier as pt

from convSearchPython.pipelines import Rewriter, StepType, IndexConf
from convSearchPython.utils.data_utils import replace_col_with_history


def _pre_process(text):
    text = re.sub(
        "[-‐‑‒–—―%\\[\\]:()/	]",  # changed to wat I think was the intention
        # u"-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t",  # original version
        _space_extend,
        text,
    )
    text = text.strip(" \n")
    text = re.sub("\\s+", " ", text)
    return text


def _process(parsed_text):
    output = {
        "word": [],
        "lemma": [],
        "pos": [],
        "pos_id": [],
        "ent": [],
        "ent_id": [],
        "offsets": [],
        "sentences": [],
    }

    for token in parsed_text:
        output["word"].append(_str(token.text))
        pos = token.tag_
        output["pos"].append(pos)

    return output


def _space_extend(matchobj):
    return "".join((" ", matchobj.group(0), " "))


def _str(s):
    """ Convert PTB tokens to normal tokens """
    s = s.lower()

    if s == "-lrb-":
        s = "("
    elif s == "-rrb-":
        s = ")"
    elif s == "-lsb-":
        s = "["
    elif s == "-rsb-":
        s = "]"
    elif s == "-lcb-":
        s = "{"
    elif s == "-rcb-":
        s = "}"
    return s


def _query_expansion(key_word_list: Dict[int, list], start_turn: int, end_turn: int):
    query_expansion = []
    if start_turn < 0:
        start_turn = 0
    for turn in range(start_turn, end_turn + 1):
        for word in key_word_list[turn]:
            query_expansion.append(word)
    return " ".join(query_expansion)


class HQERewriter(Rewriter):
    """
    HQE Rewriter based on https://github.com/castorini/chatty-goose/blob/ac9066c0aa54b9d0b6c0fb3e0cd5a46d8b64f4c1/chatty_goose/cqr/hqe.py
    """
    def __init__(self, query_map: Dict[str, Tuple[str, int]], rs: float,
                 rq: float, theta: float,
                 index: str, model: str = 'BM25', c: float = '0.75', **kwargs):
        """
        Args:
            query_map: query map structure
            rs: threshold for session keywords
            rq: threshold for query keywords
            theta: threshold for non-ambiguous query
            index: index name
            model: model to use
            c: model tuning
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._query_map = query_map
        self._rs = rs
        self._rq = rq
        self._theta = theta

        self._index = index
        self._model = model
        self._c = c

        self._transformer = None

        self._current_conv: Optional[str] = None
        self._current_turn = -1

        self._q_key_word_list = collections.defaultdict(list)
        self._q_subkey_word_list = collections.defaultdict(list)

        import spacy
        if spacy.__version__.startswith('2.'):
            self._nlp = spacy.load('en')
        else:
            self._nlp = spacy.load("en_core_web_sm")

    @property
    def name(self) -> str:
        return f'HQE-{self._model}-{self._c}' \
               f'-rs{self._rs}' \
               f'-rq{self._rq}' \
               f'-theta{self._theta}'

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

    def calc_word_score(self, query):
        nlp_query = self._nlp(_pre_process(query))
        proc_query = _process(nlp_query)
        query_words = proc_query["word"]
        proc_query["score"] = []

        for query_word in query_words:
            hits = self._transformer.search(query_word)
            try:
                score = float(hits.iloc[0]['score'])
                proc_query["score"].append(score)
            except:
                proc_query["score"].append(-1)

        return proc_query

    def key_word_extraction(self, query):
        proc_query = self.calc_word_score(query)

        for i, word in enumerate(proc_query["word"]):
            if ("NN" in proc_query["pos"][i]) or ("JJ" in proc_query["pos"][i]):
                if proc_query["score"][i] >= self._rs:
                    self._q_key_word_list[self._current_turn].append(word)
                if (proc_query["score"][i] >= self._rq) & (proc_query["score"][i] < self._rs):
                    self._q_subkey_word_list[self._current_turn].append(word)

    def rewrite_single(self, data: pd.Series) -> str:
        query = data['query']
        conv = self._query_map[data['qid']][0]
        if self._current_conv != conv:
            self._current_conv = conv
            self._q_key_word_list.clear()
            self._q_subkey_word_list.clear()
            self._current_turn = -1

        self._current_turn += 1

        self.key_word_extraction(query)
        if self._current_turn > 0:
            hits = self._transformer.search(query)
            key_word = _query_expansion(self._q_key_word_list, 0, self._current_turn)
            subkey_word = ""
            if len(hits) == 0 or hits.iloc[0]["score"] <= self._theta:
                # end_turn = self._current_turn + 1
                subkey_word = _query_expansion(self._q_subkey_word_list, 0, self._current_turn)
            query = " ".join((key_word, subkey_word, query))

        return query

    def rewrite(self, queries: pd.DataFrame) -> pd.DataFrame:
        return replace_col_with_history(
            'query',
            queries.apply(self.rewrite_single, axis=1),
            queries
        )
