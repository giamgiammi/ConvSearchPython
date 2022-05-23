"""
This module contains a pipeline based on [Incoporating Query Context into a BERT Re-ranker](https://trec.nist.gov/pubs/trec28/papers/mpii.C.pdf)
"""
from typing import Optional

import pandas as pd
from multiprocess.pool import Pool

from convSearchPython.dataset import Conversations
from convSearchPython.pipelines.baselines import PlainBM25Pipeline
from convSearchPython.searching.rewriter.dbpedia import DbPediaRewriter


class DBPediaPipeline(PlainBM25Pipeline):
    """
    Pipeline that implements a BM25 search with query
    expansion from Wikipedia and ConceptNet
    """

    def __init__(self, conversations: Conversations, num_snippets=10, snippets_doc_freq=False, **kwargs):
        """
        Args:
            num_snippets: number of wikipedia snippets to use for expansion
            snippets_doc_freq: if True the frequency of a term inside snippets will
            index: index name
            metadata: metadata to retrieve
            rm3: if to apply RM3
            c: BM25 c parameter
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: extra arguments
        """
        super().__init__(**kwargs)
        self._num_snippets = num_snippets
        self._snippets_doc_freq = snippets_doc_freq
        self._conversations = conversations
        kwargs['conversations'] = conversations

        self._steps = [DbPediaRewriter(num_snippets=num_snippets, snippets_doc_freq=snippets_doc_freq, **kwargs)] \
                      + self._steps

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None):
        data = queries

        if parallel_pool is None:
            for step in self._steps:
                data = step(data)
            return data

        for step in self._steps:
            data = self._conv_parallel_run(step, parallel_pool, data, self._conversations)
        return data

    @property
    def name(self):
        """BM25_DBPedia-numSnippets-{doc_freq}_{rm3}_none"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'
