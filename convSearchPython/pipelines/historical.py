"""
This module contains a collection of pipelines that make use of two algorithms (HQE and HAE) described in
[Query and Answer Expansion from Conversation History (Jheng-Hong Yang et al)](https://trec.nist.gov/pubs/trec28/papers/CFDA_CLIP.C.pdf).

### HQE: **Historical Query Expansion**
This is a query rewriting method that extract relevant keywords from previous
utterances in the same conversation and add them to the current query.

It's governed by:

- \\( r_s \\): hyperparameter for words that are relevant to the session (or conversation)
- \\( r_q \\): hyperparameter for words that are relevant to their containing query
- \\( \\theta \\): hyperparameter for queries that are non-ambiguous

Notes on v2:
> This project contain two implementation for HQE, the first (implemented in `convSearchPython.searching.rewriter.historical_query`)
is a custom implementation based only on what the paper describes. The second (`convSearchPython.searching.rewriter.historical_query_v2`)
is based on code at [this link](https://github.com/castorini/chatty-goose/blob/ac9066c0aa54b9d0b6c0fb3e0cd5a46d8b64f4c1/chatty_goose/cqr/hqe.py).

The *v2* parameter of HQE pipelines determines which implementation is used

### HAE: **Historical Answer Expansion**
This is a reranking method based on [BERT](https://arxiv.org/abs/1810.04805v2)

It's governed by:

- \\( \\lambda \\): decay factor for lower previous answers' weight
- \\( k \\): number of documents to return for every query

-----
Notes: the reranking phase support GPU acceleration through CUDA and multiple
GPU computing. Check the module `ranking.bert.bert_pre_marco` for the available
implementation.
"""

from convSearchPython.dataset import QueryMap
from convSearchPython.pipelines import *
from convSearchPython.pipelines.baselines import PlainBM25Pipeline
from convSearchPython.ranking.historical_answer import HAEReranker
from convSearchPython.searching.rewriter.historical_query import HistoricalQueryRewriter
from convSearchPython.searching.rewriter.historical_query_v2 import HQERewriter
from convSearchPython.utils import parse_bool


class BM25TokenPipeline(PlainBM25Pipeline):
    """
    ***This is a test pipeline.***

    Its purpose is to provide an indication for hyperparameters tuning (by saving and analyzing the run).

    It sends every single query words as a query itself.
    """

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        """
        Run the pipeline

        Split every query by its words and search them as individual queries.
        Args:
            queries: the queries DataFrame
            parallel_pool: optional multiprocessing pool

        Returns:
            Concatenated DataFrame of results
        """
        parts = []
        for _, row in queries.iterrows():
            tokens = str(row['query']).split()
            for t in tokens:
                new_row = row.copy()
                new_row['query'] = t
                new_row['qid'] = str(new_row['qid']) + '_' + t
                parts.append(new_row)
        queries = pd.DataFrame(parts)
        del parts
        return super().run_on(queries, parallel_pool)


class HistoricalQueryExpansionPipeline(PlainBM25Pipeline):
    """Historical query expansion pipeline"""
    def __init__(self, query_map: QueryMap, conversations: Conversations,
                 rs=10, rq=15, theta=30, v2=True, **kwargs):
        """
        Args:
            rs: threshold for session keywords
            rq: threshold for query keywords
            theta: threshold for non ambiguous query
            v2: if to use hqe v2
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
        self._rs = rs
        self._rq = rq
        self._theta = theta
        self._query_map = query_map
        self._conversations = conversations
        self._v2 = parse_bool(v2)

        if self._v2:
            hqe = HistoricalQueryRewriter(session_relevant_threshold=rs,
                                          query_relevant_threshold=rq,
                                          non_ambiguous_threshold=theta,
                                          query_map=query_map,
                                          **kwargs)
        else:
            hqe = HQERewriter(rs=rs, rq=rq, theta=theta, query_map=query_map, **kwargs)
        self._steps: list = [hqe] + self._steps

    @property
    def name(self):
        """BM25_HQE_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        results = queries

        if parallel_pool is None:
            for step in self._steps:
                results = step(results)
            return results

        for step in self._steps:
            results = self._conv_parallel_run(step, parallel_pool, results, self._conversations)

        return results


class HistoricalQueryAndAnswerExpansionPipeline(HistoricalQueryExpansionPipeline):
    """Pipeline that use HQExp for rewriting and HAExp for reranking"""
    def __init__(self, _lambda: float = 2, k: int = 100,
                 **kwargs):
        """
        Args:
            _lambda: decay factor
            k: cutoff for returned docs
            rs: threshold for session keywords
            rq: threshold for query keywords
            theta: threshold for non-ambiguous query
            v2: if to use hqe v2
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
        self._lambda = _lambda
        self._k = k
        self._steps.append(HAEReranker(_lambda=_lambda, k=k, **kwargs))

    @property
    def name(self):
        """BM25_HQE_[rm3]_HAE"""
        if self._apply_rm3:
            i = 3
        else:
            i = 2
        return super().name + self._steps[i].name
