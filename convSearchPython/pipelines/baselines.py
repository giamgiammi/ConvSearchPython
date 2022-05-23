"""Collection of baselines runs"""
from abc import ABC
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from convSearchPython.dataset import Conversations
from convSearchPython.pipelines import AbstractParallelPipeline, Model, RM3
from convSearchPython.searching.rewriter.concat_query import ConcatQueryRewriter
from convSearchPython.searching.rewriter.context_query import ContextQueryRewriter
from convSearchPython.searching.rewriter.first_query import FirstQueryRewriter


class PlainPipeline(AbstractParallelPipeline):
    """
    DirichletLM pipeline with no query rewriting and optional RM3

    Configure with:

    ```
    plain = convSearchPython.pipelines.baselines.PlainPipeline
    plain.mu = 2500
    plain.rm3 = false
    plain.fb_terms = 20
    plain.fb_docs = 20
    plain.fb_lambda = 0.5
    ```
    """
    def __init__(self, index: str, metadata: list, rm3=False, mu=2500, fb_terms=20, fb_docs=20,
                 fb_lambda=0.5, **kwargs):
        """
        Args:
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._index = index
        self._metadata = metadata
        self._fb_terms = fb_terms
        self._fb_docs = fb_docs
        self._fb_lambda = fb_lambda
        self._apply_rm3 = rm3
        self._mu = mu

        steps = [Model('DirichletLM', {'c': self._mu}, self._index, self._metadata)]
        if self._apply_rm3:
            steps.append(RM3(self._index, self._fb_terms, self._fb_docs, self._fb_lambda))
            steps.append(steps[0])
        self._steps = steps

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        results = queries

        if parallel_pool is None:
            for step in self._steps:
                results = step(results)
            return results

        for step in self._steps:
            results = self._parallel_run(step, parallel_pool, results)
        return results

    @property
    def name(self):
        """DLM_none_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[0].name}_none_{self._steps[1].name}'
        return f'{self._steps[0].name}_none_none'


class AbstractDLMConversationalPipeline(PlainPipeline, ABC):
    """
    Abstract class for conversational pipeline with DLM and optional RM3
    """
    def __init__(self, conversations: Conversations, **kwargs):
        """
        Args:
            conversations: conversations structure
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        self._conversations = conversations

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        results = queries

        if parallel_pool is None:
            for step in self._steps:
                results = step(results)
            return results

        for step in self._steps:
            results = self._conv_parallel_run(step, parallel_pool, results, self._conversations)

        return results


class FirstQueryPipeline(AbstractDLMConversationalPipeline):
    """
    DirichletLM pipeline with first query rewriting and optional RM3

    Configure with:

    ```
    first = convSearchPython.pipelines.baselines.FirstQueryPipeline
    first.mu = 2500
    first.rm3 = false
    first.fb_terms = 20
    first.fb_docs = 20
    first.fb_lambda = 0.5
    ```
    """

    def __init__(self, variant='no-repeat', **kwargs):
        """
        Args:
            variant: rewriter variant. Check `convSearchPython.searching.rewriter.first_query.FirstQueryRewriter`
            for more information.
            conversations: conversations structure
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        self._steps = [FirstQueryRewriter(variant=variant, **kwargs)] + self._steps

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        results = queries

        if parallel_pool is None:
            for step in self._steps:
                results = step(results)
            return results

        for step in self._steps:
            results = self._conv_parallel_run(step, parallel_pool, results, self._conversations)
        return results

    @property
    def name(self):
        """DLM_FirstQ_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'


class ContextQueryPipeline(AbstractDLMConversationalPipeline):
    """
    DirichletLM pipeline with context query rewriting and optional RM3
    """
    def __init__(self, variant='no-repeat', **kwargs):
        """
        Args:
            variant: rewriter variant. Check `convSearchPython.searching.rewriter.context_query.ContextQueryRewriter`
            for more information.
            conversations: conversations structure
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        self._steps = [ContextQueryRewriter(variant=variant, **kwargs)] + self._steps

    @property
    def name(self):
        """DLM_ContextQ_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'


class Coreference1Pipeline(AbstractDLMConversationalPipeline):
    """
    DirichletLM pipeline with Coref1 (alennlp) query rewriting and optional RM3
    """
    def __init__(self, cache_dir: Union[str, Path], allow_cuda=True, autosave=True, **kwargs):
        """
        Args:
            cache_dir: the cache dir path (None disable cache)
            allow_cuda: if True cuda is used if available
            autosave: if True the cache is automatically saved
            conversations: conversations structure
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        from convSearchPython.searching.rewriter.coref_query import AllennlpCoreferenceQueryRewriter
        self._coref = AllennlpCoreferenceQueryRewriter(
            cache_dir=cache_dir, autosave=autosave, allow_cuda=allow_cuda, **kwargs)

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        return super().run_on(self._coref(queries), parallel_pool)

    @property
    def name(self):
        """DLM_Coref1_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[0].name}_{self._coref.name}_{self._steps[1].name}'
        return f'{self._steps[0].name}_{self._coref.name}_none'


class Coreference2Pipeline(AbstractDLMConversationalPipeline):
    """
    DirichletLM pipeline with Coref2 (neuralcoref) query rewriting and optional RM3
    """
    def __init__(self, nlp: str = 'default', **kwargs):
        """
        Args:
            nlp: nlp variant. Check `convSearchPython.searching.rewriter.neuralcoref_coref_query.NeuralCorefRewriter`
            for more information
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        from convSearchPython.searching.rewriter.neuralcoref_coref_query import NeuralCorefRewriter
        self._steps = [NeuralCorefRewriter(nlp=nlp, **kwargs)] + self._steps

    @property
    def name(self):
        """DLM_Coref2_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'


class PlainBM25Pipeline(AbstractParallelPipeline):
    """
    BM25 pipeline with no query rewriting and optional RM3
    """
    def __init__(self, index: str, metadata: list, rm3=False, c=0.75, fb_terms=20, fb_docs=20,
                 fb_lambda=0.5, **kwargs):
        """
        Args:
            index: index name
            metadata: metadata to retrieve
            rm3: if to apply RM3
            c: BM25 c parameter
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._c = c
        self._index = index
        self._metadata = metadata
        self._fb_terms = fb_terms
        self._fb_docs = fb_docs
        self._fb_lambda = fb_lambda
        self._apply_rm3 = rm3

        steps = [Model('BM25', {'c': self._c}, self._index, self._metadata)]
        if self._apply_rm3:
            steps.append(RM3(self._index, self._fb_terms, self._fb_docs, self._fb_lambda))
            steps.append(steps[0])
        self._steps = steps

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None):
        results = queries

        if parallel_pool is None:
            for step in self._steps:
                results = step(results)
            return results

        for step in self._steps:
            results = self._parallel_run(step, parallel_pool, results)
        return results

    @property
    def name(self):
        """BM25_none_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[0].name}_none_{self._steps[1].name}'
        return f'{self._steps[0].name}_none_none'


class ConcatQueryPipeline(AbstractDLMConversationalPipeline):
    """
    Pipeline that rewrite queries with the concatenation of
    all utterance in the conversation up to that point
    """
    def __init__(self, **kwargs):
        """
        Args:
            conversations: conversations structure
            index: index to use (as RetrieveMetadata enum)
            metadata: metadata to retrieve
            rm3: if to apply RM3
            mu: mu parameter of dirichlet language model
            fb_terms: RM3 term number
            fb_docs: RM3 number of docs
            fb_lambda: RM3 lambda
            **kwargs: same arguments of `PlainPipeline`
        """
        super().__init__(**kwargs)
        self._steps = [ConcatQueryRewriter(**kwargs)] + self._steps

    @property
    def name(self):
        """DLM_ConcatQ_[rm3]"""
        if self._apply_rm3:
            return f'{self._steps[1].name}_{self._steps[0].name}_{self._steps[2].name}'
        return f'{self._steps[1].name}_{self._steps[0].name}_none'
