"""
Module for Pipelines that provides a conversational search method

UML Scheme of provided pipelines:
![UML scheme](../imgs/pipelines.svg)
"""
import logging
import math
import random
import string
from abc import abstractmethod, ABC
from enum import Enum, auto
from multiprocessing.pool import Pool
from typing import Optional, NamedTuple, Any, Dict, Callable, Iterable, List

import pandas as pd

from convSearchPython.basics import conf, pyterrier as pt

__pdoc__ = {
    'Pipeline.__call__': True,
    'AbstractParallelPipeline._parallel_run': True,
    'AbstractParallelPipeline._conv_parallel_run': True,
    'Step.__call__': True,
}

from convSearchPython.dataset import Conversations
from convSearchPython.utils.data_utils import queries_conversation_splitter

_INDEX_CACHE = {}


class IndexConf(NamedTuple):
    """
    Utility class for loading indexes configured in config.ini

    Usage:
    ```
    index_conf = Index.load_index('index_name')
    index = index_conf.index  # this is a pyterrier index
    props = index_conf.properties  # this is a dict of index properties

    br = pt.BatchRetrieve(index,
                          wmodel="DirichletLM",
                          properties=props)
    ```
    """
    #: pyterrier index
    index: Any
    #: dict of properties for the index (or None if default apply)
    properties: Optional[Dict[str, Any]]

    @staticmethod
    def load_index(name: str) -> 'IndexConf':
        """
        Load a pyterrier index (and relative properties) from config.

        Raises ValueError if no index is found with provided name.

        Args:
            name: the name of the wanted index inside conf.ini

        Returns:
            An IndexConf objects
        """
        if name in _INDEX_CACHE:
            return _INDEX_CACHE[name]
        logging.getLogger('IndexConf').info('loading index "%s"', name)
        prefix = f'{name}.'
        indexes = conf['INDEXES']
        path = indexes.get(prefix + 'path')
        if path is None:
            raise ValueError(f'index "{name}" not found')
        properties = {}
        for key in indexes.keys():
            if key.startswith(prefix) and (not key == f'{name}.path'):
                prop_key = key[len(prefix):]
                properties[prop_key] = indexes[key]
        if len(properties) == 0:
            properties = None
        index = IndexConf(pt.IndexFactory.of(path), properties)
        _INDEX_CACHE[name] = index
        return index

    @staticmethod
    def add_index(index, properties: Dict[str, Any] = None, name: str = None) -> str:
        """
        Add the provided index to cache
        Raises:
            KeyError: if name is provided and an index with that name already exists
            RuntimeError: if name wasn't provided and the method didn't find a suitable name
            in 100 iterations (safeguard against infinite loop)
        Args:
            index: pyterrier index
            properties: optional dict of properties
            name: optional name for the index

        Returns:
            the name of the added index
        """
        if name is None:
            characters = string.ascii_letters + string.digits
            for i in range(100):
                rdn_name = ''.join(random.choices(characters, k=10))
                if (rdn_name not in _INDEX_CACHE) and (rdn_name not in conf['INDEXES']):
                    name = rdn_name
                    break
            if name is None:
                raise RuntimeError('cannot find suitable name in 100 iteration (safeguard against infinite loop)')
        else:
            if (name in _INDEX_CACHE) or (name in conf['INDEXES']):
                raise KeyError(f'an in dex named "{name}" already exists')

        _INDEX_CACHE[name] = IndexConf(index, properties)
        return name

    @staticmethod
    def remove_index(name: str) -> bool:
        """
        Remove an index from cache.

        If the index wasn't in cache False is returned, else True is returned.

        Args:
            name: name of the index to remove

        Returns:
            True if the index was in cache, False otherwise
        """
        if name in _INDEX_CACHE:
            _INDEX_CACHE.pop(name)
            return True
        return False


class Pipeline(ABC):
    """
    Interface for a Pipeline.

    A pipeline represent a flow of execution comprised of
    various steps, that takes in input a DataFrame of queries
    and give in output a DataFrame of results.

    Pipelines will be instantiated with a dictionary of parameters comprised of:

    - parameters specified in the search configuration file
    - common parameters that are **always** passed and that a pipeline
    can decide to use, if it needs.

    For this reason a pipeline is **required** to have a `**kwargs` arguments
    inside its signature so unused arguments won't cause an error.
    For more information on reserved arguments names or pipeline instantiation in general,
    look at `convSearchPython.search`.

    ## Implementations:
    Every implementation that subclass this object needs to:

    - implement a `name` field (generally using a property) that return an easy-to-parse
    name that represent the entire pipeline
    - implement the `run_on` method, that takes the queries (and optionally a parallel pool),
    process them with the pipeline and return the results conforming with pyterrier data model
    - <ins>Not</ins> override the `__call__` method, that execute run_on internally, by default.

    ## Pipeline-like objects:
    The default way to execute a pipeline is using it as a callable object.
    That means that every object that conform to the constructor and the `__call__` method
    of this object can effectively be used as a pipeline.
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: extra unused parameters
        """

    @property
    @abstractmethod
    def name(self):
        """
        An easy-to-parse name for the pipeline that represents
        the various steps and their configuration

        Generally this should be implemented using a property
        """
        pass

    @abstractmethod
    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None):
        """
        Run the pipeline on a DataFrame of query with, optionally, a parallel pool.

        It is responsibility of the implementing class to decide how to use the parallel pool,
        and check if the needed step are parallelizable (and how).

        Args:
            queries: queries DataFrame
            parallel_pool: multiprocessing Pool

        Returns:
            A DataFrame of results conforming to pyterrier data model
        """
        pass

    def __call__(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        """Same as `run_on`"""
        return self.run_on(queries, parallel_pool)


class AbstractParallelPipeline(Pipeline, ABC):
    """
    Abstract class to help to implement a parallel pipeline.

    It provides two private methods, `_parallel_run` and `conv_parallel_run`
    that can be used by subclasses to parallelize a step of the pipeline.
    """

    @staticmethod
    def _parallel_run(step: Callable[[pd.DataFrame], pd.DataFrame],
                      pool: Pool,
                      data: pd.DataFrame, chunk_size: int = None) -> pd.DataFrame:
        """
        Run a pipeline step on a parallel pool.

        Note: this method do not consider the existence of conversations
        when dividing the data.
        Args:
            step: the callable step, must accept a DataFrame and return a DataFrame
            pool: the multiprocessing parallel pool
            data: the data (queries or results) to split
            chunk_size: (optional) if provided, number of chunk to subdivide data

        Returns:
            The resulting DataFrame
        """
        if chunk_size is not None:
            chunk_size = int(chunk_size)
        else:
            n = getattr(pool, '_processes', None)
            if n is not None and isinstance(n, int):
                chunk_size = math.ceil(len(data) / n)
                if chunk_size < 10:
                    chunk_size = 10
            else:
                chunk_size = 200

        def parts():
            l_data = len(data)
            i = 0
            while i < l_data:
                m = i + chunk_size
                if m > l_data:
                    m = l_data
                yield data.iloc[i:m]
                i = m

        parts_iterable = pool.imap(step, parts())
        return pd.concat(parts_iterable)

    @staticmethod
    def _conv_parallel_run(step: Callable[[pd.DataFrame], pd.DataFrame],
                           pool: Pool,
                           data: pd.DataFrame,
                           conversations: Conversations) -> pd.DataFrame:
        """
        Run a pipeline step on a parallel pool dividing data by conversation.
        Args:
            step: the callable step, must accept a DataFrame and return a DataFrame
            pool: the multiprocessing parallel pool
            data: the data (queries or results) to split
            conversations: conversations structure

        Returns:
            The resulting DataFrame
        """
        conv_iterator = queries_conversation_splitter(data, conversations)
        parts_iterator = pool.imap(step, conv_iterator)
        return pd.concat(parts_iterator)


class StepType(Enum):
    """
    Enum that define the type of a Step instance.
    In particular, it provides insight on the parallelizability
    of a Step.
    """

    #: Step that can be fully parallelized regardless of conversations
    FULLY_PARALLEL = auto()

    #: Step that must be executed sequentially on every conversation
    #: but can be parallelized between different ones
    CONVERSATIONALLY_PARALLEL = auto()

    #: Step that must be executed sequentially on the main process
    SEQUENTIAL = auto()


class Step(ABC):
    """
    Class that represent a generic pipeline step

    Implementation may want to override:

    - `name` to provide a representative name for the step and its parameters
    - `cleanup()` to clean cached objects after the execution
    """

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: extra unused arguments
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Representative name of the Step
        """
        pass

    @property
    @abstractmethod
    def type(self) -> StepType:
        """
        Type of the Step
        """
        pass

    def cleanup(self):
        """
        Clean eventual cached objects after the step execution.

        By default, is a no-op but can be overridden by implementations.
        """
        pass

    @abstractmethod
    def __call__(self, queries_or_results: pd.DataFrame) -> pd.DataFrame:
        """
        Apply this step on a DataFrame
        Args:
            queries_or_results: input DataFrame on witch apply this step

        Returns:
            The resulting DataFrame
        """
        pass


class Rewriter(Step, ABC):
    """
    A Rewriter is a special case of pipeline step that rewrite
    the query formulations passed in input.

    Subclasses must implement the `rewrite` method.
    This class override the `__call__` method, so it uses `rewrite` internally.
    """

    @abstractmethod
    def rewrite(self, queries: pd.DataFrame) -> pd.DataFrame:
        """
        Rewrite input queries.

        The result must conform to the pyterrier data model.
        So the old "query" column must be renamed "query_n" where
        n is the number of the formulation (starting with 0).

        Example: if the input already contains a "query_0" column,
        the "query" column must be renamed "query_1" and a new
        "query" column must be added with rewritten queries.

        Args:
            queries: input queries DataFrame

        Returns:
            DataFrame with rewritten queries
        """
        pass

    def __call__(self, queries: pd.DataFrame) -> pd.DataFrame:
        return self.rewrite(queries)


class Reranker(Step, ABC):
    """
    A Rewriter is a special case of pipeline step that rerank
    the documents and produce a newly ordered results DataFrame.

    Subclasses must implement the `rerank` method.
    This class override the `__call__` method, so it uses `rerank` internally.
    """

    @abstractmethod
    def rerank(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Rewrite input queries.

        The result must conform to the pyterrier data model.
        So the content of the "score" column must be updated.

        It is responsibility of the rewriter to check if multiple
        conversations were passed as inputs and act according, and
        to sort the resulting DataFrame before returning it.

        Args:
            results: input results DataFrame

        Returns:
            DataFrame reordered and with updated scores
        """
        pass

    def __call__(self, results: pd.DataFrame) -> pd.DataFrame:
        return self.rerank(results)


class Model(Step):
    """
    Wrapper for BatchRetrieve

    Note: mu control, when wmodel is DirichletLM, is renamed to c for convenience
    """

    def __init__(self, wmodel: str, controls: Dict[str, Any], index: str, metadata: List[str], **kwargs):
        """
        Args:
            wmodel: model name (ex. BM25)
            controls: model controls
            index: index name
            metadata: metadata list
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._wmodel = wmodel
        controls = controls.copy()
        if wmodel == 'DirichletLM' and 'mu' in controls:
            controls['c'] = controls.pop('mu')
        self._num_results = 1000
        if 'num_results' in controls:
            self._num_results = controls.pop('num_results')
        self._controls = controls
        self._index = index
        self._metadata = metadata

        self._br = None

    @property
    def wmodel(self):
        return self._wmodel

    @property
    def controls(self):
        return self._controls.copy()

    @property
    def name(self) -> str:
        parts = [self.wmodel]
        controls = {**self._controls}
        if self._num_results != 1000:
            controls['n'] = self._num_results
        for key, value in controls.items():
            parts.append(f'{key}={value}')
        return '-'.join(parts)

    @property
    def type(self) -> StepType:
        return StepType.FULLY_PARALLEL

    def cleanup(self):
        self._br = None

    def _batch_retrieve(self):
        index_conf = IndexConf.load_index(self._index)
        return pt.BatchRetrieve(index_conf.index,
                                wmodel="DirichletLM",
                                controls=self._controls,
                                properties=index_conf.properties,
                                metadata=self._metadata,
                                num_results=self._num_results)

    def __call__(self, queries: pd.DataFrame) -> pd.DataFrame:
        if self._br is None:
            self._br = self._batch_retrieve()
        return self._br(queries)

    def __getstate__(self):
        return self._wmodel, self._controls, self._index, self._metadata, self._num_results

    def __setstate__(self, state):
        self._wmodel, self._controls, self._index, self._metadata, self._num_results = state
        self._br = None


class RM3(Step):
    """
    Wrapper for RM3
    """

    def __init__(self, index: str, fb_terms: int = 20, fb_docs: int = 20, fb_lambda: float = 0.5, **kwargs):
        """
        Args:
            index: index name
            fb_terms: number of terms
            fb_docs: number of docs
            fb_lambda: rm3 parameter
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._index = index
        self._fb_terms = fb_terms
        self._fb_docs = fb_docs
        self._fb_lambda = fb_lambda

        self._rm3 = None

    @property
    def fb_terms(self):
        return self._fb_terms

    @property
    def fb_docs(self):
        return self._fb_docs

    @property
    def fb_lambda(self):
        return self._fb_lambda

    @property
    def name(self):
        return f'rm3-t{self.fb_terms}-d{self.fb_docs}-l{self.fb_lambda}'

    @property
    def type(self) -> StepType:
        return StepType.FULLY_PARALLEL

    def cleanup(self):
        self._rm3 = None

    def _get_rm3(self):
        index_conf = IndexConf.load_index(self._index)
        return pt.rewrite.RM3(index_conf.index,
                              fb_terms=self._fb_terms,
                              fb_docs=self._fb_docs,
                              fb_lambda=self._fb_lambda)

    def __call__(self, results: pd.DataFrame) -> pd.DataFrame:
        if self._rm3 is None:
            self._rm3 = self._get_rm3()
        return self._rm3(results)

    def __getstate__(self):
        return self._fb_terms, self._fb_docs, self._fb_lambda, self._index

    def __setstate__(self, state):
        self._fb_terms, self._fb_docs, self._fb_lambda, self._index = state
        self._rm3 = None


class ChainPipeline(AbstractParallelPipeline):
    """
    Pipeline that combine already constructed steps.

    While the main use is to combine objects that subclass `Step`,
    this class will work with any Step-like object that act as
    ```Callable[[DataFrame], DataFrame]```.
    """

    def __init__(self, steps: Iterable[Step], name: str,
                 conversations: Conversations, **kwargs):
        """
        Args:
            steps: iterable of Step or Step-like objects
            name: easy-to-parse pipeline name
            conversations: (reserved argument) conversations structure
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._name = name
        self._conversations = conversations
        self._steps = list(steps)
        for s in steps:
            if not callable(s):
                raise ValueError(f'step {s} is not callable')

    @property
    def steps(self) -> List[Step]:
        """List of Step objects contained in this pipeline"""
        return self._steps.copy()

    @property
    def name(self) -> str:
        """Easy-to-parse pipeline name"""
        return self._name

    def run_on(self, queries: pd.DataFrame, parallel_pool: Optional[Pool] = None) -> pd.DataFrame:
        """
        Run the pipeline.

        If a parallel pool is provided, then the steps are executed according to the following rules:

        - If a step is of type `StepType.SEQUENTIAL`, then it's executed sequentially
        outside the parallel pool
        - If a step is of type `StepType.CONVERSATIONALLY_PARALLEL`, then it's
        executed inside the parallel pool, dividing the input queries by conversation
        - If neither of that applies, the step is executed inside the parallel pool
        without regarding of the conversations

        Args:
            queries: input queries DataFrame
            parallel_pool: optional parallel pool

        Returns:
            The resulting DataFrame
        """
        data = queries

        if parallel_pool is None:
            for step in self._steps:
                data = step(data)
        else:
            for step in self._steps:
                type = getattr(step, 'type', StepType.FULLY_PARALLEL)
                if type == StepType.SEQUENTIAL:
                    data = step(data)
                elif type == StepType.CONVERSATIONALLY_PARALLEL:
                    data = self._conv_parallel_run(step, parallel_pool, data, self._conversations)
                else:
                    data = self._parallel_run(step, parallel_pool, data)

                cleanup = getattr(step, 'cleanup', None)
                if callable(cleanup):
                    cleanup()

        return data


if __name__ == '__main__':
    index_conf = IndexConf.load_index('custom')
    print(index_conf)
