"""Representation of a run"""
import gc
import gzip
import logging
from datetime import datetime, timedelta
from multiprocessing.pool import Pool
from pathlib import Path
from time import time
from typing import Optional, Tuple, Set, Dict, List

import pandas as pd
from pandas import DataFrame

from convSearchPython.basics import conf
from convSearchPython.pipelines import Pipeline
from convSearchPython.utils import save_results, group_measure, mkdir
from convSearchPython.utils.data_utils import limit_per_query
from convSearchPython.utils.evaluation import evaluate
from convSearchPython.utils.parallel_utils import TimeLogWrapper
from convSearchPython.utils.save_results import save_trec_run

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pkl_log_warn = False


def get_run_dir(name: Optional[str]) -> str:
    workdir = conf.get('GENERAL', 'workdir')
    now = datetime.now().strftime('%Y_%m_%dT%H_%M_%S')
    if name is None:
        name = 'run'
    return f'{workdir}/{name}_{now}'


class Run:
    def __init__(self, name: str = None, parallel_pool: Pool = None):
        """Class that represent a run.

        - name: run name (default None)"""
        self.__run_name = name
        self.__dir = get_run_dir(name)
        self.__pipelines: List[Tuple[str, Pipeline]] = []
        self.__names: Set[str] = set()
        self.__result: Optional[DataFrame] = None
        self.__parallel_pool = parallel_pool

    @property
    def name(self) -> str:
        return self.__run_name

    @property
    def pipelines_names(self) -> Tuple[str]:
        return tuple(self.__names)

    @property
    def was_executed(self) -> bool:
        return self.__result is not None

    def add(self, pipeline: Pipeline, name: str = None, discard_dup=False) -> 'Run':
        """Add the specified pipeline to the run.
        Return itself so calls can be chained.

        :param pipeline: the class of the pipeline
        :param name: if not None, replace the name of the pipeline (must be unique)
        :param discard_dup: if True discard run that are already present instead of
                throwing an exception"""
        if self.was_executed:
            raise Exception('cannot add pipeline to an already executed run')
        if name is None:
            name = pipeline.name
        if name in self.__names:
            if discard_dup:
                logger.warning('discarded duplicate of run %s', name)
                return self
            raise Exception(f'name collision: {name}')
        self.__names.add(name)
        self.__pipelines.append((name, pipeline))
        logger.info('added pipeline %s to the run', name)
        return self

    @staticmethod
    def __execute_loop(pipelines: List[Tuple[str, Pipeline]], queries: DataFrame, parallel_pool):
        size = len(pipelines)
        count = 0
        for name, pipeline in pipelines:
            st = time()
            res = pipeline(queries, parallel_pool=parallel_pool)
            res['name'] = name
            count += 1
            logger.info(f'[%4d/%d] completed search with pipeline %s in %s',
                        count, size, name, timedelta(seconds=(time() - st)))
            gc.collect()  # may help in some cases
            yield res

    def execute(self, queries: DataFrame, qrels: DataFrame = None, limit=0) -> DataFrame:
        """Execute the run and return the result. If qrels is provided the the label column is added to results"""
        if self.was_executed:
            raise Exception('cannot re-execute an already executed run')
        logger.info('Starting pipelines execution %s', 'in a single process' if self.__parallel_pool is None else
                    'with a parallel pool')
        logger.info('%d pipelines to execute', len(self.__pipelines))
        results = list(self.__execute_loop(self.__pipelines, queries, self.__parallel_pool))
        logger.info('all pipelines executed, starting result concatenation')
        st = time()
        self.__result = pd.concat(results)
        logger.info(f'all results concatenated in {timedelta(seconds=(time() - st))}')
        if qrels is not None:
            try:
                self.__result = self.__result.merge(qrels, how='left', on=['qid', 'docno'])
                logger.info('Added label to results')
            except Exception as ex:
                logger.error('Failed to add labels to results')
                logger.error(ex, exc_info=True)
        if limit > 0:
            self.__result = limit_per_query(self.__result, limit)
        return self.__result

    @property
    def result(self) -> Optional[DataFrame]:
        return self.__result

    def save_as_trec(self, save_global=False, save_single=True):
        """Save run in trec_eval format (gzipped)

        - save_global: save the global result (default false)
        - save_single: save single sub-runs (default true)"""
        if not self.was_executed:
            raise Exception('cannot save a run that was not executed')
        mkdir(self.__dir)
        futures = []
        if save_global:
            if self.__parallel_pool is not None:
                f = TimeLogWrapper(save_trec_run, 'run saved in trec format in %s', __name__) \
                    .run(self.__parallel_pool, (self.__result, f'{self.__dir}/run.txt.gz'))
                futures.append(f)
            else:
                st = time()
                save_trec_run(self.__result, f'{self.__dir}/run.txt.gz')
                logger.info('run saved in trec format in %s', str(timedelta(seconds=(time() - st))))
        if save_single:
            sdir = f'{self.__dir}/sub-runs'
            mkdir(sdir)
            for name, result in self.__result.groupby('name'):
                if self.__parallel_pool is not None:
                    f = TimeLogWrapper(save_trec_run, f'sub-run {name} saved in trec format in %s', __name__) \
                        .run(self.__parallel_pool, (result.reset_index(), f'{sdir}/{name}.txt.gz'))
                    futures.append(f)
                else:
                    st = time()
                    save_trec_run(result.reset_index(), f'{sdir}/{name}.txt.gz')
                    logger.info('sub-run %s saved in trec format in %s', name, str(timedelta(seconds=(time() - st))))
        for f in futures:
            f.wait()

    def save_as_csv(self, save_global=False, save_single=True, limit=0):
        """Save run in csv format (gzipped)

        - save_global: save the global result (default false)
        - save_single: save single sub-runs (default true)
        - limit: maximum number of result for query to save, 0 for all (default 0)"""
        if not self.was_executed:
            raise Exception('cannot save a run that was not executed')
        mkdir(self.__dir)
        result = self.__result
        futures = []
        if limit > 0:
            st = time()
            result = limit_per_query(result, limit)
            logger.info('reduced dataset in %s', str(timedelta(seconds=(time() - st))))
        if save_global:
            if self.__parallel_pool is not None:
                f = TimeLogWrapper(getattr(result, 'to_csv'), 'run saved as csv in %s', __name__)\
                    .run(self.__parallel_pool, [f'{self.__dir}/run.csv.gz'], {'index': False})
                futures.append(f)
            else:
                st = time()
                result.to_csv(f'{self.__dir}/run.csv.gz', index=False)
                logger.info('run saved as csv in %s', str(timedelta(seconds=(time() - st))))
        if save_single:
            sdir = f'{self.__dir}/sub-runs'
            mkdir(sdir)
            for name, res in result.groupby('name'):
                if self.__parallel_pool is not None:
                    f = TimeLogWrapper(getattr(res, 'to_csv'), f'sub-run {name} saved as csv in %s', __name__)\
                        .run(self.__parallel_pool, [f'{sdir}/{name}.csv.gz'], {'index': False})
                    futures.append(f)
                else:
                    st = time()
                    res.reset_index().to_csv(f'{sdir}/{name}.csv.gz', index=False)
                    logger.info('sub-run %s saved as csv in %s', name, str(timedelta(seconds=(time() - st))))
        for f in futures:
            f.wait()

    def save_as_pkl(self, save_global=False, save_single=True, limit=0):
        """Save run in pkl format (gzipped)

        - save_global: save the global result (default false)
        - save_single: save single sub-runs (default true)
        - limit: maximum number of result for query to save, 0 for all (default 0)"""
        if not self.was_executed:
            raise Exception('cannot save a run that was not executed')
        global pkl_log_warn
        if not pkl_log_warn:
            logger.warning('Save as pkl: you may need pandas version %s or later to deserialize', str(pd.__version__))
            pkl_log_warn = True
        mkdir(self.__dir)
        result = self.__result
        futures = []
        if limit > 0:
            st = time()
            result = limit_per_query(result, limit)
            logger.info('reduced dataset in %s', str(timedelta(seconds=(time() - st))))
        if save_global:
            if self.__parallel_pool is not None:
                f = TimeLogWrapper(getattr(result, 'to_pickle'), 'run saved as pkl in %s', __name__)\
                    .run(self.__parallel_pool, [f'{self.__dir}/run.pkl.gz'])
                futures.append(f)
            else:
                st = time()
                result.to_pickle(f'{self.__dir}/run.pkl.gz')
                logger.info('run saved as pkl in %s', str(timedelta(seconds=(time() - st))))
        if save_single:
            sdir = f'{self.__dir}/sub-runs'
            mkdir(sdir)
            for name, res in result.groupby('name'):
                if self.__parallel_pool is not None:
                    f = TimeLogWrapper(getattr(res, 'to_pickle'), f'sub-run {name} saved as pkl in %s', __name__)\
                        .run(self.__parallel_pool, [f'{sdir}/{name}.pkl.gz'])
                    futures.append(f)
                else:
                    st = time()
                    res.reset_index().to_pickle(f'{sdir}/{name}.pkl.gz')
                    logger.info('sub-run %s saved as pkl in %s', name, str(timedelta(seconds=(time() - st))))
        for f in futures:
            f.wait()

    def get_measures(self, metrics: list, qrels: DataFrame, query_map: Dict[str, Tuple[str, int]]) \
            -> Dict[str, 'RunMeasure']:
        """Return a dict of RunMeasure objects with measures (all_queries, global_mean, conversation_mean)

        - queries: queries used
        - metrics: list of metrics to compute
        - qrels: qrels DataFrame
        - query_map: query map for the used queries"""
        if not self.was_executed:
            raise Exception('cannot calculate measure for a non executed run')
        d = {}

        st = time()
        all_queries = evaluate(self.__result, metrics, qrels)
        if 'index' in all_queries.columns:
            all_queries.drop('index', axis=1, inplace=True)
        d['all_queries'] = RunMeasure('all_queries', all_queries, self.__dir)
        logger.info(f'generated dataset "all_queries" in {timedelta(seconds=(time() - st))}')

        st = time()
        global_mean = group_measure.name_measure_mean(all_queries)
        d['global_mean'] = RunMeasure('global_mean', global_mean, self.__dir)
        logger.info(f'generated dataset "global_mean" in {timedelta(seconds=(time() - st))}')

        st = time()
        conversation_mean = group_measure.per_conversation_mean(all_queries, query_map)
        d['conversation_mean'] = RunMeasure('conversation_mean', conversation_mean, self.__dir)
        logger.info(f'generated dataset "conversation_mean" in {timedelta(seconds=(time() - st))}')

        st = time()
        parsable_group = group_measure.parsable_measures(all_queries, query_map)
        parsable_path = mkdir(Path(self.__dir, 'parsable_measures'))
        for k, v in parsable_group.items():
            d[f'parsable_{k}'] = RunMeasure(k, v, parsable_path)
        logger.info(f'generated \'parsable datasets\' in {timedelta(seconds=(time() - st))}')

        logger.info('all measures generated')

        return d


class RunMeasure:
    def __init__(self, name: str, measures: DataFrame, run_dir: str):
        """Measure wrapper that provide convenient methods for saving

        - name: name of the measure (ex. mean, conv_mean, all_queries, ...)
        - measures: DataFrame with the measure
        - run_dir: directory of the relative run"""
        self.__name = name
        self.__measures = measures
        self.__dir = run_dir

    @property
    def name(self) -> str:
        return self.__name

    @property
    def measures(self) -> DataFrame:
        return self.__measures

    def save_simple(self, gz=False):
        """Save measure as simple

        - gz: if enable compression (default false)"""
        mkdir(self.__dir)
        if gz:
            with gzip.open(f'{self.__dir}/{self.name}.txt.gz', 'wt') as file:
                save_results.save_simple(self.measures, file)
        else:
            save_results.save_simple(self.measures, f'{self.__dir}/{self.name}.txt')

    def save_as_csv(self, gz=False):
        """Save measures as csv

        - gz: if enable compression (default false)"""
        mkdir(self.__dir)
        ext = 'csv.gz' if gz else 'csv'
        self.measures.to_csv(f'{self.__dir}/{self.name}.{ext}', index=False)

    def save_as_pkl(self, gz=True):
        """Save measures as pkl

        - gz: if enable compression (default true)"""
        global pkl_log_warn
        if not pkl_log_warn:
            logger.warning('Save as pkl: you may need pandas version %s or later to deserialize', str(pd.__version__))
            pkl_log_warn = True
        mkdir(self.__dir)
        ext = 'pkl.gz' if gz else 'pkl'
        self.measures.to_pickle(f'{self.__dir}/{self.name}.{ext}')
