"""Evaluation utilities"""
import pandas as pd
from pandas import DataFrame
from pyterrier.utils import Utils

from convSearchPython.utils.sort import key_str_with_int


def evaluation_key(data: pd.Series) -> pd.Series:
    """Apply padding to qid"""
    if data.name != 'qid':
        return data
    data = pd.Series((key_str_with_int(x) for x in data))
    return data


def evaluate(run: DataFrame, metrics: list, qrels: DataFrame) -> DataFrame:
    """Evaluate the run with specified metrics and qrels"""
    rows = []
    names = pd.unique(run['name'])
    for name in names:
        named_run = run[run['name'] == name]
        measure_dict = Utils.evaluate(named_run, qrels, metrics, True)
        for qid in measure_dict:
            for measure in measure_dict[qid]:
                rows.append([name, qid, measure, measure_dict[qid][measure]])
    return pd.DataFrame(rows, columns=['name', 'qid', 'measure', 'value'])\
        .sort_values(['name', 'qid'], key=evaluation_key)\
        .reset_index()
