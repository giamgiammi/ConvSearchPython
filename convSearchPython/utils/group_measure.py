"""Utilities for grouping measure/results"""
from typing import Dict, Tuple

import pandas as pd


def name_measure_mean(measures: pd.DataFrame) -> pd.DataFrame:
    """Return the data grouped by name, measure and aggregated with mean"""
    return measures.groupby(["name", "measure"]).aggregate("mean").reset_index(drop=True)


def conversation_measure(measures: pd.DataFrame, query_map: Dict[str, Tuple[str, int]]) -> pd.DataFrame:
    """Return a copy of measures with the added column conversation"""
    measures = measures.copy()
    conv = []
    for i, row in measures.iterrows():
        conv.append(query_map[row['qid']][0])
    measures['conversation'] = conv
    return measures


def aggregate_per_conversation(measures: pd.DataFrame, query_map: Dict[str, Tuple[str, int]]) -> pd.DataFrame:
    """Output the dataset aggregated per conversation with mean"""
    measures = conversation_measure(measures, query_map)
    return measures\
        .groupby(['name', 'conversation', 'measure'])\
        .aggregate('mean')\
        .reset_index(drop=True)


def per_conversation_mean(measures: pd.DataFrame, query_map: Dict[str, Tuple[str, int]]) -> pd.DataFrame:
    """Output the mean of the per-conversation mean of the measures"""
    measures = conversation_measure(measures, query_map)
    return measures\
        .groupby(['name', 'conversation', 'measure'])\
        .aggregate('mean')\
        .groupby(['name', 'measure'])\
        .aggregate('mean')\
        .reset_index()


def parsable_measures(measures: pd.DataFrame, query_map: Dict[str, Tuple[str, int]]) -> Dict[str, pd.DataFrame]:
    """Return a dict measure -> dataframe where dataframe are created to be
    easily parsed from csv. Requires in input conversation_mean as outputted
    by per_per_conversation_mean"""
    measures = conversation_measure(measures, query_map)
    d = {}
    for measure, m_group in measures.groupby('measure'):
        d[str(measure)] = m_group.reset_index()[['name', 'conversation', 'qid', 'value']]
    return d
