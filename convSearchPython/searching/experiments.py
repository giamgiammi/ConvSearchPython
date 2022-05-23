"""Collection of function to make experiments"""
from convSearchPython.basics import pyterrier as pt
from pyterrier.transformer import TransformerBase
from pandas import DataFrame
from typing import Tuple, Iterable
import pyterrier.measures as measures


def ml_experiment(named_methods: Iterable[Tuple[str, TransformerBase]], queries: DataFrame, qrels: DataFrame) -> DataFrame:
    """Do an experiments returning the measures used in meleAt"""
    methods = [x[1] for x in named_methods]
    names = [x[0] for x in named_methods]
    return pt.Experiment(
        methods,
        queries, qrels,
        eval_metrics=[measures.AP, measures.nDCG @ 3, measures.P @ 1, measures.P @ 3, measures.RR, measures.R @ 200],
        names=names,
        perquery=True,
        filter_by_qrels=True,
    )

