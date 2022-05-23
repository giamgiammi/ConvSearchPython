"""Utilities for saving results"""
import gzip
from typing import Union, List, Tuple, TextIO

import pandas as pd
from tabulate import tabulate

from convSearchPython.utils.data_utils import pyterrier2trec


def save_simple(data: pd.DataFrame, dest: Union[str, TextIO], showindex=False) -> None:
    """Save data as simple txt file with non-highlighted columns

     - data: DataFrame to save
     - dest: either a str path or a file-like object
     - showindex: if True row index is shown"""
    if isinstance(dest, str):
        with open(dest, 'w') as file:
            save_simple(data, file)
    else:
        print(tabulate(data, headers='keys', tablefmt='plain', showindex=showindex), file=dest)


def save_multiple(data: List[Tuple[str, pd.DataFrame]], dest: Union[str, TextIO], save_func=save_simple, showindex=False) -> None:
    """Save multiple data in a txt file

    - data: list of tuples (name, Dataframe) to save
    - dest: either a str path or a file-like object
    - save_func: optional parameter with the function used to save a single DataFrame (default save_simple)
    - showindex: if True row index is shown"""
    if isinstance(dest, str):
        with open(dest, "w") as file:
            save_multiple(data, file)
    else:
        for d in data:
            print('{}:'.format(d[0]), file=dest)
            save_func(d[1], dest, showindex=showindex)
            print('\n', file=dest)


def save_latex_table(data: pd.DataFrame, dest: Union[str, TextIO], showindex=False) -> None:
    """Save data as latex table

     - data: DataFrame to save
     - dest: either a str path or a file-like object
     - showindex: if True row index is shown"""
    if isinstance(dest, str):
        with open(dest, 'w') as file:
            save_latex_table(data, file)
    else:
        print(tabulate(data, headers='keys', tablefmt='latex', showindex=True), file=dest)


def save_trec_run(data: pd.DataFrame, dest: Union[str, TextIO], name=None):
    """Save data as trec run"""
    if isinstance(dest, str):
        if dest.endswith('.gz'):
            with gzip.open(dest, 'wt') as file:
                save_trec_run(data, file, name=name)
        else:
            with open(dest, 'w') as file:
                save_trec_run(data, file, name=name)
    else:
        print(tabulate(pyterrier2trec(data, name=name), tablefmt='plain', showindex=False), file=dest)
