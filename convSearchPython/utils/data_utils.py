"""Utilities for data manipulation"""
import re
from typing import Dict, List, Generator, Union, Iterable, Any, Set, Callable

import pandas as pd
from pandas import DataFrame

from convSearchPython.dataset import Conversations
from convSearchPython.utils.evaluation import evaluation_key


def measures_pivot_table(data: pd.DataFrame) -> pd.DataFrame:
    """Manipulate a dataFrame so the measures are in columns"""
    return data.pivot(index='name', columns='measure', values='value')


def pyterrier2trec(run: pd.DataFrame, name=None) -> pd.DataFrame:
    """Convert a pyterrier run to the trec equivalent"""
    run['second'] = ['Q0'] * len(run.index)
    if name is not None:
        run['name'] = [name] * len(run.index)
    else:
        if 'name' not in run.columns:
            raise Exception('needs to specify name')

    return run[['qid', 'second', 'docno', 'rank', 'score', 'name']]


def limit_per_query(data: DataFrame, limit: int) -> DataFrame:
    """Limit the number of result per query"""
    parts = []
    for _, group in data.groupby(['qid', 'name']):
        group = group.reset_index()
        if len(group) <= limit:
            parts.append(group)
        else:
            parts.append(group[:limit])
    return pd.concat(parts)


def limit_per_qid(data: DataFrame, limit: int) -> DataFrame:
    """Limit the number of result per query"""
    parts = []
    for _, group in data.groupby('qid'):
        group = group.reset_index()
        if len(group) <= limit:
            parts.append(group)
        else:
            parts.append(group[:limit])
    return pd.concat(parts)


def guess_type(value: str, no_bool=False,
               no_str=False, no_float=False,
               allow_quote_for_str=False) -> Union[str, int, float, bool]:
    """Try to guess the type of value (str, int, float or bool) and return it according"""
    value = str(value)
    if allow_quote_for_str:
        v = value.strip()
        if v.startswith('"') and v.endswith('"'):
            return v[1:-1]
    if not no_bool:
        low_val = value.lower().strip()
        if low_val == 'true':
            return True
        if low_val == 'false':
            return False
    try:
        return int(value)
    except ValueError:
        if no_float:
            if no_str:
                raise
            return value
        try:
            return float(value)
        except ValueError:
            if no_str:
                raise
            return value


def queries_conversation_splitter(queries: DataFrame, conversation: Conversations) \
        -> Generator[DataFrame, None, None]:
    for conv in conversation.values():
        q = queries[queries['qid'].isin(conv)]
        q.reset_index(inplace=True, drop=True)
        yield q


def replace_col_with_history(name: str, values: Iterable[Any], df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Replace (or add) a column with specified values and name to the DataFrame
    If the column already exists, save the old one as `name_0`.
    If name_0 is already presents it will be renamed as name_1 (and so on...).

    Args:
        name: name of the column
        values: column values
        df: DataFrame to edit
        inplace: if True, edit the DataFrame, else return a new one

    Returns:
        The edited DataFrame or a new one
    """
    if not inplace:
        df = df.copy()

    history: Set[str] = set(x for x in df.columns if x == name or x.startswith(f'{name}_'))
    if name not in history:
        df[name] = values
        return df

    history.remove(name)
    suffixes = []
    for h in history:
        parts = h.split('_')
        if len(parts) != 2:
            continue
        s = parts[1]
        try:
            suffixes.append(int(s))
        except ValueError:
            continue
    suffixes.sort(reverse=True)
    for i in suffixes:
        df[f'{name}_{i+1}'] = df[f'{name}_{i}']
    df[f'{name}_0'] = df[name]
    df[name] = values
    return df


def sort_results_update_rank(results: DataFrame, conversations: Conversations) -> DataFrame:
    """
    Sort results by score and update rank

    Args:
        results: DataFrame to sort
        conversations: conversations structure

    Returns:
        A new sorted DataFrame
    """
    parts = []
    for qid_list in conversations.values():
        for qid in qid_list:
            d = results[results['qid'] == qid].copy()
            d.sort_values(['qid', 'score'], key=evaluation_key, inplace=True)
            d['rank'] = range(len(d))
            parts.append(d)
    return pd.concat(parts)


def split_comma_not_quotes(value: str) -> List[str]:
    """
    Split a string by comma (,) preventing the split when
    the comma is inside quotation marks (").
    Quotation marks are **not** removed in the process.

    Raises:
        ValueError: if quotation marks are not closed

    Args:
        value: the string to split

    Returns:
        A list of string
    """
    count = value.count('"')
    if count == 0:
        return value.split(',')
    elif count % 2 != 0:
        raise ValueError('non-closed quotation marks')
    else:
        parts = []
        start = 0
        quote = False
        for i, c in enumerate(value):
            if c == ',' and not quote:
                parts.append(value[start:i])
                start = i + 1
            elif c == '"':
                quote = not quote
        if start != len(value):
            parts.append(value[start:])
        return parts


if __name__ == '__main__':
    _d = pd.DataFrame({
        'qid': [0, 1, 2],
        'query': ['a', 'b', 'c']
    })

    print(_d, '\n')

    _d = replace_col_with_history('query', ['e', 'f', 'g'], _d)

    print(_d, '\n')

    _d = replace_col_with_history('query', ['h', 'i', 'j'], _d)

    print(_d, '\n')

    _d = replace_col_with_history('query', ['k', 'l', 'm'], _d)

    print(_d, '\n')

    _s = 'a, "b, c", d'
    print(_s)
    print(split_comma_not_quotes(_s))
    if _s != ','.join(split_comma_not_quotes(_s)):
        print('ERROR with split_comma_not_quotes!')
