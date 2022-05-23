"""Find different in docs returned by different methods"""
import argparse
import datetime
from pathlib import Path
from typing import Tuple, Iterable, Dict, Set

import pandas as pd

from convSearchPython.utils.evaluation import evaluation_key


def get_run_paths_interactively() -> Tuple[Path]:
    """Interactively ask for run paths"""
    runs = []
    while True:
        r = input('Run file/folder to load (end with void line):').strip()
        if r == '':
            break
        path = Path(r)
        if path.is_dir():
            runs += list(path.glob('*.pkl.gz'))
        else:
            runs.append(path.absolute())
    if len(runs) == 0:
        raise Exception('No run file specified')
    return tuple(runs)


def get_run_paths(runs: Iterable[Path]) -> Tuple[Path]:
    """Get the run paths from a collection of paths expanding the folders"""
    run_paths = []
    for p in runs:
        if p.is_dir():
            run_paths += list(p.glob('*.pkl.gz'))
        else:
            run_paths.append(p.absolute())
    return tuple(run_paths)


def get_names_runs(run_paths: Iterable[Path]) -> Tuple[Set[str], Dict[str, pd.DataFrame]]:
    """Load the runs and return methods names and runs datasets"""
    runs = {}
    names = set()
    for p in run_paths:
        data: pd.DataFrame = pd.read_pickle(str(p))
        data = data.sort_values(['name', 'qid', 'rank'], key=evaluation_key)
        _names = pd.unique(data['name'])
        if len(_names) > 1:
            for n in _names:
                if n in runs:
                    raise Exception(f'Duplicated name {n}')
                runs[n] = data[data['name'] == n]
        elif len(_names) == 1:
            if _names[0] in runs:
                raise Exception(f'Duplicated name {_names[0]}')
            runs[_names[0]] = data
        else:
            raise Exception(f'no column name in {p}')
        names = names.union(_names)
    return names, runs


def limit_queries(data: pd.DataFrame, k: int) -> pd.DataFrame:
    """Limit the number of queries

    - data: original dataframe
    - k: limit at the top-k queries
    - min_label: consider only row with label >= of this value"""
    parts = []
    for _, group in data.groupby(['qid', 'name']):
        if len(group) <= k:
            parts.append(group)
        else:
            parts.append(group[:k])
    return pd.concat(parts)


def get_unique_docs(runs: Dict[str, pd.DataFrame]) -> Dict[str, set]:
    """For every dataframe in runs, generate a set of the docs absents inside other dataframes"""
    docs_per_df = {}
    for name, df in runs.items():
        docs_per_df[name] = set(pd.unique(df['docno']))
    unique_docs = {}
    for name, docs in docs_per_df.items():
        others = set()
        for _name, _docs in docs_per_df.items():
            if name != _name:
                others = others.union(_docs)
        unique_docs[name] = docs.difference(others)
    return unique_docs


def add_rm3_comparison(runs: Dict[str, pd.DataFrame], unique_docs: Dict[str, set]):
    """Update runs and unique_docs with rm3 comparison"""
    pairs = []
    seen = set()
    for name, data in runs.items():
        base_name = name[:-4] if name.endswith(' RM3') else name
        if base_name not in seen:
            pairs.append((base_name, f'{base_name} RM3'))
            seen.add(base_name)
    for base, rm3 in pairs:
        uniq_rm3 = get_unique_docs({base: runs[base], rm3: runs[rm3]})
        index = f'{base} - RM3 comparison'
        runs[index] = pd.concat((runs[base], runs[rm3]))
        unique_docs[index] = uniq_rm3[base].union(uniq_rm3[rm3])


def write_excel(runs: Dict[str, pd.DataFrame], unique_docs: Dict[str, set], output_path: Path, info: pd.Series):
    """Write excel with unique docs rows"""
    with pd.ExcelWriter(str(output_path), mode='w') as excel_file:
        info.to_excel(excel_file, sheet_name='INFO', header=False)
        excel_file.sheets['INFO'].set_column(0, 0, 15)
        excel_file.sheets['INFO'].set_column(1, 1, 50)
        for name, docs in unique_docs.items():
            data = runs[name]
            cols = [c for c in data.columns if c.startswith('query')]
            cols = ['name', 'qid', 'docno', 'rank', 'score', 'label', 'text'] + cols
            data[data['docno'].isin(docs)][cols] \
                .to_excel(excel_file, sheet_name=name, index=False)


def main():
    started = datetime.datetime.now()
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('run', action='store', type=Path,
                            help='run file or directory containing run files to load', nargs='*', default=[])
    arg_parser.add_argument('-k, --top-k', action='store', type=int, dest='k', default=0,
                            help='Limit search at top-k docs')
    arg_parser.add_argument('-l, --min-label', action='store', type=int, dest='label', default=-1,
                            help='Minimum label value to consider')
    arg_parser.add_argument('-o, --output', action='store', type=Path, dest='out',
                            help='path where to write the excel file (default to unique.xlsx in the parent '
                                 'of the first run argument)', default=None)
    arg_parser.add_argument('-r, --rm3-comparison', action='store_true', dest='rm3', default=False,
                            help='include RM3 comparison')

    opt = arg_parser.parse_args()
    print(opt)

    run_paths = get_run_paths(opt.run) if opt.run else get_run_paths_interactively()

    names, runs = get_names_runs(run_paths)

    print(names)
    print('runs: {\n' + '\n\t'.join(f'len({n}): {len(d)}' for n, d in runs.items()) + '\n}')

    if opt.label >= 0:
        for name, df in runs.items():
            runs[name] = df[df['label'] >= opt.label]

    if opt.k > 0:
        for name, df in runs.items():
            runs[name] = limit_queries(df, opt.k)
        print('runs: {\n' + '\n\t'.join(f'len({n}): {len(d)}' for n, d in runs.items()) + '\n}')

    unique_docs = get_unique_docs(runs)

    if opt.rm3:
        add_rm3_comparison(runs, unique_docs)

    output_path: Path = opt.out
    if output_path is None:
        output_path = opt.run[0].parent.absolute() if opt.run else run_paths[0].parent.absolute()
        output_path = Path(output_path, 'unique.xlsx').absolute()
    else:
        output_path = output_path.absolute()
        s_out = str(output_path)
        if not (s_out.endswith('.xlsx') or s_out.endswith('.ods') or s_out.endswith('.xlsx')):
            output_path = Path(str(output_path) + '.xlsx')
    print(f'output_path: {output_path}')

    # _info = {
    #     'created at': f'{started} - {datetime.datetime.now()}',
    #     'top k': opt.k if opt.k > 0 else 'no limit',
    #     'min label': opt.label if opt.label >= 0 else 'no limit',
    #     'rm3 comparison': 'included' if opt.rm3 else 'excluded',
    #     'run paths': [str(x) for x in run_paths],
    # }
    info = pd.Series(dtype='object')
    info['created at'] = f'{started} - {datetime.datetime.now()}'
    info['top k'] = opt.k if opt.k > 0 else 'no limit'
    info['min label'] = opt.label if opt.label >= 0 else 'no limit'
    info['rm3 comparison'] = 'included' if opt.rm3 else 'excluded'
    write_excel(runs, unique_docs, output_path, info)


if __name__ == '__main__':
    main()
