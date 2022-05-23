"""Plot results"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Generator, Tuple, Callable, List, Union, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from convSearchPython.utils import mkdir, group_measure
from convSearchPython.utils.evaluation import evaluation_key

# workaround for anaconda in wayland sessions

if os.environ.get('QT_QPA_PLATFORM') == 'wayland':
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# reduce log
plt.set_loglevel('warn')


def _linestyles(repeat=1):
    styles = ['solid', 'dashed', 'dotted', 'dashdot', ]
    while True:
        for style in styles:
            for _ in range(repeat):
                yield style


def _plot(data: List[pd.DataFrame], save_path: Path, title: str, names: List[str] = None,
          linestyles=0, x='qid', y='value', figsize=None, sep_coord=None):
    if figsize is None:
        plt.figure(figsize=(25, 10))
    elif figsize == 'default':
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation='vertical', fontsize=10)
    linestyles = _linestyles(linestyles) if linestyles > 0 else None
    data = data if isinstance(data, list) else [data]
    for d in data:
        style = next(linestyles) if linestyles is not None else None
        plt.plot(x, y, data=d, linestyle=style)
    if names is not None:
        plt.legend(names)
    if sep_coord is not None:
        for c in sep_coord:
            plt.axvline(x=c, color='black', linestyle=(0, (1, 10)), linewidth=1)
    plt.tight_layout()
    plt.margins(0.01)
    plt.savefig(save_path)
    plt.close()


def plot_measures_no_grouping(data: pd.DataFrame, path: Path, xy=None, figsize=None,
                              sep_coord=None) -> Generator:
    """Return a generator of (func, *args) for creating a plot for every measure without grouping"""
    for name in pd.unique(data['name']):
        sub_path = Path(path, name)
        mkdir(sub_path)
        for measure in pd.unique(data['measure']):
            sub_data = data[(data['measure'] == measure) & (data['name'] == name)]
            save_path = Path(sub_path, f'{measure}.svg')
            title = f'{name} - {measure} (all queries)'
            if xy is None:
                yield _plot, {
                    'data': sub_data,
                    'save_path': save_path,
                    'title': title,
                    'figsize': figsize,
                    'sep_coord': sep_coord,
                }
            else:
                yield _plot, {
                    'data': sub_data,
                    'save_path': save_path,
                    'title': title,
                    'x': xy[0],
                    'y': xy[1],
                    'figsize': figsize,
                    'sep_coord': sep_coord,
                }


def plot_condensed_measures(data: pd.DataFrame, path: Path, xy=None, figsize=None,
                            sep_coord=None) -> Generator:
    """Similar to plot_measures_no_grouping but plot of same name are drawn on the same figure"""
    for measure in pd.unique(data['measure']):
        sub_data = []
        names = []
        for name in pd.unique(data['name']):
            names.append(name)
            sub_data.append(data[(data['measure'] == measure) & (data['name'] == name)])
        save_path = Path(path, f'{measure}.svg')
        title = f'{measure} (all queries)'
        if xy is None:
            yield _plot, {
                'data': sub_data,
                'save_path': save_path,
                'title': title,
                'names': names,
                'linestyles': 3,
                'figsize': figsize,
                'sep_coord': sep_coord,
            }
        else:
            yield _plot, {
                'data': sub_data,
                'save_path': save_path,
                'title': title,
                'names': names,
                'linestyles': 3,
                'x': xy[0],
                'y': xy[1],
                'figsize': figsize,
                'sep_coord': sep_coord,
            }


def plot_measures_condensed_names(data: pd.DataFrame, path: Path, xy=None, figsize=None,
                                  sep_coord=None) -> Generator:
    """Similar to plot_measures_no_grouping but compare rm3 and plain variants"""
    names: List[str] = list(pd.unique(data['name']))
    grouped_names = []
    while len(names) > 0:
        name = names.pop()
        if name.endswith('RM3'):
            rm3_name = name
            name = rm3_name[:-4]
            names.remove(name)
        else:
            rm3_name = names.remove(f'{name} RM3')
        grouped_names.append((name, rm3_name))
    for name, rm3_name in grouped_names:
        sub_path = Path(path, name)
        mkdir(sub_path)
        for measure in pd.unique(data['measure']):
            sub_data1 = data[(data['measure'] == measure) & (data['name'] == name)]
            sub_data2 = data[(data['measure'] == measure) & (data['name'] == rm3_name)]
            save_path = Path(sub_path, f'{measure}-RM3-comp.svg')
            title = f'{name} - {measure} RM3 comparison (all queries)'
            if xy is None:
                yield _plot, {
                    'data': [sub_data1, sub_data2],
                    'save_path': save_path,
                    'title': title,
                    'names': [f'{measure}', f'{measure} RM3'],
                    'figsize': figsize,
                    'sep_coord': sep_coord,
                }
            else:
                yield _plot, {
                    'data': [sub_data1, sub_data2],
                    'save_path': save_path,
                    'title': title,
                    'names': [f'{measure}', f'{measure} RM3'],
                    'x': xy[0],
                    'y': xy[1],
                    'figsize': figsize,
                    'sep_coord': sep_coord,
                }


def _boxplot(data: pd.DataFrame, save_path: Path, title: str, figsize=None, condensed=False, optimal_width=None):
    if figsize is None:
        figsize = (25, 10)
    elif figsize == 'default':
        figsize = None

    optimal_width = optimal_width if optimal_width is not None else 0.15

    plt.figure(figsize=figsize)
    if condensed:
        sns.boxplot(data=data, x='conversation', y='value', hue='name', width=len(data['name'].unique())*optimal_width)
    else:
        sns.boxplot(data=data, x='conversation', y='value')
    plt.title(title)
    plt.tight_layout()
    plt.margins(0.01)
    plt.savefig(save_path)
    plt.close()


def plot_per_conversation_box(data: pd.DataFrame, path: Path, sort=True) -> Generator:
    for name in pd.unique(data['name']):
        sub_path = mkdir(Path(path, name))
        for measure in pd.unique(data['measure']):
            sub_data = data[(data['name'] == name) & (data['measure'] == measure)][['conversation', 'value']]
            if sort:
                medians = sub_data.groupby('conversation')\
                        .aggregate('median')\
                        .sort_values(by='value', ascending=False)
                sub_data['conversation'] = sub_data['conversation'].astype('category').cat.set_categories(medians.index)
                sub_data.sort_values('conversation')
            yield _boxplot, {
                'data': sub_data,
                'save_path': Path(sub_path, f'{measure}.svg'),
                'title': f'{name} - {measure}',
                'figsize': 'default',
            }


def plot_per_conversation_box_condensed(data: pd.DataFrame, path: Path, sort=True, optimal_width=None, figsize=None) -> Generator:
    for measure in pd.unique(data['measure']):
        sub_data = data[data['measure'] == measure][['conversation', 'value', 'name']]
        if sort:
            medians = sub_data.groupby('conversation') \
                .aggregate('median') \
                .sort_values(by='value', ascending=False)
            sub_data['conversation'] = sub_data['conversation'].astype('category').cat.set_categories(medians.index)
            sub_data.sort_values('conversation')
        yield _boxplot, {
            'data': sub_data,
            'save_path': Path(path, f'{measure}.svg'),
            'title': f'{measure}',
            'figsize': figsize,
            'condensed': True,
            'optimal_width': optimal_width
        }


def run_job(args: Tuple[Callable, Union[Tuple, Dict]]):
    if isinstance(args[1], dict):
        args[0](**args[1])
    else:
        args[0](*args[1])


def get_measures_path_interactively():
    path = input('Measures file to use: ')
    path = path.strip()
    if path == '':
        print('Invalid path')
        exit(1)
    return Path(path).absolute()


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('measures', action='store', type=Path, help='measure dataframe in pkl', nargs='?')
    arg_parser.add_argument('-o, --output-dir', action='store', type=Path, dest='out_dir',
                            help='output dir (default to same dir as measures)', default=None)
    arg_parser.add_argument('-p, --processes', action='store', type=int, dest='processes',
                            help='number of process to use (default 1)', default=1)
    arg_parser.add_argument('-K, --kind', action='append', type=str, default=[], dest='kind',
                            help='kind of plot to create: all, lines, box')
    arg_parser.add_argument('-w, --optimal-width', action='store', type=float, dest='optimal_width', default=0,
                            help='condensed plot optimal width')

    opt = arg_parser.parse_args()
    print(opt)

    if len(opt.kind) == 0:
        opt.kind.append('all')

    optimal_width = opt.optimal_width
    if optimal_width == 0:
        optimal_width = None

    measures_path = opt.measures.absolute() if opt.measures is not None else get_measures_path_interactively()
    output_dir = opt.out_dir if opt.out_dir is not None else measures_path.parent
    output_dir = Path(output_dir, 'plots')
    mkdir(output_dir)

    measures_data = pd.read_pickle(str(measures_path)) \
        .sort_values(['name', 'qid'], key=evaluation_key) \
        .reset_index()

    names = tuple(pd.unique(measures_data['name']))

    from convSearchPython.dataset.CAsT import cast_dataset, CastDataset
    queries, qrels, conversations, query_map = cast_dataset(CastDataset.cast2019)

    conv_keys = sorted(conversations.keys(), key=int)
    unique_qids = set(pd.unique(measures_data['qid']))

    def _sep_coord():
        count = 0
        for k in conv_keys:
            conv = set(conversations[k]).intersection(unique_qids)
            # print(f'{k}: {conv}')
            if len(conv) == 0:
                continue
            if count == 0:
                count = len(conv) - 1 + 0.5
                yield count
            else:
                count = count + len(conv)
                yield count

    sep_coord = tuple(_sep_coord())[:-1]
    # print(sep_coord)

    def jobs():
        no_rm3_names = [x for x in names if 'RM3' not in x and 'rm3 not in x']

        if 'all' in opt.kind or 'line' in opt.kind:
            # all queries
            all_queries_path = mkdir(Path(output_dir, 'measures_all_queries'))
            for x in plot_measures_no_grouping(measures_data, all_queries_path, sep_coord=sep_coord):
                yield x
            for x in plot_measures_condensed_names(measures_data, all_queries_path, sep_coord=sep_coord):
                yield x
            no_rm3 = measures_data[measures_data['name'].isin(no_rm3_names)]
            for x in plot_condensed_measures(no_rm3, all_queries_path, sep_coord=sep_coord):
                yield x

            # per conversation
            conv_queries_path = mkdir(Path(output_dir, 'measures_per_conversation'))
            conv_data = group_measure.aggregate_per_conversation(measures_data, query_map)
            for x in plot_measures_no_grouping(conv_data, conv_queries_path, xy=('conversation', 'value'), figsize='default'):
                yield x
            for x in plot_measures_condensed_names(conv_data, conv_queries_path, xy=('conversation', 'value'), figsize='default'):
                yield x
            no_rm3 = conv_data[conv_data['name'].isin(no_rm3_names)]
            for x in plot_condensed_measures(no_rm3, conv_queries_path, xy=('conversation', 'value'), figsize='default'):
                yield x

        if 'all' in opt.kind or 'box' in opt.kind:
            # box plots
            box_per_conv_path = mkdir(Path(output_dir, 'box_measures_per_conversation'))
            data_with_conv = measures_data.copy()
            data_with_conv['conversation'] = [query_map[x][0] for x in measures_data['qid']]
            for x in plot_per_conversation_box(data_with_conv, box_per_conv_path):
                yield x
            no_rm3 = data_with_conv[data_with_conv['name'].isin(no_rm3_names)]
            for x in plot_per_conversation_box_condensed(no_rm3, box_per_conv_path, optimal_width=optimal_width):
                yield x

    if opt.processes == 1:
        for j in jobs():
            run_job(j)
    elif opt.processes > 1:
        with Pool(opt.processes) as pool:
            for _ in pool.imap(run_job, jobs()):
                pass
    else:
        raise Exception(f'Invalid processes number {opt.processes}')


if __name__ == '__main__':
    main()
