"""Plot the overall relevant docs"""
import math
from pathlib import Path

from pandas import DataFrame

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from convSearchPython.dataset.CAsT import CastDataset
from convSearchPython.plot.plot_utils import plot_bars, plot_bars_sub
from convSearchPython.basics import conf
from convSearchPython.dataset import QueryMap


def plot_docs_per_query(qrels: DataFrame, path: Path, query_map: QueryMap, size=None):
    data = qrels[qrels['label'] > 0].groupby('qid').agg('count')
    data['qid'] = data.index
    data['docs'] = data['docno']
    data['conv'] = [query_map[q][0] for q in data['qid']]

    datas = []
    for conv_id in data['conv'].unique():
        conv = data[data['conv'] == conv_id]
        conv['index'] = range(1, len(conv)+1)
        datas.append((conv, conv_id))

    cols = 5
    rows = math.ceil(len(datas)/cols)
    if size is None:
        size = (15, 10)
    plot_bars_sub('index', 'docs', rows, cols, datas, size, True, path, 'Relevant docs')


def plot_docs_per_conv(qrels: DataFrame, path: Path, query_map: QueryMap):
    data = qrels.copy()
    data['conv'] = [query_map[q][0] for q in data['qid']]
    data = data[data['label'] > 0].groupby('conv').agg('count')
    data['conv'] = data.index
    data['docs'] = data['docno']
    plot_bars('conv', 'docs', data, path, None, False, 'Relevant docs per conversation', None)


def main():
    queries, qrels, conversations, query_map = CastDataset.cast2019.get_dataset()
    workdir = conf.get('GENERAL', 'workdir')
    plot_docs_per_query(qrels, Path(workdir, 'relevant_docs.svg'), query_map)
    plot_docs_per_conv(qrels, Path(workdir, 'relevant_docs_per_conv.svg'), query_map)


if __name__ == '__main__':
    main()
