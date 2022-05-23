"""Plot queries heatmap"""
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Tuple, Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from convSearchPython.plot.plot_utils import argparse_positive_int_tuple_or_none
from convSearchPython.dataset.CAsT import CastDataset
from convSearchPython.basics import conf, pyterrier as pt


def jaccard(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))


def _vect(union, s):
    for v in union:
        yield int(v in s)


def cos_sim(a: set, b: set):
    union = list(a.union(b))
    arr_a = np.array(list(_vect(union, a)))
    arr_b = np.array(list(_vect(union, b)))
    return np.dot(arr_a, arr_b) / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b))


textscorerTf = pt.text.scorer(body_attr="text", wmodel="Tf")


def tf(a: str, b: str):
    d = DataFrame([['a', a, 'b', b]], columns=["qid", "query", "docno", "text"])
    rtr = textscorerTf.transform(d)
    return rtr


def common_terms(queries: DataFrame):
    terms = {}
    for _, row in queries.iterrows():
        terms[row['qid']] = set(x.lower().strip() for x in row['query'].split())
    qids = queries['qid'].unique()
    common_jaccard = []
    common_cos = []
    common_tf = []
    for current in qids:
        nums = []
        j = []
        c = []
        t = []
        for qid in qids:
            # if current == qid:
            #     continue
            v = len((terms[current].intersection(terms[qid])))
            nums.append(v)
            j.append(jaccard(terms[current], terms[qid]))
            c.append(cos_sim(terms[current], terms[qid]))
            t.append(tf(queries[queries['qid'] == current].iloc[0]['query'], queries[queries['qid'] == qid].iloc[0]['query']).iloc[0]['score'])
        common_jaccard.append(j)
        common_cos.append(c)
        common_tf.append(t)
    return {'jaccard': common_jaccard, 'cosine': common_cos, 'tf': common_tf}


def plot_heatmap(data: Iterable[Tuple[Any, str]], path: Path, rows: int, cols: int, size: tuple, tight: bool,
                 cbar: bool, suptitle: str, sup_pos: float):
    fig, axs = plt.subplots(rows, cols, figsize=size)

    if cbar:
        cbar_ax = fig.add_axes([.94, .3, .03, .4])
        fig.tight_layout(rect=[0, 0, .94, 1])
    elif tight:
        fig.set_tight_layout(True)
    if suptitle is not None:
        fig.suptitle(suptitle, y=sup_pos)

    i = 0
    j = 0
    for d, name in data:
        axs[i, j].set_title('conv {}'.format(name))
        ticks = list(range(1, len(d)+1))
        if cbar and i == j == 0:
            sns.heatmap(d, cbar=True, square=True, xticklabels=ticks, yticklabels=ticks, ax=axs[i, j], cbar_ax=cbar_ax)
        else:
            sns.heatmap(d, cbar=False, square=True, xticklabels=ticks, yticklabels=ticks, ax=axs[i, j])

        j = j + 1
        if j >= cols:
            j = 0
            i = i + 1

    fig.savefig(path)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', default=False, dest='all', help='keep all conversation (not only testing ones)')
    parser.add_argument('-c --columns', type=int, default=5, dest='cols', help='number of columns')
    parser.add_argument('-s, --size', type=argparse_positive_int_tuple_or_none, dest='size', default=None, help='figure size')
    parser.add_argument('--not-tight', action='store_true', dest='not_tight', default=None, help='no tight layout')
    parser.add_argument('-b, --cbar', action='store_true', dest='cbar', default=False, help='show cbar')
    parser.add_argument('-p, --suptitle_pos', type=float, default=None, dest='sup', help='suptitle position')

    types = (
        'all',
        'cosine',
        'jaccard',
        'tf',
    )
    parser.add_argument('-k, --kind', choices=types, default='all', dest='kind', help='kind of heatmap')

    opt = parser.parse_args()

    queries, qrels, conversations, query_map = CastDataset.cast2019.get_dataset()
    workdir = conf.get('GENERAL', 'workdir')

    if not opt.all:
        qids = set(qrels['qid'])
        new_conv = {}
        for conv_id, conv_list in conversations.items():
            conv_list = list(set(conv_list).intersection(qids))
            if len(conv_list) != 0:
                new_conv[conv_id] = conv_list
        conversations = new_conv
        del new_conv, qids

    len_convs = len(conversations)
    cols = opt.cols
    rows = math.ceil(len_convs / cols)

    if opt.size is None:
        if opt.cbar:
            size = (13, 12)
        else:
            size = (11, 10)
    else:
        size = opt.size

    if opt.sup is None:
        if opt.cbar:
            sup_pos = 0.995
        else:
            sup_pos = 0.99
    else:
        sup_pos = opt.sup

    data = []
    for conv_id, conv_list in conversations.items():
        conv_queries = queries[queries['qid'].isin(conv_list)]
        data.append((common_terms(conv_queries), conv_id))

    if opt.kind == 'all' or 'all' in opt.kind:
        kinds = types[1:]
    elif isinstance(opt.kind, str):
        kinds = [opt.kind]
    else:
        kinds = list(opt.kind)

    for kind in kinds:
        path = Path(workdir, 'heatmap_{}.svg'.format(kind))
        it = ((x[kind], y) for x, y in data)
        plot_heatmap(it, path, rows, cols, size, not opt.not_tight, opt.cbar, 'Queries similarity ({})'.format(kind), sup_pos)


if __name__ == '__main__':
    main()
