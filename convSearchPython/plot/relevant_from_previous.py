"""Plot relevant docs from previous query"""
import math
from argparse import ArgumentParser
from pathlib import Path

from pandas import DataFrame

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from convSearchPython.plot.plot_utils import argparse_positive_int_tuple_or_none, plot_bars, plot_bars_sub
from convSearchPython.dataset.CAsT import CastDataset
from convSearchPython.basics import conf
from convSearchPython.dataset import Conversations


def previous_relevant(qrels: DataFrame, conversations: Conversations) -> DataFrame:
    relevant = qrels[qrels['label'] > 0]
    ids = []
    docs = []
    for conv_list in conversations.values():
        for i in range(1, len(conv_list)):
            rel = set(relevant[relevant['qid'] == conv_list[i]]['docno'])
            if len(rel) == 0:
                continue
            prev_rel = set(relevant[relevant['qid'] == conv_list[i-1]]['docno'])
            still_rel = prev_rel.intersection(rel)
            ids.append(conv_list[i])
            docs.append(len(still_rel))
    return DataFrame({'qid': ids, 'docs': docs})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-c, --conversations', type=str, nargs='+', dest='conv', default=None, help='list of conversation to include')
    parser.add_argument('-o, --output', type=Path, default=None, dest='out', help='output file path')
    parser.add_argument('-s, --size', type=argparse_positive_int_tuple_or_none, dest='size', default=None, help='figure size')
    parser.add_argument('-t, --tight', action='store_true', dest='tight', default=None, help='tight layout')

    opt = parser.parse_args()

    queries, qrels, conversations, query_map = CastDataset.cast2019.get_dataset()
    workdir = conf.get('GENERAL', 'workdir')

    if opt.conv is not None:
        size = None
        tight = False
        vals = []
        for k, v in conversations.items():
            if k in opt.conv:
                vals.extend(v)
        qrels = qrels[qrels['qid'].isin(vals)]
    else:
        # size = (75, 7)
        size = (15, 10)
        tight = True

    if opt.tight is not None:
        tight = opt.tight

    if opt.size is not None:
        size = opt.size

    if opt.out is None:
        path = Path(workdir, 'prev_rel_docs_in_sub_queries.svg')
    else:
        path = opt.out

    prev_rel = previous_relevant(qrels, conversations)

    if opt.conv is not None:
        plot_bars('qid', 'docs', prev_rel, path, size, tight,
                  'Number of docs still relevant from previous query in the same conversation', None)
    else:
        data = []
        for conv_id, conv_list in conversations.items():
            conv = prev_rel[prev_rel['qid'].isin(conv_list)]
            if len(conv) > 0:
                conv['qid'] = range(2, len(conv) + 2)
                data.append((conv, conv_id))
        cols = 5
        rows = math.ceil(len(data) / cols)
        plot_bars_sub('qid', 'docs', rows, cols, data, size, tight, path,
                      'Number of docs still relevant from previous query in the same conversation')


if __name__ == '__main__':
    main()
