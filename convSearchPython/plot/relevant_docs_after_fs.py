"""Plot the number/value of relevant docs inside the result from the first query of every conversation"""
import argparse
import itertools
from pathlib import Path


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import convSearchPython.plot.plot_utils as pils
from convSearchPython.dataset.CAsT import CastDataset
from convSearchPython.pipelines import *
from convSearchPython.plot.plot_utils import plot_bars, plot_bars_sub
from convSearchPython.utils.data_utils import queries_conversation_splitter


def get_qid_with_relevant_qrels(qrels: pd.DataFrame):
    data = qrels[['qid', 'label']].groupby('qid').agg('sum').reset_index()
    qids = data[data['label'] > 0]['qid'].unique()
    return set(qids)


def get_recall(n: int, qid: str, qrels: pd.DataFrame) -> float:
    count = len(qrels[(qrels['qid'] == qid) & (qrels['label'] > 0)])
    return n / count


def base_search(model, n, index: str):
    queries, qrels, conversations, query_map = CastDataset.cast2019.get_dataset()
    qid_with_relevant_qrels = get_qid_with_relevant_qrels(qrels)
    queries = queries[queries['qid'].isin(qid_with_relevant_qrels)]
    tr = pt.BatchRetrieve(IndexConf.load_index(index).index,
                          wmodel=model,
                          controls={'c': 0.75 if model == "BM25" else 2500},
                          properties=IndexConf.load_index(index).properties,
                          metadata=['docno'],
                          num_results=n)
    relevant_num = {}
    relevant_value = {}
    for conv in queries_conversation_splitter(queries, conversations):
        res = tr(conv.iloc[0:1])
        docnos = set(res['docno'].unique())
        for qid in conv['qid'].unique():
            rel = qrels[(qrels['qid'] == qid) & (qrels['label'] > 0) & (qrels['docno'].isin(docnos))]
            relevant_num[qid] = len(rel)
            relevant_value[qid] = rel['label'].sum()

    qids = tuple(x for x in itertools.chain.from_iterable(conversations.values()) if x in qid_with_relevant_qrels)
    numbers = [relevant_num[x] for x in qids]
    values = [relevant_value[x] for x in qids]
    recall = [get_recall(relevant_num[x], x, qrels) for x in qids]
    conv_ids = [query_map[x][0] for x in qids]
    firsts_set = set(x[0] for x in conversations.values())
    firsts = [True if x in firsts_set else False for x in qids]
    data = pd.DataFrame({
        'qid': qids,
        'relevant_docs': numbers,
        'labels_sum': values,
        'recall': recall,
        'conv': conv_ids,
        'is_first': firsts,
    })

    return data


def plot_relevant_qid_per_qid(model, num, size, path, exclude_firsts, what, tight, index):
    data = base_search(model, num, index)
    if exclude_firsts:
        data = data[~data['is_first']]
    if path is None:
        ex = '_excluded-firsts' if exclude_firsts else ''
        path = f'relevant_{what}{ex}_{model}_{num}.svg'
    path = str(Path(conf.get('GENERAL', 'workdir'), path))
    if exclude_firsts:
        sub = f'based on {num} results, 1° utt. excluded'
    else:
        sub = f'based on {num} results'

    datas = []
    for conv_id in data['conv'].unique():
        conv = data[data['conv'] == conv_id]
        conv['index'] = range(1, len(conv) + 1)
        datas.append((conv, conv_id))

    cols = 5
    rows = math.ceil(len(datas)/cols)

    if what == 'labels':
        plot_bars_sub('index', 'labels_sum', rows, cols, datas, size, tight, path, 'Relevant labels sum per qid ({})'.format(sub))
    elif what == 'docs':
        plot_bars_sub('index', 'relevant_docs', rows, cols, datas, size, tight, path, 'Relevant docs per qid ({})'.format(sub))
    elif what == 'recall':
        plot_bars_sub('index', 'recall', rows, cols, datas, size, tight, path, 'Recall per qid ({})'.format(sub))


def plot_relevant_per_conv(model, num, size, path, exclude_firsts, what, tight, index):
    data = base_search(model, num, index)
    if exclude_firsts:
        data = data[~data['is_first']]
    if path is None:
        ex = '_excluded-firsts' if exclude_firsts else ''
        path = f'relevant_{what}{ex}_per_conv_{model}_{num}.svg'
    path = str(Path(conf.get('GENERAL', 'workdir'), path))
    if exclude_firsts:
        sub = f'based on {num} results (first utterances excluded from plot)'
    else:
        sub = f'based on {num} results'
    if what == 'labels':
        data = data.groupby('conv').agg('sum').reset_index()
        plot_bars('conv', 'labels_sum', data, path, size, tight, 'Relevant labels sum per conversation ({})'.format(sub), None)
    elif what == 'docs':
        data = data.groupby('conv').agg('sum').reset_index()
        plot_bars('conv', 'relevant_docs', data, path, size, tight, 'Relevant docs per conversation ({})'.format(sub), None)
    elif what == 'recall':
        data = data.groupby('conv').agg('mean').reset_index()
        plot_bars('conv', 'recall', data, path, size, tight, 'Recall per conversation ({})'.format(sub), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m, --model', choices=('BM25', "DLM"), dest='model', default='DLM', help='search model')
    parser.add_argument('-n, --num', type=pils.argparse_positive_int, dest='num', default=10000, help='number of results')
    parser.add_argument('-s, --size', type=pils.argparse_positive_int_tuple_or_none, dest='size', default=None, help='figure size')
    parser.add_argument('-o, --output', type=pils.argparse_path_or_none, dest='path', default=None, help='output file')
    parser.add_argument('-x, --exclude-firsts', action='store_true', dest='exclude_firsts', default=False, help='exclude first utterance for every conversation')
    parser.add_argument('-t, --tight', action='store_true', dest='tight', default=False, help='tight layout')
    plots = (
        'qid-docs',
        'qid-labels',
        'qid-recall',
        'conv-docs',
        'conv-labels',
        'conv-recall',
    )
    parser.add_argument('-p, --plot', choices=plots, dest='plot', default='conv-docs', help='kind of plot')
    parser.add_argument('-i, --index', default='custom', dest='index', help='index to use')

    opt = parser.parse_args()
    print(opt)

    model = 'DirichletLM' if opt.model == 'DLM' else opt.model
    num = opt.num
    size = opt.size
    path = opt.path
    exclude_firsts = opt.exclude_firsts
    plot = opt.plot
    tight = opt.tight

    on, kind = plot.split('-')

    if on == 'qid':
        plot_relevant_qid_per_qid(model, num, size, path, exclude_firsts, kind, tight, opt.index)
    elif on == 'conv':
        plot_relevant_per_conv(model, num, size, path, exclude_firsts, kind, tight, opt.index)


if __name__ == '__main__':
    main()
