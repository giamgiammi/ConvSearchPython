"""Test for classical IR task"""
import argparse
from os import environ
from pathlib import Path

from pandas import DataFrame
# noinspection PyUnresolvedReferences
from pyterrier.measures import AP, NDCG, RR, R, P

from convSearchPython.basics import pyterrier as pt


def index_nyt(file_path: str, index_path: Path):
    if index_path.exists():
        return pt.IndexFactory.of(str(index_path))

    import tarfile
    import ir_datasets
    BeautifulSoup = ir_datasets.lazy_libs.bs4().BeautifulSoup

    def file_loop(s_file: tarfile.TarFile):
        for member in s_file:
            if member.name.startswith('.'):
                continue
            if member.name.endswith('.tar'):
                file = s_file.extractfile(member)
                with tarfile.open(fileobj=file, mode='r:') as i_file:
                    for x in file_loop(i_file):
                        yield x
            if member.name.endswith('.tgz') or member.name.endswith('.tar.gz'):
                file = s_file.extractfile(member)
                with tarfile.open(fileobj=file, mode='r:gz') as i_file:
                    for x in file_loop(i_file):
                        yield x
            if member.name.endswith('.xml'):
                data = tfile.extractfile(member).read()
                tree = BeautifulSoup(data, 'lxml-xml')
                did = tree.find('doc-id')
                hl1 = tree.find('hl1')
                full_text = tree.find('block', {'class': 'full_text'})
                yield {
                    'docno': did,
                    'headline': hl1,
                    'body': full_text,
                }
    with tarfile.open(file_path, 'r:gz') as tfile:
        indexer = pt.IterDictIndexer(str(index_path))
        index_ref = indexer.index(file_loop(tfile), fields=['headline', 'body'])
        return pt.IndexFactory.of(index_ref)


def create_or_load_index(dataset, path: Path, fields):
    if path.exists():
        return pt.IndexFactory.of(str(path))
    print(f'Create index for {dataset} in {path}')
    indexer = pt.index.IterDictIndexer(str(path))
    index_ref = indexer.index(dataset.get_corpus_iter(), fields=fields)
    return pt.IndexFactory.of(index_ref)


def search(index, dataset, result_path: Path):
    print(f'Starting search with dataset {dataset} on index {index}')
    BM25 = pt.BatchRetrieve(index, wmodel="BM25")
    BM25_RM3 = BM25 >> pt.rewrite.RM3(index) >> BM25
    QLM = pt.BatchRetrieve(index, wmodel="DirichletLM")
    QLM_RM3 = QLM >> pt.rewrite.RM3(index) >> QLM

    print('searching')
    metrics = pt.Experiment([BM25, BM25_RM3, QLM, QLM_RM3],
                            dataset.get_topics('title'),
                            dataset.get_qrels(),
                            eval_metrics=[P@30, AP, NDCG@3, RR, R@200, P@1, P@3],
                            names=['BM25', 'BM25 RM3', 'QLM', 'QLM RM3'])
    metrics: DataFrame
    print(f'saving result to {result_path}')
    metrics.groupby('name').aggregate('mean').to_pickle(str(result_path))


def index_and_search(dataset_name: str, fields, index_path: Path, result_path: Path):
    dataset = pt.get_dataset(dataset_name)
    index = create_or_load_index(dataset, index_path, fields)
    search(index, dataset, result_path)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-I, --index', required=True, dest='index', type=Path, help='Index path')
    arg_parser.add_argument('-R, --result', required=True, dest='result', type=Path, help='Result path')
    arg_parser.add_argument('-D, --dataset', required=True, dest='dataset', type=str, help='Dataset name')

    opt = arg_parser.parse_args()

    name = opt.dataset.lower()
    index_path = opt.index.absolute()
    result_path = opt.result.absolute()
    if name == 'robust04' or name == 'rb04':
        index_and_search('irds:trec-robust04', ['text', 'marked_up_doc'], index_path, result_path)
    elif name == 'core17':
        # index_and_search('irds:nyt/trec-core-2017', ['headline', 'body'], index_path, result_path)
        dataset = pt.get_dataset('irds:nyt/trec-core-2017')
        index = index_nyt(f'{environ["HOME"]}/.ir_datasets/nyt/nyt.tgz', index_path)
        search(index, dataset, result_path)
    elif name == 'core18':
        index_and_search('irds:wapo/v2/trec-core-2018', ['url', 'title', 'author', 'kicker', 'body'], index_path, result_path)
    else:
        raise Exception(f'Invalid dataset {name}')


if __name__ == '__main__':
    main()
