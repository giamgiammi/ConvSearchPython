"""Create index from cast and msmarco"""
import argparse
from datetime import timedelta
from time import time
from typing import Generator, Dict
from sys import exit, stderr

from convSearchPython.parse import msmarco, car


def msmarco_generate() -> Generator[Dict[str, str], None, None]:
    """Generator to index msmarco"""
    parser = msmarco.create_parser(conf.get("MSMARCO", "collection"), conf.get("MSMARCO", "duplicates"),
                                   buffering=int(conf.get("GENERAL", "buffer_size_bytes")))
    for doc in parser:
        yield {
            'docno': doc.id(),
            'text': doc.body()
        }


def car_generate() -> Generator[Dict[str, str], None, None]:
    """Generator to index car"""
    parser = car.create_parser(conf.get("TREC_CAR", "collection"),
                               buffering=int(conf.get("GENERAL", "buffer_size_bytes")))
    for doc in parser:
        yield {
            'docno': doc.id(),
            'text': doc.body()
        }


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('index', action='store', help='name of the index to create', default=None, nargs='?')
    arg_parser.add_argument('-l, --list', action='store_true', dest='list', help='List possible indexes to create')
    arg_parser.add_argument('-t, --threads', action='store', type=int, dest='threads', default=0,
                            help='Override number of threads')

    opt = arg_parser.parse_args()
    print(opt)

    from basics import conf, pyterrier as pt
    from pipelines import Index

    if opt.list:
        print('Available Index configuration:')
        for i in Index:
            print(f'    {i.name}')
        exit(0)
    elif opt.index is None:
        print('You must specify an index', file=stderr)
        exit(1)

    threads = int(conf.get("GENERAL", "threads"))
    if opt.threads > 0:
        threads = opt.threads
    if threads <= 0:
        threads = 1
    try:
        index: Index = getattr(Index, opt.index)
    except:
        print(f'Invalid index {opt.index}', file=stderr)
        raise

    indexer = pt.IterDictIndexer(index.value, threads=threads, overwrite=True)
    if index.get_properties() is not None:
        for k, v in index.get_properties().items():
            indexer.setProperty(k, v)

    def all_generate():
        print("Indexing msmarco")
        for d in msmarco_generate():
            yield d
        print("Indexing car")
        for d in car_generate():
            yield d

    start = time()
    indexer.index(all_generate(),
                  meta=['docno', 'text'],
                  meta_lengths=[44, 4096],
                  fields=['text'])
    print(f'Indexed in {timedelta(seconds=(time() - start))}')
