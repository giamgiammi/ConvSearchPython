"""Load data from trec car"""
import logging
from configparser import ConfigParser
from sys import stdout
from typing import Generator

from trec_car.read_data import iter_paragraphs

from convSearchPython.parse.document import Document


def create_parser(path: str, buffering=-1) -> Generator[Document, None, None]:
    """Create a parser for trec CAR

    Args:
        path: path to the dataset file
        buffering: optional buffering size (in bytes)
    Returns:
        A generator of Document instance containing parsed CAR documents"""

    file = open(path, "rb", buffering=buffering)
    it = iter_paragraphs(file)

    def gen():
        parsed = 0
        for data in it:
            parsed += 1
            if parsed % 100000 == 0:
                logging.info("parsed {} documents".format(parsed))
            yield Document("CAR_" + data.para_id.strip(), data.get_text().strip())
        file.close()
        logging.info("Parse ended. Parsed {} documents".format(parsed))

    return gen()


if __name__ == '__main__':
    logging.basicConfig(stream=stdout, level=logging.WARNING)
    conf = ConfigParser()
    conf.read("../config.ini")
    parser = create_parser(conf.get("TREC_CAR", "collection"),
                           buffering=int(conf.get("GENERAL", "buffer_size_bytes")))

    # for doc in parser:
    #     input("press enter to continue...")
    #     print(doc)
    # c = 0
    # for doc in parser:
    #     c += 1
    #     if len(doc.id()) == 0:
    #         print(f"empty id on doc #{c}")
    #     if len(doc.body().strip()) == 0:
    #         print(f"doc {doc.id()} has empty body")
    # print(f"parsed {c} docs")
    max = 0
    for doc in parser:
       l = len(doc.id())
       if l > max:
           max = l
    print(f"max id len: {max}")
