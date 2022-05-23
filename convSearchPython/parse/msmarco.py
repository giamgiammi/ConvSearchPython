"""Load data from MsMarco collection in tsv"""
from configparser import ConfigParser
from sys import stdout
from typing import Optional, Generator, Set

from convSearchPython.parse.document import Document
import logging


def parse_line(line: str) -> Optional[Document]:
    """Parse a line from msmarco tsv and return the retrospective document
    Args:
        line: the raw line from the tsv file
    Returns:
        A Document instance or None if there was an error"""
    parts = line.split("\t")
    if len(parts) != 2:
        logging.error("Invalid line: %s".format(line))
        return None
    return Document("MARCO_" + parts[0].strip(), parts[1].strip())


def parse_duplicates(path: str) -> Set[str]:
    """
    Parse msmarco duplicates file and return a set of document to skip
    Args:
        path: path to the duplicates file

    Returns:
        A set containing the document ids of the duplicates
    """
    with open(path, "r") as file:
        duplicates = set()
        for line in file:
            parts = line.split(":", maxsplit=1)
            for doc in parts[1].split(","):
                duplicates.add(doc.strip())
        return duplicates


def create_parser(path: str, duplicates_path: str, buffering=-1) -> Generator[Document, None, None]:
    """
    Create a parser for msmarco collection
    Args:
        path: path to the collection file
        duplicates_path: path to the duplicates file
        buffering: optional buffering size (in bytes)

    Returns:
        A generator of Document instances parsed from the msmarco collection
    """
    duplicates = parse_duplicates(duplicates_path)
    file = open(path, "r", buffering=buffering)

    def gen():
        parsed = 0
        total = 0
        logging.info("Started parsing msmarco...")
        for line in file:
            total += 1
            doc = parse_line(line)
            if doc.id() in duplicates:
                logging.debug("Skipping duplicate: {}".format(doc.id()))
                continue
            parsed += 1
            if parsed % 10000 == 0:
                logging.info("Parsed {} docs ({} skipped)".format(parsed, total - parsed))
            yield doc
        logging.info("Parsing ended. {} documents parsed ({} skipped)".format(parsed, total - parsed))
        file.close()
    return gen()


if __name__ == '__main__':
    # test
    logging.basicConfig(stream=stdout, level=logging.WARNING)
    conf = ConfigParser()
    conf.read("../config.ini")
    parser = create_parser(conf.get("MSMARCO", "collection"), conf.get("MSMARCO", "duplicates"),
                           buffering=int(conf.get("GENERAL", "buffer_size_bytes")))
    # c = 0
    # for doc in parser:
    #     c += 1
    #     if len(doc.id().strip()) == 0:
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
