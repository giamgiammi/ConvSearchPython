"""Contain the class Document that describe a parsed document"""


class Document:
    """
    Class that describe a parsed document
    """
    def __init__(self, doc_id: str, body: str):
        """
        Args:
            doc_id: document unique identifier
            body: content of the document
        """
        self.__id = doc_id
        self.__body = body

    def id(self) -> str:
        """Return the title"""
        return self.__id

    def body(self) -> str:
        """Return the body"""
        return self.__body

    def dict(self) -> dict:
        """Return document as dict"""
        return {"id": self.__id, "body": self.__body}

    def __str__(self) -> str:
        return "Document{{id={},body={}}}".format(self.__id, self.__body)




