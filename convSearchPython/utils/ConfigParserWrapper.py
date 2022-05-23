"""
Wrapper to ConfigParser to make its keys case-sensitive
"""

# noinspection PyUnresolvedReferences
from configparser import ConfigParser as _ConfigParser, SectionProxy


class ConfigParser(_ConfigParser):
    """
    Case-sensitive ConfigParser
    """
    def __init__(self):
        super().__init__()
        self.optionxform = str
