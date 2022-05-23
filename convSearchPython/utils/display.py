"""Display utilities"""

from tabulate import tabulate


def ascii_display(d) -> str:
    """Return ascii representation of d as a table"""
    return tabulate(d, headers='keys', tablefmt='psql')
