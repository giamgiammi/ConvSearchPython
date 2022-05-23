import logging
import tempfile
from decimal import Decimal
from pathlib import Path, PurePath
from typing import Union, Generator


def mkdir(dir_path: Union[str, Path]):
    """Make directory wrapper. Do not cause error if directory already exists.
    Return the Path of the directory"""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mktmpdir(parent: Union[str, Path]) -> Path:
    """Return a newly created temp directory inside specified path.

    The directory must be manually cleared"""
    tmp = tempfile.mkdtemp(dir=str(parent))
    return Path(tmp)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s_value = str(value).lower().strip()
    if s_value == 'true':
        return True
    elif s_value == 'false':
        return False
    logging.getLogger(__name__ + '/parse_bool').warning('value \'%s\' is not true/false, using bool(value) = %s',
                                                        str(value), str(bool(value)))
    return bool(value)


def recursive_rm(path: Path):
    if not issubclass(path.__class__, PurePath):
        path = Path(path)

    if not path.is_dir():
        raise Exception(f'path {path} is not a dir')

    for item in path.iterdir():
        if item.is_dir():
            recursive_rm(item)
        else:
            item.unlink()
    path.rmdir()


def or_else(a, b):
    """Return a if a is not None, else b"""
    return a if a is not None else b


def float_range_decimal(start, stop, step) -> Generator[Decimal, None, None]:
    """Generate numbers from start to stop adding step at every iteration.
    Uses Decimal numbers to avoid errors.

    Note: stop is included if reached"""
    start = Decimal(start)
    stop = Decimal(stop)
    step = Decimal(step)
    value = start
    while value <= stop:
        yield value
        value += step
