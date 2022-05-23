"""Imports utils"""
import importlib
from typing import Union, Type, Callable


def import_object(name: str):
    """
    Import an object from the full reference.

    Raises:
         ValueError: if no object with that name is found.

    Args:
        name: full reference to object name

    Returns:
        Class object
    """
    parts = [x.strip() for x in name.split('.')]
    module = importlib.import_module('.'.join(parts[0:-1]))
    attrs = dir(module)
    try:
        att = attrs[attrs.index(parts[-1])]
        _class = getattr(module, att)
        return _class
    except ValueError:
        raise ValueError(f'cannot import object "{name}", make sure it\'s in format package.module.class')


def instantiate_class(_class: Union[str, Type], *args, **kwargs):
    """
    Instantiate a class.

    Raises:
        ValueError: if _class is a string and do not refer to a valid object
        TypeError: if the arguments are incorrect
    Args:
        _class: class object or string pointing to a class object
        *args: positional arguments for the class
        **kwargs: named arguments for the class

    Returns:

    """
    if isinstance(_class, str):
        _class = import_object(_class)
    obj = _class(*args, **kwargs)
    return obj


def import_callable(name: str) -> Callable:
    """
    Same as `import_object`, but check that the imported object
    is callable.

    Raises:
         ValueError: if no object with that name is found.
         TypeError: if object is not callable

    Args:
        name: full reference to object name

    Returns:
        Callable object
    """
    obj = import_object(name)
    if not callable(obj):
        raise TypeError(f'Object {obj} is not callable')
    return obj
