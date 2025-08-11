from __future__ import annotations

import logging
from typing import (
    Any,
    Container,
    Optional,
    TypeVar,
    Union
)


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
T = TypeVar('T')

# Define the sentinel object
class _Undefined: pass
Undefined = _Undefined()
"""
Sentinel for when None is a possible value.

>>> foo = {"bar": None}

.. Problem::
    >>> foo.get("bar", None) is None

    But `None` often is ambiguous, could mean "bar" is not defined OR that `None` is its value.

.. Solution::
    >>> foo.get("bar", Undefined)
    None

    Unambiguous result, we know "bar" is defined, and that its value is `None.`
"""
AcceptsNone = Union[Optional[T], _Undefined]
"""
Type for the `Undefined` sentinel.
.. Example::
    Allows mimicking the behaviour of the dict.get() method:

    >>> def getValue(d: Dict, k: Any, default: AcceptsNone[str] = Undefined) -> str:
    ...     if default is Undefined:
    ...         raise KeyError("Key does not exist and no default passed.")
    ...     return default

    >>> getValue({"a": 1}, "b")
    Traceback (most recent call last):
    KeyError: Key does not exist and no default passed.

    >>> getValue({"a": 1}, "b", default=None)
    None
"""


def is_stringy(obj):
    """Return True if the input object is a string-like entity (str, unicode, bytearray, bytes), False otherwise.

    Args:
        obj (any argument): Something to check whether it is string-like or not

    Return:
        is_string_like (bool): True if the input is like a string/unicode, False otherwise

    >>> is_stringy("Hello, world!")
    True

    >>> is_stringy(["a", "b", "c"])
    False
    """

    return isinstance(obj, (str, bytearray, bytes))


def is_stringy_type(obj: Any) -> bool:
    """ Returns whether the given object is an instanciable stringy type.

    Args:
        obj (type): Object to check.

    Returns:
        bool: `True` if type is stringy type, `False` if not.

    >>> is_stringy_type(str)
    True

    >>> is_stringy_type("str")
    False

    >>> is_stringy_type(list)
    False
    """
    try:
        obj = obj()
    except TypeError:
        return False

    return is_stringy(obj)


def is_collection(obj):
    """Return True for a collection which is not a string-type, False otherwise.

    Args:
        obj (any argument): Something to check whether it is a collection type object or not.

    Return:
        is_a_collection (bool): True if the object is some kind of collection (but not a string), False otherwise

    >>> is_collection("Hello, world!")
    False

    >>> is_collection(["a", "b", "c"])
    True
    """

    if is_stringy(obj):
        return False

    elif isinstance(obj, Container):
        return True

    else:
        try:
            iter(obj)
            return True

        except TypeError:
            return False


def is_collection_type(type_: type) -> bool:
    """ Returns whether the given type is a collection type.

    Args:
        type_ (type): Type object to check.

    Returns:
        bool: `True` if type is collection type, `False` if not.

    >>> is_collection_type(list)
    True

    >>> is_collection_type([1, 2])
    False

    >>> is_collection_type(str)
    False
    """
    try:
        obj = type_()
    except TypeError:
        return False

    return is_collection(obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
