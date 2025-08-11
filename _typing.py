from __future__ import annotations

from functools import wraps
import inspect
import logging
from typing import (
    Any,
    Container,
    Iterable,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union
)


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
T = TypeVar('T')
KT = TypeVar("KT")  # Mapping key generic type.
VT = TypeVar("VT")  # Mapping value generic type.

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


class ImpedanceError(RuntimeError):
    """Class for raising bad fn call arguments and similar impedance mismatches."""


class MissingEnvar(RuntimeError):
    """Class for reporting a missing envarionment variable."""


class classproperty(property):
    """Acts as a combination of property and classmethod.

    Introduced natively in python 3.9, but will be deprecated again in 3.11,
    hence why our own implementation.

    Source:
        https://stackoverflow.com/questions/1697501/staticmethod-with-property/7864317#7864317
    """
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()


def iterable_params(*params: List[str], iterableType: Type[Iterable] = list):
    """ Decorator to make any given param value iterable.

    ie: if a single element is given, cast it as an iterable containing that one element.

    .. Example::
        >>> @iterable_params("bar")
        ... def foo(bar):
        ...     return bar

        >>> foo("hello")
        ['hello']

        >>> foo(["hello"])
        ['hello']

    .. Equivalent::
        >>> def foo(bar):
        ...     if not is_collection(bar):
        ...         bar = [bar]
        ...     return bar

    Args:
        *params (str): Parameters whose values to force to iterables.
        iterableType (Type[Iterable], optional): Type to enforce for non-string iterable values.
            Defaults to `list`.

    Raises:
        ValueError: If `iterableType` is not a valid sequence type.
        ImpedanceError: If any of the `params` don't match any param in the caller's signature.
    """

    if not is_collection_type(iterableType):
        raise ValueError(f"{iterableType} is not an iterable non-string type.")

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            callArgs = inspect.getcallargs(function, *args, **kwargs)

            for param in params:
                if param not in callArgs:

                    raise ImpedanceError(
                        f"{function.__qualname__} {iterable_params.__name__} "
                        f"decorator got an unexpected argument name '{param}'."
                    )

                value = callArgs[param]

                if not is_collection(value):
                    callArgs[param] = iterableType([value])

            retval = function(**callArgs)

            return retval
        return wrapper
    return decorator



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


def is_mapping_type(obj: Any) -> bool:
    """ Returns whether the given object is an instanciable mapping type.

    Args:
        obj (Any): Object to check.

    Returns:
        bool: `True` if type is mapping type, `False` if not.

    >>> all(map(is_mapping_type, [dict, frozenmap]))
    True

    >>> any(map(is_mapping_type, [{"a": 1}, list]))
    False
    """
    try:
        instance = obj()
    except TypeError:
        return False

    return is_mapping(instance)


def is_mapping(obj: Any) -> bool:
    """Check if the given object is a key/value mapping.

    Args:
        obj (Any): Object to check.

    Return:
        bool: `True` if the input behaves like a mapping, `False` otherwise.

    >>> is_mapping({"a": 1})
    True

    >>> is_mapping(["a", "b", "c"])
    False
    """
    try:
        obj.items()
    except (AttributeError, TypeError):
        return False

    try:
        obj["test"]
    except KeyError:
        pass
    else:
        return False

    return True


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


def get_qualified_name(object_: object, withModule: bool = False) -> str:
    """Return an appropriate name for the input object_, depending on whether its a class, fn, etc.

    Args:
        object_ (Any): Object whose name to extract.

    Returns:
        str: Object name
    """
    if inspect.isfunction(object_):
        name = f"{object_.__name__}()"
    elif inspect.ismethod(object_):
        name = f"{object_.__func__.__name__}() in class {object_.__self__.__class__.__name__}"
    elif inspect.isbuiltin(object_):
        name = object_.__name__
    elif inspect.isclass(object_):
        name = object_.__name__
    elif inspect.ismodule(object_):
        name = object_.__name__
    else:
        name = object_.__class__.__name__

    if withModule:
        name = f"{object_.__module__}.{name}"

    return name


class frozenmap(Mapping[KT, VT]):
    """Non-mutable equivalent to `dict`.

    Links:
        Source - https://stackoverflow.com/a/2704866
        PEP - https://peps.python.org/pep-0603/

    Most common uses for this object would be: 
    - When memoizing function calls for functions with unknown arguments. 
    - To declare a constant at module/class level, without worrying about mutability.
    - To pass as a default parameter value, without worrying about mutability.

    Note:
        Without this object, the most common solution to store a hashable equivalent of a dict 
        (where the values are hashable) is something like: `tuple(sorted(kwargs.items()))`

        The obvious drawback of this solution is lookup is not possible unless re-cast to a dict.

    Examples:
        >>> x = frozenmap(a=1, b=2)
        >>> y = frozenmap(a=1, b=2)
        >>> x is y
        False
        >>> x == y
        True
        >>> x == {'a': 1, 'b': 2}
        True
        >>> d = {x: 'foo'}
        >>> d[y]
        'foo'
    """
    def __init__(self, *args, **kwargs: VT):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self) -> KT:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        """
        >>> frozenmap(a=1, b={1, 2}, c=[1, 2, 3])
        frozenmap(a=1, b={1, 2}, c=[1, 2, 3])
        """
        formatted = []
        for key, value in self.items():
            formatted.append(f"{key}={value!r}")

        return f"{self.__class__.__name__}({', '.join(formatted)})"

    def __getitem__(self, key: KT) -> VT:
        return self._d[key]

    def __hash__(self) -> int:
        """
        >>> m = frozenmap(a=1, b=2)
        >>> n = {m: 3}
        >>> o = frozenmap(a=1, b=2)
        >>> n[o]
        3
        """
        # It would have been simpler and maybe more obvious to 
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). 
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash


if __name__ == "__main__":
    import doctest

    doctest.testmod()
