"""
Low-level module for argument parsing utilities.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import shlex
from collections import defaultdict
from enum import Enum, EnumMeta
from typing import Dict, List, Optional, Tuple, TypeVar

from . import _typing
from ._typing import Undefined

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_ValueT = TypeVar("_ValueT")
""" Represents any python object than is parsable into a bash argument representation.

Disclaimer:
    This is a conscious misuse of `TypeVar`, as ValueT does not represent one single specific type,
    but serves as a practical way to document code using this concept throughout this module.
"""

_ArgDictT = Dict[argparse.Action, List[str]]
# Type for a dictionary with actions as keys, and the required list of arguments to pass.


class ParserError(RuntimeError):
    """Raise if incorrect arguments are passed to a parser."""

    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        message = f"{parser.prog}: {message}\n{parser.format_usage()}"
        super().__init__(message)


class ArgumentUnparser:
    """For a given `ArgumentParser` and python values, can parse a list of shell string arguments.

    Examples:  
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("--files", type=str, action="append", nargs="+")
        >>> _ = parser.add_argument("--amount", type=int)
        >>> _ = parser.add_argument("--debug", action="store_true")

        >>> ArgumentUnparser(parser).unparseArgs([["/path/to/file"], ["/path/to/file3"]], 12)
        ['--files', '/path/to/file', '--files', '/path/to/file3', '--amount', '12']

        >>> ArgumentUnparser(parser).unparseArgs(files=[["/path/to/file"]], amount=12, debug=True)
        ['--files', '/path/to/file', '--amount', '12', '--debug']

        >>> ArgumentUnparser(parser).unparseArgs([["/path/to/file"]])
        ['--files', '/path/to/file']

    Todo:
        TODO: Missing a round-tripping system:

        Obviously, this is a bad example, because we wouldn't usually set the type to list, 
        we'd probably use nargs/append instead. Please ignore that.

        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument("supported", type=int)     # '123' -> 123
        >>> _ = parser.add_argument("unsupported", type=list)  # 'abc' -> ['a', 'b', 'c']
        >>> ArgumentUnparser(parser).unparseArgs(123, ['a', 'b', 'c'])
        ['123', '\\'[\\'"\\'"\\'a\\'"\\'"\\', \\'"\\'"\\'b\\'"\\'"\\', \\'"\\'"\\'c\\'"\\'"\\']\\'']

        Why this result? Obviously, 123 -> '123' is not a problem.
        However, "['a', 'b', 'c']" cannot be round-tripped to the parser action's `list` type.

        Something like a `overrideAction(actionName, **kwargs)` method would maybe suffice?
    """

    def __init__(self, parser: argparse.ArgumentParser, quoteArgs: bool = True):
        """
        Args:
            parser (argparse.ArgumentParser): Parser to use to unparse.
            quoteArgs (bool, optional): Pass `False` to return args as is, without quoting.
                `True` (default) runs args through `shlex.quote` before returning,
                which quotes only when necessary.
        """
        self.parser = parser
        self.quoteArgs = quoteArgs

    def unparseArgs(self, *args, **kwargs) -> List[str]:
        """Unparse the given args/kwargs using an argparse parser.

        Args:
            parser (argparse.ArgumentParser): Parser to use to unparse arguments.

        Raises:
            ParserError: If too many positional arguments are passed.
            ParserError: If insufficient or too many arguments are passed.

        Returns:
            list[str]: List of unparsed shell arguments.
        """
        argsDict = self._unparsePositionalArgs(*args)
        kwargsDict = self._unparseKeywordArgs(**kwargs)
        overlap = set(argsDict) & set(kwargsDict)

        if overlap:
            raise ParserError(
                self.parser, f"Keyword parameters already passed as positional values: {overlap}"
            )

        argsDict.update(self._unparseKeywordArgs(**kwargs))

        # Verify if any required arguments are missing.
        requiredActions = {action for action in self.parser._actions if action.required}
        missingActions = requiredActions - {*argsDict}

        if missingActions:
            fActions = ", ".join(map(lambda action: action.dest, missingActions))
            raise ParserError(self.parser, f"Insufficient arguments passed, missing: {fActions}")

        unparsedArgs = list(itertools.chain.from_iterable(argsDict.values()))

        return unparsedArgs

    def _unparsePositionalArgs(self, *args: _ValueT) -> _ArgDictT:
        """Distribute *args into actions.

        Note:
            If more *args are passed than positional actions exist on the parser,
            the extra args are spilled into keyword actions,
            in which case they are unparsed into the keyword argument return dict instead of positional.

            This diverges from argparse, which would simply crash, but feels more pythonic.

        Args:
            args (_ValueT): Python value to unparse to shell argument(s).

        Returns:
            _ArgDictT: Actions and their parsed values.
        """
        positionalActions, keywordActions = _getActionLists(self.parser)
        argsDict = {}

        for action, arg in zip((*positionalActions, *keywordActions), args):
            unparsed = self._unparseArgument(action, arg)
            argsDict[action] = unparsed

        return argsDict

    def _unparseKeywordArgs(self, **kwargs: _ValueT) -> List[str]:
        """Unparse **kwargs into the parsers actions.

        Note:
            **kwargs is in the python sense, not argparse/shell sense.
            Ie: A kwarg can match a positional action's name,
            in which case the argument is unparsed to the positional return dict instead of keyword.

            This diverges from argparse, which would simply crash, but feels more pythonic.

        Args:
            kwargs (_ValueT): Python value to unparse to shell argument(s).

        Returns:
            List[str]: Actions and their parsed values.
        """
        argsDict = defaultdict(list)

        for name, value in kwargs.items():
            action = self.findAction(name)
            previouslyPassed = argsDict.get(action, Undefined)

            if previouslyPassed is not Undefined:
                usage = formatActionAsString(action)
                log.warning(
                    f"{usage}: argument passed more than once: {previouslyPassed!r} // {value!r}"
                )

            argsDict[action].extend(self._unparseArgument(action, value))

        return argsDict

    def findAction(self, name: str, strict: bool = True) -> Optional[argparse.Action]:
        """Find an action matching the given name.

        Args:
            name (str): Name of the action to find.
            strict (bool, optional): Pass `False` to return None if no action is found.
                Defaults to `True`, which raises an exception.

        Raises:
            ParserError: If the name doesn't match any of the parser's actions and strict is `True`.

        Returns:
            argparse.Action | None: Action matching the given name, if any.
        """
        action = None

        # WATCHME: Destination might be tricky when you take actions into account?
        #   It works fine for boolean actions (store true/false)
        #   but hasn't been tested into other weird actions.
        for action_ in self.parser._actions:
            actionNames = (action_.dest, *action_.option_strings)
            if name in actionNames:
                action = action_
                break

        if action is None and strict:
            raise ParserError(self.parser, f"'{name}' does not match any of the parser's arguments.")

        return action

    def _getDefault(self, action: argparse.Action) -> _ValueT:
        """ Return the type-cast default for the given action.

        This method mimics argparse behaviour, which casts the action default to its type.

        Args:
            action (argparse.Action): Action to fetch a default for.

        Returns:
            _ValueT: The action's default.
        """
        # WATCHME: This might be too much guesswork.
        default = self.parser.get_default(action.dest)

        if action.type:
            try:
                default = action.type(default)
            except Exception:
                usage = formatActionAsString(action)
                log.warning(f"{usage}: could not cast default {default} to {action.type}")

        return default

    def _unparseArgumentValues(self, action: argparse.Action, values: List[_ValueT]) -> List[str]:
        """Unparse keyword and value argument(s) as necessary for the given action and values.

        This method exists to handle stacking action types such as "append".

        Args:
            action (argparse.Action): Action to unparse arguments for.
            values (List[_ValueT]): Python values of the action to unparse.

        Returns:
            List[str]: Unparsed arguments.
        """
        if values == [self._getDefault(action)]:
            return []

        requiresOpStr = isOptionStringRequired(action)

        unparsedArgs = []

        # HACK: Added this to handle nargs=0 in the case of boolean arguments.
        if action.nargs == "+":
            nargs = max(len(values), 1)
        elif action.nargs == "*":
            nargs = len(values)
        elif action.nargs == "?":
            nargs = min(len(values), 1)
        elif action.nargs is None:
            nargs = 1
        else:
            nargs = int(action.nargs)

        for index, value_ in enumerate(values, start=1):
            if isOptionStringRequired(action, value_):
                requiresOpStr = True

            if index > nargs:
                continue

            unparsedArgs.append(self._unparseArgumentValue(action, value_))

        if requiresOpStr:
            unparsedArgs.insert(0, action.option_strings[0])

        return unparsedArgs

    def _unparseArgument(self, action: argparse.Action, value: _ValueT) -> List[str]:
        """Unparse value argument(s) as necessary for the given action.

        Args:
            action (argparse.Action): Action to unparse arguments for.
            value (_ValueT): Python value of the action to unparse.

        Returns:
            List[str]: Unparsed arguments.
        """
        usage = formatActionAsString(action)

        if action.nargs == 0:
            return self._unparseArgumentValues(action, [value])

        if action.nargs in ("?", None):
            nargs = 1
            value = [value]
        else:
            nargs = action.nargs

        # Iterable so we can return remaining values to unparse.
        if not _typing.is_collection(value):
            raise ParserError(self.parser, f"{usage}: only takes iterable values. invalid value: {value}")

        elif nargs not in ("+", "*") and len(value) != nargs:
            raise ParserError(
                self.parser, f"{usage}: takes exactly {nargs} values, {len(value)} passed."
            )

        unparsedArgs = []

        if not isinstance(action, argparse._AppendAction):
            value = [value]

        for value_ in value:
            unparsedArgs.extend(self._unparseArgumentValues(action, value_))

        return unparsedArgs

    def _unparseArgumentValue(self, action: argparse.Action, value: _ValueT) -> str:
        """Unparse a value to pass to the given action.

        Args:
            action (argparse.Action): Action to unparse value for.
            value (_ValueT): Python value to unparse.

        Returns:
            str: Value unparsed from the given value passable to the given action.

        Raises:
            ParserError: If the action does not accept values, but a value was passed.
            ParserError: If the given value cannot be cast by the action's type.
        """
        default = self.parser.get_default(action.dest)

        if action.nargs == 0 and value != default:
            usage = formatActionAsString(action)
            raise ParserError(self.parser, f"{usage}: Does not accept values, yet {value!r} was passed.")

        try:
            self.parser._get_value(action, value)
        except Exception as exception:
            usage = formatActionAsString(action)
            message = f"{usage}: Could not cast {type(value)} value ({value!r}) to {action.type}."

            raise ParserError(self.parser, message) from exception

        unparsed = str(value)

        if self.quoteArgs:
            unparsed = shlex.quote(unparsed)

        return unparsed


def isOptionStringRequired(action: argparse.Action, value: _ValueT = None) -> bool:
    """Returns whether the given `value` means the `action` needs to be passed as a flag.

    Args:
        action (argparse.Action): Action to compare to value.
        value (_ValueT, optional): Boolean-able value to compare against action.

    Returns:
        bool: `True` if the `value` means the argument flag needs to be specified, else `False`.
    """
    if not action.option_strings:
        return False

    if action.nargs == 0:
        type_ = type(action.const)

        # Recast is necessary for python-like truthiness.
        #   Without recast: 2 == True --> False
        return action.const == type_(value)

    return True


def hasFallbackValue(action: argparse.Action) -> bool:
    """
    Returns:
        bool: Whether the action has a fallback value.
    """
    return action.default or action.const


def _getActionLists(
    parser: argparse.ArgumentParser, skipHelp: bool = True
) -> Tuple[List[argparse.Action], List[argparse.Action]]:
    """Returns the positional and keyword actions of the given parser as separate lists.

    Args:
        parser (argparse.ArgumentParser): Argument parser whose actions to parse.
        skipHelp (bool, optional): If `True` (default), skips the help action.

    Returns:
        list[argparse.Action], list[argparse.Action]: Positional actions and keyword actions.
    """
    positionalActions = []
    keywordActions = []
    for action in parser._actions:
        if skipHelp and isinstance(action, argparse._HelpAction):
            continue

        if action.option_strings:
            keywordActions.append(action)
        else:
            positionalActions.append(action)

    return positionalActions, keywordActions


class ChoiceEnumMeta(EnumMeta):
    """ Enum metaclass to simplify round-tripping with argparse. 

    See `ChoiceEnum` for an example.
    """
    def __call__(cls, value, *args, **kwargs):
        """ Override call to make values round-trippable."""
        member = cls._member_map_.get(value)
        if member:
            return super().__getitem__(value)

        return super().__call__(value, *args, **kwargs)


class ChoiceEnum(Enum, metaclass=ChoiceEnumMeta):
    """ Enum class to facilitate `ArgumentUnparser` compatible "choice" arguments.

    Examples:
        Define a choice enum:
            >>> class Choice(ChoiceEnum):
            ...     A = 0

        It allows is to pass a name instead of a value when calling:
            >>> str(Choice("A"))
            'A'

        Pass it to a parser's action:
            >>> parser = argparse.ArgumentParser()
            >>> action = parser.add_argument("choice", type=Choice,  choices=Choice)

        Arg is parsed to an enum correctly:
            >>> namespace = parser.parse_args(["A"])
            >>> namespace.choice
            <Choice.A: 0>

        And can be round tripped from an enum:
            >>> action.type(namespace.choice)
            <Choice.A: 0>

        Which allows ingestion by the unparser:
            >>> ArgumentUnparser(parser).unparseArgs(choice=namespace.choice)
            ['A']
    """
    def __str__(self) -> str:
        """Overriding str makes round-tripping easier.

        The unparser can simply str(enumValue) -> 'enumName'
        Which can then be reingested into a CLI (see examples in class docstring).

        Returns:
            str: The member's name.
        """
        return self.name


def formatActionAsString(action: argparse.Action) -> str:
    """Represents an action as a user-readable string.

    Exists because `action.format_usage()` doesn't cover positional actions.

    Args:
        action (argparse.Action): Parser `Action` to format.

    Returns:
        str: A user-readable string representation of an action.
    """
    if action.option_strings:
        actionUsage = action.format_usage()
    else:
        actionUsage = action.dest

    return actionUsage


if __name__ == "__main__":
    import doctest
    doctest.testmod()
