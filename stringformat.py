"""
Low-level string formatting utilities.

Do *NOT* import anything else than built-ins OR other `onpycommon` modules with only built-in imports.

To extend `FormatString` to work with new formatting syntaxes, create a token factory by subclassing `AbstractTokenFactory`

"""
# /usr/bin/env python3
from __future__ import annotations

import abc
import dataclasses
import logging
import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Type, TypeVar, Union, Sequence

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


__all__ = (
    'log',
    'StringFormatError',
    'InvalidFormatError',
    'ParsingError',
    'extractValues',
    'FormatString',
    'FormatToken',
    'FormatType',
    'AbstractTokenFactory',
    'OldStyleTokenFactory',
    'NewStyleTokenFactory',
    'RepeatedSymbolTokenFactory',
)

T = TypeVar("T")
ClassOrInst_T = Union[T, Type[T]]


def formatValues(formatString: str, *args, **kwargs) -> str:
    """Formats a given string's format token's with the given values.

    Harnesses `str.format`, extending the functionality to other supported syntaxes.

    Example:
        `str.format` can only format `{}` tokens:
            >>> str.format("Rendering ({:.0%}) %s.####", .65, "image", 4)
            'Rendering (65%) %s.####'

        Whereas `formatValues` format all tokens:
            >>> formatValues("Rendering ({:.0%}) %s.####", .65, "image", 4)
            'Rendering (65%) image.0004'

    Args:
        formatString (str): String containing tokens to format.
        *args, **kwargs: Passed to `str.format(*args, **kwargs)`

    Returns:
        str: The format string with tokens filled with the given values.
    """
    newStyle = FormatString(formatString).translate(NewStyleTokenFactory)
    formatted = newStyle._string.format(*args, **kwargs)

    return formatted


def extractValues(formatString: str, string: str, cast: bool = True, strict: bool = True) -> Optional[List[T]]:
    """Parses the given `string` according to the `formatString` and
    returns a tuple of the extracted values.

    Args:
        formatString (str): String with format tokens (eg: %s, %04d) to extract values from `string`.
        string (str): String that you want to parse using the `formatString` pattern.
        cast (bool, optional): Pass `False` to skip casting values to their matched type,
            and return the raw matched strings. Default is `True`.
        strict (bool, optional): Pass `False` to return `None` instead of raising.
            Default is `True`.

    Returns:
        Optional[List]: Matched values from the `formatString`/`string` match.
            Returns `None` if the pattern doesn't match and `strict` is `False`.
    """
    values = FormatString(formatString).extractValues(string, cast=cast, strict=strict)
    return values


class StringFormatError(Exception):
    """Raise when any string-formatting related errors occur."""
    pass


class InvalidFormatString(StringFormatError):
    """Raise when invalid string format expression causes a parsing issue."""
    def __init__(self, message=None, match: re.Match=None, *args: object):
        message_ = ''
        if match:
            message_ += (
                fr'Invalid format expression "{match.group()!s}" '
                f'at position [{match.start()}:{match.end()}]:\n'
                fr'{match.string!s}'
                '\n'
                f'{"^": >{match.start() + 1}}\n'
            )
        if message:
            message_ += message
        super().__init__(message_, *args)


class ParsingError(StringFormatError):
    """Raise when an error occurs because a pattern was incorrectly parsed internally,
    and causes matching issues.

    These errors should be addressed by fixing regex patterns in this module.
    """
    pass


class FormatString:
    """Represents a format string containing format expressions (eg: %s, %04d).

    Allows parsing these expressions into tokens, matching other strings against it
    and extracting values from said strings.

    Args:
        formatString (str): A string containing format expressions. (eg: '%s.%05d.png')
        tokenFactories (Sequence[Type[AbstractTokenFactory]]):
            Pass a factory for each expression language you want to check for.
            Default is [OldStyleTokenFactory], for % style expressions only.

    Examples:
        >>> fstring = FormatString("%s {name} - #### - rendering ({percent:.0%})")

        Get all groups:
            >>> fstring.extractValues("marinette beauty - 0004 - rendering (45%)")
            ['marinette', 'beauty', 4, 0.45]

        Get named groups:
            >>> fstring.groupDict("marinette beauty - 0004 - rendering (45%)")
            {'name': 'beauty', 'percent': 0.45}

        Translate to another syntax:
            >>> fstring.translate(NewStyleTokenFactory)
            FormatString("{!s} {name} - {:04d} - rendering ({percent:.0%})")

        Translate a specific token:
            >>> fstring.translateToken(fstring.formatTokens[-2], RepeatedSymbolTokenFactory("#", pad=2))
            FormatString("%s {name} - ## - rendering ({percent:.0%})")

            >>> fstring.translateToken(fstring.formatTokens[-1], NewStyleTokenFactory("d", fill="0", pad=2))
            FormatString("%s {name} - #### - rendering ({percent:02d})")
    """

    def __init__(self, formatString: str, tokenFactories: Sequence[Type[AbstractTokenFactory]] = None) -> None:
        self._string: str = str(formatString)

        self.tokenFactories = list(tokenFactories or TOKEN_FACTORIES)

    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the format string.
        """
        return self._string

    def __repr__(self) -> str:
        """
        Returns:
            str: A python construction expression for the format string.
        """
        return f'{self.__class__.__name__}("{self._string}")'

    @property
    @lru_cache()
    def _escapedString(self) -> str:
        """Returns the format string, with all special regex characters escaped."""
        escapedString = re.escape(self._string)
        return escapedString

    @property
    @lru_cache()
    def formatTokens(self) -> List[FormatToken]:
        """Returns string formatting tokens from the format string as `FormatToken` objects.

        Raises:
            ParsingError: If discovered tokens overlap.

        Returns:
            List[FormatToken]: String formatting tokens matched in the format string.
        """
        tokens = []
        for tokenFactory in self.tokenFactories:
            for token in tokenFactory.generateTokensFromString(self._string):
                overlappingTokens = [
                    other for other in tokens
                    if token.start <= other.end and token.end >= other.start
                ]
                if overlappingTokens:
                    raise ParsingError(
                        f"Ambiguous {tokenFactory.__name__} token {token} overlaps with previously matched tokens"
                        f" {overlappingTokens}. Registered factories may not be compatible."
                    )
                tokens.append(token)

        tokens = sorted(tokens, key=lambda token: token.start)

        return tokens

    def addFactory(self, factoryClass: Type[AbstractTokenFactory]):
        """
        Registers a token factory to the `FormatString` if it is not already present.
        Use this to allow `FormatString` to discover several different format syntaxes.

        Args:
            factoryClass (Type[AbstractTokenFactory]):
                A subclass of `AbstractTokenFactory` (not instance).
        """
        if factoryClass not in self.tokenFactories:
            self.tokenFactories.append(factoryClass)

    def removeFactory(self, factoryClass: Type[AbstractTokenFactory]):
        """
        Removes a token factory from the `FormatString` if it is present.

        Args:
            factoryClass (Type[AbstractTokenFactory]):
                A subclass of `AbstractTokenFactory` (not instance).
        """
        if factoryClass in self.tokenFactories:
            self.tokenFactories.remove(factoryClass)

    def extractValues(self, string: str, cast: bool = True, strict: bool = True) -> Optional[List[T]]:
        """Parses the given string according to the format string and
        returns a tuple of the extracted values.

        Args:
            string (str): String that you want to parse using the `formatString` pattern.
            cast (bool, optional): Pass `False` to skip casting values to their matched type,
                and return the raw matched strings. Default is `True`.
            strict (bool, optional): Pass `False` to return `None` instead of raising.
                Default is `True`.

        Returns:
            Optional[List]: Matched values from the `formatString`/`string` match.
                Returns `None` if the pattern doesn't match and `strict` is `False`.
        """
        match = self.match(string, strict=strict)

        if match is None:
            return

        values = match.groups()
        if not cast:
            return values

        return list(self._iterCastValuesFromTokens(values))

    def groupDict(
        self, string: str, cast: bool = True, strict: bool = True
    ) -> Optional[Dict[str, Union[T, str]]]:
        """Parses the given string according to the format string's named tokens and
        returns a dictionary of extracted values by group name.

        Args:
            string (str): String that you want to parse using the `formatString` pattern.
            cast (bool, optional): Pass `False` to skip casting values to their matched type,
                and return the raw string values. Default is `True`.
            strict (bool, optional): Pass `False` to return `None` instead of raising.
                Default is `True`.

        Returns:
            Dict[str, Any | str] | None: Token names with their corresponding values.
                Returns raw strings values if `cast` is `False`.
                Returns `None` if the pattern doesn't match and `strict` is `False`.
        """
        match = self.match(string, strict=strict)

        if match is None:
            return

        values = match.groups()

        groupDict = {}
        for token, value in zip(self.formatTokens, values):
            if not token.groupName:
                continue

            value = token.castValue(value) if cast else value
            groupDict[token.groupName] = value

        return groupDict

    def match(self, string: str, strict: bool = True) -> re.Match:
        """Parses the given string according to the format string and returns a match.

        Args:
            string (str): String that you want to parse using the `formatString` pattern.
            strict (bool, optional): Pass `False` to return `None` instead of raising.
                Default is `True`.

        Raises:
            StringFormatError: If `string` does not match the regex pattern
                converted from `formatString`.

        Returns:
            re.Match: Match for the given string.
        """
        pattern = self.asRegex()
        log.debug(f'{pattern!s}.match("{string!s}")')
        match = pattern.match(string)

        if match is None:
            noMatchMessage = f"{string!r} does not match format string {self._string!r}. No value could be extracted."
            if not strict:
                log.debug(noMatchMessage)
                return
            else:
                raise StringFormatError(noMatchMessage)

        return match

    def asRegex(self) -> re.Pattern:
        r"""Returns the format string as a regex pattern
        by escaping the string and replacing format tokens
        with their corresponding regex patterns.

        Returns:
            re.Pattern: A compiled regex pattern.

        >>> FormatString('main_main_albedo_left.%05d.exr').asRegex().pattern
        'main_main_albedo_left\\.([0\\d]{5,})\\.exr'
        """
        regexString = ''
        lastPosition = 0
        lastIndex = len(self.formatTokens) - 1

        for index, token in enumerate(self.formatTokens):
            regexString += fr"{token.string[lastPosition:token.start]}({token.regex.pattern})"
            lastPosition = token.end

            if index == lastIndex:
                regexString += token.string[lastPosition:]

        log.debug(fr"regexString={regexString!s}")
        return re.compile(regexString)

    def _iterCastValuesFromTokens(self, rawValues: List[str]) -> Iterator[T]:
        """Cast the given values, in order, according to the format string's parsed token types.
        Depends on the order of values being the same as the order of tokens.

        Args:
            rawValues (List[str]): String values that need to be casted or converted based
                on their associated `FormatToken`.

        Raises:
            ParsingError: If a mismatch between the converted expression
                and the extracted group values (type, length, pattern...) is found.
                (If it happens, it is a bug and should probably be fixed in the FORMAT_TYPES patterns.)

        Yields:
            Values cast by their associated token's recorded type.
        """
        if len(rawValues) > len(self.formatTokens):
            raise ParsingError(
                f"Too many values to match: {rawValues}. "
                f"Should be {len(self.formatTokens)}. "
                "Please check internal regex patterns."
            )
        elif len(rawValues) < len(self.formatTokens):
            raise ParsingError(
                f"Not enough values to match: {rawValues}. "
                f"Should be {len(self.formatTokens)}. "
                "Please check internal regex patterns."
            )

        for value, token in zip(rawValues, self.formatTokens):
            castValue = token.castValue(value)
            yield castValue

    def translate(self, factory: Type[AbstractTokenFactory]) -> FormatString:
        """Translate the format string to the given factory's syntax.

        >>> FormatString("mystring.####.png").translate(NewStyleTokenFactory)._string
        'mystring.{:04d}.png'

        Args:
            factory (Type[AbstractTokenFactory]): Token factory type to translate to.

        Returns:
            FormatString: The translated format string object.
        """
        fstring = self

        while True:
            for token in fstring.formatTokens:
                factoryType = type(token.factory)

                if factoryType != factory:
                    break
            else:
                break

            fstring = fstring.translateToken(token, factory)

        return fstring

    def translateToken(self, token: FormatToken, factory: ClassOrInst_T[AbstractTokenFactory]) -> FormatString:
        """Translate the given token to the given factory, or factory type's syntax.

        Args:
            token (FormatToken): Token to translate.
            factory (ClassOrInst_T[AbstractTokenFactory]): Class or instance to translate to.
                If class: Only apply the factory's syntax, preserving the token's properties.
                If instance: Apply the factory's syntax AND its properties.

        Raises:
            StringFormatError: If the token and factory have incompatible types.

        Returns:
            FormatString: The updated format string object.

        >>> fstring = FormatString("image.####.png")
        
        >>> fstring.translateToken(fstring.formatTokens[0], NewStyleTokenFactory)
        FormatString("image.{:04d}.png")
        """
        if not isinstance(factory, AbstractTokenFactory):
            formatType = next(iter(factory.findCompatibleTypes(token.factory.type)), None)

            if formatType is None:
                raise StringFormatError(f"Token type {token.factory.type} has no compatible format type in {factory}")

            factory = factory(
                formatType.identifier, 
                fill=token.factory.fill, 
                pad=token.factory.pad
            )

        if factory == token.factory:
            return self

        tokenString = factory.format(groupName=token.groupName)
        fstring = self.replaceToken(token, tokenString)

        return fstring

    @staticmethod
    def _unescapeRegex(string) -> str:
        """Reverts `re.escape()`.

        Returns:
            str: An unescaped string.
        """
        unescaped = re.sub(r'\\(.)', r'\1', str(string))

        return unescaped

    def replaceToken(self, search: FormatToken, replace: Union[str, FormatToken]) -> FormatString:
        """Search and replace a given token.

        Args:
            search (FormatToken): Token to search for.
            replace (FormatToken): Token or string to replace with.

        Returns:
            FormatString: The updated format string object.
        """
        for token in self.formatTokens:
            if token == search:
                break
        else:
            raise StringFormatError(f"{self} has no such token: {search}")

        if hasattr(replace, "string"):
            string = replace.string
        else:
            string = str(replace)

        beforeString = self._unescapeRegex(search.string[:search.start])
        afterString = self._unescapeRegex(search.string[search.end:])

        translatedString = f"{beforeString}{string}{afterString}"

        log.debug(f"{self._string} -> {translatedString}")
        fstring = self.__class__(translatedString)

        return fstring


class FormatToken:
    """Represents a string formatting expression (eg: %s, %04d) token within a larger string.

    Is what `re.Match` is to a `re.Pattern`, with a start position and end position of the token.

    Args:
        match (re.Match): Match to construct token from.
        regex (re.Pattern): The regex pattern that matches values corresponding to the token.
        formatType (FormatType): The format type for the token.
    """
    class TypeCastError(TypeError):
        """Raise this error when a `FormatToken` cannot cast a given value."""
        def __init__(self, token: FormatToken, value, message=None, *args):
            message = (
                f"Error casting {value} with {token!r} "
                f"at position [{token.start}:{token.end}]:\n"
                fr"{token.string!s}"
                "\n"
                f"{'^': >{token.start + 1}}\n"
            )
            if message:
                message += message
            super().__init__(message, *args)

    def __init__(self, match: re.Match, factory: AbstractTokenFactory):
        self._match = match
        self.factory = factory

    def __str__(self) -> str:
        return f"{self.group}({self.valueType.__name__})[{self.start}:{self.end}]"

    def __repr__(self) -> str:
        """Return a string representation of the token.

        Returns:
            str: A python construction expression for the object.
        """
        return (
            f"{self.__class__.__name__}("
                f"name={self.groupName!r}, "
                fr"regex='{self.regex.pattern}', "
                f"valueType={self.valueType.__name__}, "
                f"start={self.start}, "
                f"end={self.end}, "
                fr"match='{self.group}'"
            r")"
        )

    @property
    def groupName(self) -> str:
        """
        Returns:
            str: The match's group name.
        """
        return self._match.groupdict().get("name")

    @property
    def string(self) -> str:
        """
        Returns:
            str: The full string the token was matched in.
        """
        return self._match.string

    @property
    def group(self) -> str:
        """
        Returns:
            str: The matched sub-string.
        """
        return self._match.group()

    @property
    def regex(self) -> re.Pattern:
        """
        Returns:
            re.Pattern: The regex pattern that matches values corresponding to the token.
        """
        return self.factory.regex

    @property
    def valueType(self) -> Type:
        """
        Returns:
            Type: The python type appropriate to cast values matched by the token's regex.
        """
        return self.factory.type.valueType

    @property
    def start(self) -> int:
        """
        Returns:
            int: The start position of the format token in its total format string.
        """
        return self._match.start()

    @property
    def end(self) -> int:
        """
        Returns:
            int: The end position of the format token in its total format string.
        """
        return self._match.end()

    def castValue(self, value: str):
        """Checks if `value` matches the token's regex, then
        attempts to cast it to the token's value type.

        Args:
            value (str): String value matched by the token's regex.

        Raises:
            FormatToken.TypeCastError: When `value` doesn't match the token regex.
            FormatToken.TypeCastError: When `value` causes a TypeError when casting its type.

        Returns:
            Any: Returns the value, cast into the token's type.
        """
        if self.regex.match(value) is None:
            raise self.TypeCastError(self, value)

        try:
            castValue = self.valueType(value)
        except TypeError as typeError:
            raise ParsingError(self, value) from typeError

        return castValue


@dataclasses.dataclass(frozen=True)
class FormatType:
    """Represents a format expression type.

    Example:
        With old-style % string format syntax, it corresponds to the 'd' in '%04d'.

    Args:
        identifier (str): The character denoting format type (eg: 'd' or 'i' in % format syntax).
        regexTemplate (str): Pattern matching the expression type's specific syntax.
        valueType (Type): Type to use to type-cast values matching this type.
        niceName (str): The nice name of the format type.
            Only use alphanumeric characters, as can be used for matching groups in regex.
    """
    identifier: str
    regexTemplate: str
    valueType: Type
    niceName: str


class AbstractTokenFactory(abc.ABC):
    """String formatting expression (eg: %s, %04d) token factory.

    Is a dynamic `re.match` which adapts based on the token's format type.
    The format type, among other information, is parsed from a given string
    using `generateTokensFromString`.

    Format types need to be defined in the `FORMAT_TYPES` enum as an instance of `FormatType`.

    Args:
        typeSymbol (str): The type name of format expression. Corresponds to the 'd' in '%04d'.

    Raises:
        InvalidFormatString: If a non-existent type name is given.
    """
    @property
    @abc.abstractmethod
    def FORMAT_TYPES(self) -> Dict[str, FormatType]:
        """Declares all supported token format types for the factory's expression syntax/language.

        This is where to define patterns to match tokens against a FORMATTED string,
        use `getDiscoveryPattern` to define expressions for format expression strings.

        (expression string: '%04d', formatted string: '0001')

        Returns:
            Dict[str, FormatType]: A mapping of `FormatType` by their respective unique symbol.
                (eg: 'd' is the unique symbol for a '%d' expression in old-style % string formatting.)
        """
        pass

    def __init__(self, typeSymbol: str, fill: str = None, pad: int = 1):
        formatType = self.FORMAT_TYPES.get(typeSymbol)
        if formatType is None:
            availableTypes = [f"'{formatType.identifier}'" for formatType in self.FORMAT_TYPES.values()]
            raise InvalidFormatString(f"Unknown format type '{typeSymbol}'. "
                                      f"Please use one of {', '.join(availableTypes)}.")

        self._fill = None

        self.type = formatType
        self.pad = pad

        if fill is not None:
            self.fill = fill

        if fill and self.type.valueType != str and not fill.isdigit():
            raise ValueError(f"'{fill}' is an invalid fill value for numeric token {self!r}.")

    def __repr__(self) -> str:
        """Return a string representation of the token factory.

        Returns:
            str: A python construction expression for the object.
        """
        return (
            f"{self.__class__.__name__}("
                f"type={self.type.niceName!r}, "
                f"fill={self.fill!r}, "
                f"pad={self.pad!r}"
            ")"
        )

    def __eq__(self, other: AbstractTokenFactory) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        hashKey = f"<class '{self.__class__.__name__}' type={hash(self.type)};fill={self.fill};pad={self.pad}>"

        return hash(hashKey)

    @property
    def fill(self) -> str:
        """Return object's fill.

        Returns:
            : Object's fill
        """
        if self._fill:
            return self._fill

        if self.type.valueType != str:
            return "0"

        return ""

    @fill.setter
    def fill(self, value: str):
        self._fill = str(value)

    @property
    @lru_cache()
    def regex(self) -> re.Pattern:
        r"""Returns a regex expression for the tokens this factory will generate.

        >>> OldStyleTokenFactory('d', fill='0', pad=4).regex.pattern
        '[0\\d]{4,}'
        """
        stringPattern = self.type.regexTemplate.format(fill=self.fill, pad=self.pad)
        pattern = re.compile(stringPattern)

        return pattern

    @classmethod
    def findCompatibleTypes(cls, otherType: FormatType) -> List[FormatType]:
        compatibleTypes = []

        for formatType in cls.FORMAT_TYPES.values():
            if formatType.valueType == otherType.valueType:
                compatibleTypes.append(formatType)

        return compatibleTypes

    @classmethod
    @abc.abstractmethod
    def getDiscoveryPattern(self) -> re.Pattern:
        """Returns a `re.Pattern` to match tokens using the factory's syntax/language.

        This is the pattern to match against a format EXPRESSION string,
        use `FORMAT_TYPES` to define expressions for formatted strings.

        (expression string: '%04d', formatted string: '0001')

        Bear in mind, format strings need to be escaped, so make sure to match against
        escaped characters. If matching '#', match '\#' instead.

        Example:
            The discovery pattern for matching any ##.. tokens would be:
            r'((?:\\\#)+)'

        Returns:
            re.Pattern: A pattern to match against an escaped format expression string.
        """
        pass

    @classmethod
    def generateTokensFromString(cls, string: str) -> Iterator[FormatToken]:
        """Returns all parsed string formatting tokens in a given string.

        Args:
            string (str): A string to identify tokens in.

        Returns:
            List[FormatToken]: String formatting tokens parsed from the string.
        """
        string = re.escape(string)
        discoveryPattern = cls.getDiscoveryPattern()
        log.debug(fr"re.finditer('{discoveryPattern.pattern!s}', '{string!s}')")
        for match in discoveryPattern.finditer(string):
            token = cls.generateTokenFromMatch(match)
            yield token

    @classmethod
    @abc.abstractmethod
    def generateTokenFromMatch(cls, match: re.Match) -> FormatToken:
        """Returns a parsed string formatting token from a `re.Match` object.

        Args:
            match (re.Match): A regex match to identify a single token with.

        Returns:
            FormatToken: String formatting token parsed from the match.
        """
        pass

    @abc.abstractmethod
    def format(self, groupName: str = None) -> str:
        raise NotImplementedError


class OldStyleTokenFactory(AbstractTokenFactory):
    """Old-style % string formatting expression (eg: %s, %04d) token factory.

    Is a dynamic `re.match` which adapts based on the token format type.
    Eg: A '%d' will not match the same way a '%s' will.

    https://docs.python.org/3/library/string.html#format-specification-mini-language

    Args:
        typeName (str): The type name of format expression. Corresponds to the 'd' in '%04d'.
        fill (str, optional): The "fill" character of the expression. Corresponds to the '0' in '%04d'.
            Defaults to ''.
        pad (pad, optional): The max amount of fill characters to use. Corresponds to the 4 in '%04d'.
            Defaults to 1.

    >>> OldStyleTokenFactory('s')
    OldStyleTokenFactory(type='string', fill='', pad=1)
    """

    @staticmethod
    def _hexadecimal(string: str) -> int:
        """
        Returns:
            int: A hexadecimal integer representing the given string.
        """
        return int(string, 16)

    @staticmethod
    def _octal(string: str) -> int:
        """
        Returns:
            int: An octal integer representing the given string.
        """
        return int(string, 8)

    FORMAT_TYPES = {
        'd': FormatType('d', r'[{fill}\d]{{{pad},}}', int, 'signed integer decimal'),
        'i': FormatType('i', r'[{fill}\d]{{{pad},}}', int, 'signed integer decimal'),
        'o': FormatType('o', r'[{fill}0-7]{{{pad},}}', _octal, 'signed octal value'),
        'x': FormatType('x', r"0x[{fill}0-9a-f]{{{pad},}}", _hexadecimal, 'signed hexadecimal (lowercase)'),
        'X': FormatType('X', r"0x[{fill}0-9A-F]{{{pad},}}", _hexadecimal, 'signed hexadecimal (uppercase)'),
        'f': FormatType('f', r"[{fill}\d]+\.?\d{{{pad}}}", float, 'floating point decimal format (lowercase)'),
        'F': FormatType('F', r"[{fill}\d]+\.?\d{{{pad}}}", float, 'floating point decimal format (uppercase)'),
        's': FormatType('s', r'.*', str, 'string'),
    }
    """All supported old-style % format types.

    Example:
        'image.%04d.png' contains a ':04d' token, which matches the `Decimal format (d)` format type.

        %04d regex representation  is '\d{4,}'  (eg: '0004').
        %04d callable to cast its value is `int`.

    Note:
        We need the comma in {n,} for numbers to match when the number goes beyond padding.
        For example, '0004' is a valid string for '%04d', but so is '4444444'.
    """

    @classmethod
    @lru_cache()
    def getDiscoveryPattern(cls) -> re.Pattern:
        """A pattern matching any old-style % formatting tokens.

        Returns:
            re.Pattern: A pattern to match against an escaped format expression string.
        """
        discoveryPattern = re.compile(
            r'%(?:'  # Optional expression syntax group.
                r'(?:\\.)?'  # Some format expression syntaxes (eg. float) have a leading '.' character.
                r'(?P<fill>\w)?'  # Corresponds to the '0' in '%04d'. Optional.
                r'(?P<pad>\d+)'  # Corresponds to the 4 in '%04d'.
            ')?'
            fr'(?P<formatType>[{"".join(cls.FORMAT_TYPES)}])'  # Corresponds to the 'd' in '%04d'.
        )
        return discoveryPattern

    @classmethod
    def generateTokenFromMatch(cls, match: re.Match) -> FormatToken:
        """Returns a parsed string formatting token from a `re.Match` object.

        Raises:
            InvalidFormatString: If an invalid string format expression causes a parsing issue.

        Returns:
            FormatToken: String formatting token parsed from the match.
        """
        formatType = match["formatType"] or "s"
        fill = match["fill"] or ""

        if cls.FORMAT_TYPES.get(formatType).valueType != str:
            fill = "0"

        try:
            pad = int(match["pad"] or "1")
        except TypeError:
            raise InvalidFormatString(
                match, fr"Padding needs to be an int, not '{match['pad']}'."
            )

        factory = cls(formatType, fill=fill, pad=pad)
        token = FormatToken(match, factory)
        return token

    def format(self, groupName: str = None) -> str:
        string = ""

        if self.fill or self.pad > 1:
            string = f"{string}.{self.fill}{self.pad}"

        string = f"%{string}{self.type.identifier}"

        return string


class NewStyleTokenFactory(OldStyleTokenFactory):
    """New-style {} string formatting expression (eg: {0}, {:04d}) token factory.

    Args:
        typeName (str): The type name of format expression. Corresponds to the 'd' in '{:04d}'.
            Defaults to 's'.
        fill (str, optional): The "fill" character of the expression. Corresponds to the '0' in '{:04d}'.
            Defaults to ''.
        pad (pad, optional): The max amount of fill characters to use. Corresponds to the 4 in '{:04d}'.
            Defaults to 1.
    """
    @staticmethod
    def _binary(string: str) -> int:
        return int(string, 2)

    @staticmethod
    def _percent(string: str) -> int:
        return float(string.strip('%')) / 100

    FORMAT_TYPES = {
        "b": FormatType("b", r"[{fill}\d]{{{pad},}}", _binary, "binary"),
        "e": FormatType("e", r"\d+\.[{fill}\d]{{{pad},}}E\+\d{{2}}", float, "scientific (lowercase)"),
        "E": FormatType("E", r"\d+\.[{fill}\d]{{{pad},}}E\+\d{{2}}", float, "scientific (uppercase)"),
        "%": FormatType("%", r"[{fill}\d]+\.?\d{{{pad}}}%", _percent, "percentage"),
        **OldStyleTokenFactory.FORMAT_TYPES
    }
    """All supported new-style '{}' format types.

    Example:
        'image.{:04d}.png' contains a ':04d' token, which matches the `Decimal format (d)` format type.

        {:04d} regex representation  is '\d{4,}'  (eg: '0004').
        {:04d} callable to cast its value is `int`.
    """

    def __init__(self, typeName: Optional[str] = None, **kwargs):
        if typeName is None:
            typeName = "s"

        super().__init__(typeName, **kwargs)

    @classmethod
    @lru_cache()
    def getDiscoveryPattern(cls) -> re.Pattern:
        """A pattern matching any new-style {} formatting tokens.

        TODO: Look at subclassing `string.Formatter`.
            https://docs.python.org/3/library/string.html#custom-string-formatting

        Returns:
            re.Pattern: A pattern to match against an escaped format expression string.
        """
        names = "".join(cls.FORMAT_TYPES)
        # TODO: Reverse conversion when parsing tokens?
        discoveryPattern = re.compile(
            r'\\{'                         # Opening bracket.

                r'(?P<name>\w+?)?'             # Name or index of token.

                r'(?:!'                        # Conversion character.
                    r'(?P<conversion>[ras])'       # Conversion mode.
                r')?'                          # Close conversion group.

                r'(?::'                        # Format spec character.
                    r'(?:\\.)?'                    # Precision character for float-like format types.
                    r'(?P<fill>\w)?'               # Corresponds to the '0' in '{:04d}'.
                    r'(?P<pad>\d+)?'               # Corresponds to the 4 in '{:04d}'.
                    fr'(?P<formatType>[{names}])'  # Corresponds to the 'd' in '{:04d}'.
                r')?'                          # Close format spec group.

            r'\\}'                         # Closing bracket.
        )

        return discoveryPattern

    def format(self, groupName: str = None) -> str:
        """Format as a string token.

        TODO: This should be in `FormatToken` class.
        """
        string = ""

        if groupName is not None:
            string = f"{groupName}"

        if self.fill or self.pad > 1:
            string = f"{string}:{self.fill}{self.pad}"

        if self.type.valueType in (str, repr):
            string = f"{string}!{self.type.identifier}"
        else:
            string = f"{string}{self.type.identifier}"

        string = f"{{{string}}}"

        return string


class RepeatedSymbolTokenFactory(AbstractTokenFactory):
    """Factory for string format expressions using repeating symbol syntax (eg: 'image.####.png').

    Example:
        >>> RepeatedSymbolTokenFactory('#')
        RepeatedSymbolTokenFactory(type='integer', fill='0', pad=1)
    """

    FORMAT_TYPES = {
        '#': FormatType('#', r'[{fill}\d]{{{pad},}}', int, 'integer'),
    }
    """
    All supported symbol repetition format types.

    Example:
        'image.####.png' contains a '####' token, which matches the '#' format type.
        The format type defines that the regex representation of '####' is r'\d{4,}'
        and can be cast as an `int`.
    """

    @classmethod
    @lru_cache()
    def getDiscoveryPattern(cls) -> re.Pattern:
        """Pattern to identify any repeated symbols defined in the `FORMAT_TYPES`.

        Example:
            Matches '####' in 'image.####.png'.

        Returns:
            re.Pattern: A pattern to match against an escaped format expression string.
        """
        typePatterns = []
        for typeSymbol, formatType in cls.FORMAT_TYPES.items():
            typePatterns.append(f'(?P<{formatType.niceName}>'
                # Adding an extra \ in front of the escaped name,
                # because the token's string should also be pre-escaped.
                # eg: '(?:\\\#)+'
                fr'(?:\{re.escape(typeSymbol)})+'
            ')')
        discoveryPattern = '|'.join(typePatterns)
        discoveryPattern = re.compile(discoveryPattern)
        return discoveryPattern

    @classmethod
    def generateTokenFromMatch(cls, match: re.Match) -> FormatToken:
        """Returns a parsed string formatting token from a `re.Match` object.

        Raises:
            InvalidFormatString: If an invalid string format expression causes a parsing issue.

        Returns:
            FormatToken: String formatting token parsed from the match.
        """
        unescapedMatch = match.group().replace('\\', '')
        formatType = ''.join(set(unescapedMatch))
        pad = unescapedMatch.count(formatType)
        factory = cls(formatType, pad=pad)
        token = FormatToken(match, factory)
        return token

    def format(self, groupName: str = None) -> str:
        string = self.type.identifier * self.pad

        return string


TOKEN_FACTORIES = (
    OldStyleTokenFactory,
    NewStyleTokenFactory,
    RepeatedSymbolTokenFactory,
)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
