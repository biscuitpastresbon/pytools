from __future__ import annotations

import contextlib
from functools import wraps
import logging
import sys
from typing import Callable


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def logStdout(name: str = None, stdoutLevel: int = None, stderrLevel: int = None):
    """Decorator to redirect all stdout/stderr to logging.

    Args:
        name (str, optional): Name to log from.
            Defaults to the module of the decorated function.
        stdoutLevel (int, optional): Logging level for stdout. Defaults to INFO.
        stderrLevel (int, optional): Logging level for stderr. Defaults to ERROR.

    >>> @logStdout
    ... def func():
    ...     print("hello")  # [module.to.func][INFO]: hello

    >>> @logStdout(name="stdout", stdoutLevel=logging.WARNING)
    ... def func():
    ...     print("hello")  # [stdout][WARNING]: hello
    """

    def wrapper(fn: Callable):
        stdoutLogging = StdoutLogging(
            name=name or fn.__module__,
            stdoutLevel=stdoutLevel,
            stderrLevel=stderrLevel,
        )
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stdoutLogging:
                fn(*args, **kwargs)
        return wrapped

    if callable(name):
        fn = name
        name = None
        return wrapper(fn)  # @logStdout

    return wrapper  # @logStdout()


class StdoutLogging:
    """ Context manager to redirect any stdout/err to logging.

    >>> with StdoutLogging("test") as stdoutLogging:
    ...     print("hello")                   # [INFO][test]: hello
    ...     print("hello", file=sys.stderr)  # [ERROR][test]: hello
    ...     stdoutLogging.stdoutLevel = logging.WARNING
    ...     print("hello")                   # [WARNING][test]: hello
    """
    class OutputLogger:
        """ Stdout/stderr logger class."""
        def __init__(self, name: str = None, level: int = None):
            """
            Args:
                name (str, optional): Logger name. Defaults to `root`.
                level (int, optional): Logging level. Defaults to INFO.
            """
            self.logger = logging.getLogger(str(name) or logging.root.name)
            self.level = level or logging.INFO

        @property
        def name(self) -> str:
            """The logger's name."""
            return self.logger.name

        @name.setter
        def name(self, name: str):
            """Set the logger's name."""
            self.logger.name = name

        def write(self, msg: str):
            """Buffer write overwrite method."""
            if msg and not msg.isspace():
                self.logger.log(self.level, msg)

        def flush(self):
            """Flush STD out/err. Note that this assumes that they haven't been remapped."""
            sys.stdout.flush()
            sys.stderr.flush()

    @property
    def stdoutLevel(self) -> int:
        """The context manager's stdout logging level."""
        return self._stdoutLogger.level

    @stdoutLevel.setter
    def stdoutLevel(self, level: int):
        """Set the context manager's loggers' level."""
        self._stdoutLogger.level = level

        if not self._withinContext:
            log.info(f"Setting initial level: {level}")
            self._initialOutputLevel = level

    @property
    def stderrLevel(self) -> int:
        """The context manager's stderr logging level."""
        return self._stderrLogger.level

    @stderrLevel.setter
    def stderrLevel(self, level: int):
        """Set the context manager's loggers' level."""
        self._stderrLogger.level = level

        if not self._withinContext:
            log.info(f"Setting initial level: {level}")
            self._initialErrorLevel = level

    @property
    def name(self) -> str:
        """The context manager's loggers' name."""
        return self._stdoutLogger.name

    @name.setter
    def name(self, name: str):
        """Set the context manager's loggers' name."""
        for logger in self._loggers:
            logger.name = name

        if not self._withinContext:
            log.info(f"Setting initial name: {name}")
            self._initialName = name

    def __init__(self, name: str = None, stdoutLevel: int = None, stderrLevel: int = None):
        """
        Args:
            name (str, optional): Name to give the std logger. Defaults to `root`.
            stdoutLevel (int, optional): Level to give stdout logging. Defaults to INFO.
            stderrLevel (int, optional): Level to give stderr logging. Defaults to ERROR.
        """
        stderrLevel = logging.ERROR if stderrLevel is None else stderrLevel
        self._stderrLogger = self.OutputLogger(name, level=stderrLevel)
        self._stdoutLogger = self.OutputLogger(name, level=stdoutLevel)
        self._loggers = [self._stderrLogger, self._stdoutLogger]

        self._stderrRedirector = contextlib.redirect_stderr(self._stderrLogger)
        self._stdoutRedirector = contextlib.redirect_stdout(self._stdoutLogger)

        self._initialName = self.name
        self._initialOutputLevel = self.stdoutLevel
        self._initialErrorLevel = self.stderrLevel
        self._withinContext = False

    def __call__(self, name: str = None, level: int = None) -> StdoutLogging:
        """Only re-call context with `with`.

        Args:
            name (str, optional): Modifies name of the std logger. Defaults to None.
            level (int, optional): Modifies level of stdout logging. Defaults to None.
        """
        withinContext = self._withinContext
        self._withinContext = True

        if name:
            self.name = name

        if level:
            self.stdoutLevel = level

        self._withinContext = withinContext
        return self

    def __enter__(self) -> StdoutLogging:
        """ Context manager enter.

        Returns:
            StdoutLogging: Context manager instance.
        """
        self._withinContext = True
        self._stdoutRedirector.__enter__()
        self._stderrRedirector.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Context manager exit."""
        self.name = self._initialName
        self.stdoutLevel = self._initialOutputLevel
        self.stderrLevel = self._initialErrorLevel

        # let contextlib do any exception handling here
        self._stdoutRedirector.__exit__(exc_type, exc_value, traceback)
        self._stderrRedirector.__exit__(exc_type, exc_value, traceback)
        self._withinContext = False
