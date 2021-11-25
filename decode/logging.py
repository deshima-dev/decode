# coding: utf-8
__all__ = ["setlogger"]


# standard library
import logging
from copy import copy
from pathlib import Path

logger = logging.getLogger(__name__)


# module constants
DATEFORMAT = "%Y-%m-%d %H:%M:%S"
LOGFORMAT = "{asctime} | {levelname:8} | {funcName}: {message}"
DEFAULTLEVEL = "INFO"


# classes
class setlogger(object):
    def __init__(self, level=None, filename=None, overwrite=False, encoding="utf-8"):
        self.logger = logging.getLogger("decode")
        self.logger.addHandler(logging.NullHandler())
        # save current state
        self.oldhandlers = copy(self.logger.handlers)
        self.oldlevel = copy(self.logger.level)
        # set new state
        self.sethandlers(filename, overwrite, encoding)
        self.setlevel(level)

    def sethandlers(self, filename, overwrite, encoding):
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        if filename is None:
            handler = logging.StreamHandler()
        else:
            filename = str(Path(filename).expanduser())
            mode = "w" if overwrite else "a"
            handler = logging.FileHandler(filename, mode, encoding)

        formatter = logging.Formatter(LOGFORMAT, DATEFORMAT, style="{")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setlevel(self, level):
        level = DEFAULTLEVEL if level is None else level.upper()
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.handlers = self.oldhandlers
        self.logger.level = self.oldlevel
