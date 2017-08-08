# coding: utf-8

# public items
__all__ = [
    'setlogfile',
    'setloglevel',
]

# standard library
import logging
import os
import sys

# dependent packages
import decode as dc

# module constants
DATEFORMAT = '%Y-%m-%d %H:%M:%S'
FORMAT = '{asctime} {name:25} [{levelname}] {message}'


# functions
def setlogfile(filename=None, overwrite=False, *, logger=None):
    """Create a file where messages will be logged.

    Args:
        filename (str): A file name of logging.
            If not spacified, messages will be printed to stdout.
        overwrite (bool): Whether overwriting the file or not if it already exists.
        logger (logging.Logger, optional): A logger. Default is `dc.logger`.

    """
    if logger is None:
        logger = dc.logger

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if filename is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        filename = os.path.expanduser(filename)
        if overwrite:
            handler = logging.FileHandler(filename, 'w', encoding='utf-8')
        else:
            handler = logging.FileHandler(filename, 'a', encoding='utf-8')

    formatter = logging.Formatter(FORMAT, DATEFORMAT, style='{')
    handler.setFormatter(formatter)
    handler.setLevel(logger.level)
    logger.addHandler(handler)


def setloglevel(level='INFO', *, logger=None):
    """Set a logging level above which messages will be logged.

    Args:
        level (str or int): A logging level. Default is 'INFO'.
        logger (logging.Logger, optional): A logger. Default is `dc.logger`.

    References
        https://docs.python.jp/3/library/logging.html#logging-levels

    """
    if logger is None:
        logger = dc.logger

    logger.setLevel(level.upper())
    for handler in logger.handlers:
        handler.setLevel(level.upper())
