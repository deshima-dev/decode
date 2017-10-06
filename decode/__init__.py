# coding: utf-8

from .array import *
from .logging import *
from . import utils

# default logger
import logging
logger = logging.getLogger('decode')
logger.propagate = False
setlogfile(logger=logger)
setloglevel(logger=logger)
del logging
