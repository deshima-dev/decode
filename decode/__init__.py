# coding: utf-8

from . import utils
from .core import *
from . import io
from .logging import *

# default logger
import logging
logger = logging.getLogger('decode')
logger.propagate = False
setlogfile(logger=logger)
setloglevel(logger=logger)
del logging
