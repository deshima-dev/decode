 # coding: utf-8

# public items
__all__ = [
    'youtube',
    'superspec',
]

# standard library
from logging import getLogger
import webbrowser as web
from urllib.request import quote

# dependent packages
import decode as dc

# local constants
RESERVED = ';/?:@&=+$,'


def youtube(keyword=None):
    """Open youtube.

    Args:
        keyword (optional): Search word.
    """
    if keyword is None:
        web.open('https://www.youtube.com/watch?v=L_mBVT2jBFw')
    else:
        web.open(quote('https://www.youtube.com/results?search_query={}'.format(keyword), RESERVED))


def superspec(*args, **kwargs):
    def message(*args, **kwargs):
        print('Incompatible spectrometer!')

    for item in dir(dc):
        if item.startswith('__'):
            continue

        setattr(dc, item, message)

    message()
