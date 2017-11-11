 # coding: utf-8

# public items
__all__ = [
    'youtube',
]

# standard library
from logging import getLogger
import webbrowser as web
from urllib.request import quote

# local constants
RESERVED = ';/?:@&=+$,'


def youtube(keyword=None):
    if keyword is None:
        web.open('https://www.youtube.com/watch?v=L_mBVT2jBFw')
    else:
        web.open(quote('https://www.youtube.com/results?search_query={}'.format(keyword), RESERVED))
