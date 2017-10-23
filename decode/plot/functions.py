# coding: utf-8

# public items
__all__ = [
    'plotspectrum',
]

# standard library
# from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits


# functions
def plotspectrum(dataarray, shape, **kwargs):
    if dataarray.type == 'dca':
        pass
    elif dataarray.type == 'dcc':
        xr.DataArray.dcc.plotspectrum(dataarray, shape, **kwargs)
    elif dataarray.type == 'dcs':
        pass
    else:
        pass
