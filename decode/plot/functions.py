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
    """Plot a spectrum diagram.

    Args:
        dataarray (xarray.DataArray): Dataarray with which the spectrum is plotted.
        shape (str): The shape of mask, 'box' or 'circle'.
        kwargs (optional):
            When 'box' is specified as shape,
                xc: Center of x.
                yc: Center of y.
                width: Width.
                height: Height.
                xmin: Minimum of x.
                xmax: Maximum of x.
                ymin: Minimum of y.
                ymax: Maximum of y.
            When 'circle' is specified as shape,
                xc: Center of x.
                yc: Center of y.
                radius: Radius.
            xlim (list): Range of x.
            ylim (list): Range of y.

    Notes:
        All kwargs should be specified as pixel coordinates.
    """
    if dataarray.type == 'dca':
        pass
    elif dataarray.type == 'dcc':
        xr.DataArray.dcc.plotspectrum(dataarray, shape, **kwargs)
    elif dataarray.type == 'dcs':
        pass
    else:
        pass
