# coding: utf-8

# public items
__all__ = [
    'plotcoords',
    'plotweather',
    'plotspectrum',
    'plottimestream'
]

# standard library
import os
from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.style.use('seaborn-darkgrid')
    plt.style.use('seaborn-pastel')
except:
    pass
import xarray as xr
from astropy.io import fits


# functions
def plotcoords(dataarray, ax, coords, scantypes=None):
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotcoords(dataarray, ax, coords, scantypes)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plotweather(dataarray, axs):
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotweather(dataarray, axs)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plotspectrum(dataarray, ax, xtick, ytick, aperture, **kwargs):
    """Plot a spectrum diagram.

    Args:
        dataarray (xarray.DataArray): Dataarray with which the spectrum is plotted.
        aperture (str): The shape of mask, 'box' or 'circle'.
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
        xr.DataArray.dcc.plotspectrum(dataarray, ax, xtick, ytick, aperture, **kwargs)
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plottimestream(array, ax, label=None):
    logger = getLogger('decode.plot.plottimestream')

    kidtpdict = {0: 'wideband', 1: 'filter', 2: 'blind'}
    if label is not None:
        ax.plot(array.time, array, label=label)
        ax.legend()
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.set_xlabel('time', fontsize=20, color='grey')
    ax.set_ylabel(str(array.datatype.values), fontsize=20, color='grey')
    kidid = int(array.kidid)
    try:
        kidtp = kidtpdict[int(array.kidtp)]
    except KeyError:
        kidtp = 'filter'
    ax.set_title('ch #{} ({})'.format(kidid, kidtp), color='grey')

    dataname = label if label is not None else 'data'
    logger.info('{} (ch={}) has been plottd.'.format(label, kidid))
