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
import xarray as xr
from astropy.io import fits


# functions
def plotcoords(dataarray, ax, coords, scantypes=None, **kwargs):
    """Plot coordinates.

    Args:
        dataarray (xarray.DataArray): Dataarray which the coodinate information is included.
        ax (matplotlib.axes): Axis you want to plot on.
        coords (list): Name of x axis and y axis.
        scantypes (list): Scantypes. If None, all scantypes are used.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotcoords(dataarray, ax, coords, scantypes, **kwargs)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plotweather(dataarray, axs, **kwargs):
    """Plot weather information.

    Args:
        dataarray (xarray.DataArray): Dataarray which the weather information is included.
        axs (list(matplotlib.axes)): Axes you want to plot on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotweather(dataarray, axs, **kwargs)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plotspectrum(dataarray, ax, xtick, ytick, aperture, **kwargs):
    """Plot a spectrum.

    Args:
        dataarray (xarray.DataArray): Dataarray which the spectrum information is included.
        ax (matplotlib.axes): Axis you want to plot on.
        xtick (str): Type of x axis.
            'freq': Frequency [GHz].
            'id': Kid id.
        ytick (str): Type of y axis.
            'peak': Peak.
            'sum': Summation.
            'mean': Under construction.
        aperture (str): The shape of aperture.
            'box': Box.
            'circle': Circle.
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


def plottimestream(array, ax, xtick='time', **kwargs):
    """Plot timestream data.

    Args:
        array (xarray.DataArray): Array which the timestream data are included.
        ax (matplotlib.axes): Axis you want to plot on.
        xtick (str): Type of x axis.
            'time': Time.
            'index': Time index.
        kwargs (optional): Plot options passed to ax.plot().
    """
    logger = getLogger('decode.plot.plottimestream')

    kidtpdict = {0: 'wideband', 1: 'filter', 2: 'blind'}
    if xtick == 'time':
        ax.plot(array.time, array, **kwargs)
        ax.set_xlabel('time', fontsize=20, color='grey')
    elif xtick == 'index':
        ax.plot(array.time, array, **kwargs)
        ax.set_xlabel('time index', fontsize=20, color='grey')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.legend()
    ax.set_ylabel(str(array.datatype.values), fontsize=20, color='grey')

    kidid = int(array.kidid)
    try:
        kidtp = kidtpdict[int(array.kidtp)]
    except KeyError:
        kidtp = 'filter'
    ax.set_title('ch #{} ({})'.format(kidid, kidtp), fontsize=20, color='grey', y=1.15)

    logger.info('timestream data (ch={}) has been plotted.'.format(kidid))
