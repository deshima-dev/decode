# coding: utf-8

# public items
__all__ = [
    'plotcoords',
    'plotweather',
    'plotspectrum',
    'plottimestream',
    'plotpsd',
    'plotallanvar'
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
from scipy.signal import hanning
from ..utils.ndarray.functions import psd, allan_variance


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
        raise KeyError(dataarray.type)


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
        raise KeyError(dataarray.type)


def plotspectrum(dataarray, ax, xtick, ytick, aperture, **kwargs):
    """Plot a spectrum.

    Args:
        dataarray (xarray.DataArray): Dataarray which the spectrum information is included.
        ax (matplotlib.axes): Axis you want to plot on.
        xtick (str): Type of x axis.
            'freq': Frequency [GHz].
            'id': Kid id.
        ytick (str): Type of y axis.
            'max': Maximum.
            'sum': Summation.
            'mean': Mean.
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
            Remaining kwargs are passed to ax.step().

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
        raise KeyError(dataarray.type)


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
    elif xtick == 'index':
        ax.plot(np.ogrid[:len(array.time)], array, **kwargs)
    ax.set_xlabel('{}'.format(xtick), fontsize=20, color='grey')
    # for label in ax.get_xticklabels():
    #     label.set_rotation(45)
    ax.set_ylabel(str(array.datatype.values), fontsize=20, color='grey')
    ax.legend()

    kidid = int(array.kidid)
    try:
        kidtp = kidtpdict[int(array.kidtp)]
    except KeyError:
        kidtp = 'filter'
    ax.set_title('ch #{} ({})'.format(kidid, kidtp), fontsize=20, color='grey')

    logger.info('timestream data (ch={}) has been plotted.'.format(kidid))


def plotpsd(data, dt, ndivide=1, window=hanning, overlap_half=False, ax=None, **kwargs):
    """Plot PSD (Power Spectral Density).

    Args:
        data (np.ndarray): Input data.
        dt (float): Time between each data.
        ndivide (int): Do averaging (split data into ndivide, get psd of each, and average them).
        overlap_half (bool): Split data to half-overlapped regions.
        ax (matplotlib.axes): Axis the figure is plotted on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if ax is None:
        ax = plt.gca()
    vk, psddata = psd(data, dt, ndivide, window, overlap_half)
    ax.loglog(vk, psddata, **kwargs)
    ax.set_xlabel('Frequency [Hz]', fontsize=20, color='grey')
    ax.set_ylabel('PSD', fontsize=20, color='grey')
    ax.legend()


def plotallanvar(data, dt, tmax=10, ax=None, **kwargs):
    """Plot Allan variance.

    Args:
        data (np.ndarray): Input data.
        dt (float): Time between each data.
        tmax (float): Maximum time.
        ax (matplotlib.axes): Axis the figure is plotted on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if ax is None:
        ax = plt.gca()
    tk, allanvar = allan_variance(data, dt, tmax)
    ax.loglog(tk, allanvar, **kwargs)
    ax.set_xlabel('Time [s]', fontsize=20, color='grey')
    ax.set_ylabel('Allan Variance', fontsize=20, color='grey')
    ax.legend()
