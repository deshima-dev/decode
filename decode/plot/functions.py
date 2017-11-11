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
def plotcoords(dataarray, coords, scantypes=None, save=True, **kwargs):
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotcoords(dataarray, coords, scantypes, save, **kwargs)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


def plotweather(dataarray, save=True, **kwargs):
    if dataarray.type == 'dca':
        xr.DataArray.dca.plotweather(dataarray, save, **kwargs)
    elif dataarray.type == 'dcc':
        pass
    elif dataarray.type == 'dcs':
        pass
    else:
        raise KerError(dataarray.type)


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
        raise KerError(dataarray.type)


def plottimestream(scanarray, fittedarray=None, filteredarray=None, chs=None, peakfind=True,
                   save=True, **kwargs):
    logger = getLogger('decode.plot.plottimestream')
    if save:
        if not os.path.exists('timestream'):
            os.mkdir('timestream')
    kidtps = np.array(['wideband'] * 7 + ['filter'] * 50 + ['blind'] * 6)
    peaks  = []
    if chs is None:
        chs = np.ogrid[0:63]

    for ch in chs:
        fig, ax = plt.subplots(1, 2, **kwargs)
        ### raw dataとfitted dataのplot
        ax[0].plot(scanarray.time, scanarray[:, ch], label='scan')
        if fittedarray is not None:
            ax[0].plot(fittedarray.time, fittedarray[:, ch], label='fitted')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel(str(scanarray.datatype.values))
        ax[0].set_title('ch #{} ({})'.format(ch, kidtps[ch]), color='grey')
        ax[0].legend()
        ### filtered dataのplot
        if filteredarray is not None:
            ax[1].plot(filteredarray.time, filteredarray[:, ch], label='filtered')
            if peakfind:
                if kidtps[ch] != 'blind' and kidtps[ch] != 'bad':
                    ### time stream上でのpeak同定
                    peak = np.nanmax(filteredarray[:, ch])
                    peaks.append(peak)
                else:
                    peaks.append(np.nan)
            ax[1].set_xlabel('time index')
            ax[1].set_ylabel(str(scanarray.datatype.values))
            ax[1].set_title('ch #{} ({})'.format(ch, kidtps[ch]), color='grey')
            ax[1].legend()
        fig.tight_layout()
        if save:
            fig.savefig('timestream/ch{}.png'.format(ch))
            logger.info('timestream/ch{}.png has been created.'.format(ch))
    logger.info('chs peaks')
    logger.info('{} {}'.format(chs, peaks))
