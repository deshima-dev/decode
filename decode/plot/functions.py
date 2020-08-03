# coding: utf-8


# public items
__all__ = [
    "plotcoords",
    "plot_tcoords",
    "plottimestream",
    "plot_timestream",
    "plotspectrum",
    "plot_spectrum",
    "plot_chmap",
    "plotpsd",
    "plotallanvar",
]


# standard library
from logging import getLogger


# dependent packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning
from ..utils.ndarray.functions import psd, allan_variance
from ..utils.misc.functions import deprecation_warning


# module logger
logger = getLogger(__name__)


# functions
def plot_tcoords(array, coords, scantypes=None, ax=None, **kwargs):
    """Plot coordinates related to the time axis.

    Args:
        array (xarray.DataArray): Array which the coodinate information is included.
        coords (list): Name of x axis and y axis.
        scantypes (list): Scantypes. If None, all scantypes are used.
        ax (matplotlib.axes): Axis you want to plot on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if ax is None:
        ax = plt.gca()

    if scantypes is None:
        ax.plot(array[coords[0]], array[coords[1]], label="ALL", **kwargs)
    else:
        for scantype in scantypes:
            ax.plot(
                array[coords[0]][array.scantype == scantype],
                array[coords[1]][array.scantype == scantype],
                label=scantype,
                **kwargs
            )
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])
    ax.set_title("{} vs {}".format(coords[1], coords[0]))
    ax.legend()

    logger.info("{} vs {} has been plotted.".format(coords[1], coords[0]))


def plot_timestream(array, kidid, xtick="time", scantypes=None, ax=None, **kwargs):
    """Plot timestream data.

    Args:
        array (xarray.DataArray): Array which the timestream data are included.
        kidid (int): Kidid.
        xtick (str): Type of x axis.
            'time': Time.
            'index': Time index.
        scantypes (list): Scantypes. If None, all scantypes are used.
        ax (matplotlib.axes): Axis you want to plot on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if ax is None:
        ax = plt.gca()

    index = np.where(array.kidid == kidid)[0]
    if len(index) == 0:
        raise KeyError("Such a kidid does not exist.")
    index = int(index)

    if scantypes is None:
        if xtick == "time":
            ax.plot(array.time, array[:, index], label="ALL", **kwargs)
        elif xtick == "index":
            ax.plot(np.ogrid[: len(array.time)], array[:, index], label="ALL", **kwargs)
    else:
        for scantype in scantypes:
            if xtick == "time":
                ax.plot(
                    array.time[array.scantype == scantype],
                    array[:, index][array.scantype == scantype],
                    label=scantype,
                    **kwargs
                )
            elif xtick == "index":
                ax.plot(
                    np.ogrid[: len(array.time[array.scantype == scantype])],
                    array[:, index][array.scantype == scantype],
                    label=scantype,
                    **kwargs
                )
    ax.set_xlabel("{}".format(xtick))
    ax.set_ylabel(str(array.datatype.values))
    ax.legend()

    kidtpdict = {0: "wideband", 1: "filter", 2: "blind"}
    try:
        kidtp = kidtpdict[int(array.kidtp[index])]
    except KeyError:
        kidtp = "filter"
    ax.set_title("ch #{} ({})".format(kidid, kidtp))

    logger.info("timestream data (ch={}) has been plotted.".format(kidid))


def plot_spectrum(cube, xtick, ytick, aperture, ax=None, **kwargs):
    """Plot a spectrum.

    Args:
        cube (xarray.DataArray): Cube which the spectrum information is included.
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
        ax (matplotlib.axes): Axis you want to plot on.
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
    if ax is None:
        ax = plt.gca()

    # pick up kwargs
    xc = kwargs.pop("xc", None)
    yc = kwargs.pop("yc", None)
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    xmin = kwargs.pop("xmin", None)
    xmax = kwargs.pop("xmax", None)
    ymin = kwargs.pop("ymin", None)
    ymax = kwargs.pop("ymax", None)
    radius = kwargs.pop("radius", None)
    exchs = kwargs.pop("exchs", None)

    # labels
    xlabeldict = {"freq": "frequency [GHz]", "id": "kidid"}

    cube = cube.copy()
    datatype = cube.datatype
    if aperture == "box":
        if None not in [xc, yc, width, height]:
            xmin, xmax = int(xc - width / 2), int(xc + width / 2)
            ymin, ymax = int(yc - width / 2), int(yc + width / 2)
        elif None not in [xmin, xmax, ymin, ymax]:
            pass
        else:
            raise KeyError("Invalid arguments.")
        value = getattr(cube[xmin:xmax, ymin:ymax, :], ytick)(dim=("x", "y"))
    elif aperture == "circle":
        if None not in [xc, yc, radius]:
            pass
        else:
            raise KeyError("Invalid arguments.")
        x, y = np.ogrid[0 : len(cube.x), 0 : len(cube.y)]
        mask = (x - xc) ** 2 + (y - yc) ** 2 < radius ** 2
        mask = np.broadcast_to(mask[:, :, np.newaxis], cube.shape)
        masked = np.ma.array(cube.values, mask=~mask)
        value = getattr(np, "nan" + ytick)(masked, axis=(0, 1))
    else:
        raise KeyError(aperture)

    if xtick == "freq":
        kidfq = cube.kidfq.values
        freqrange = ~np.isnan(kidfq)
        if exchs is not None:
            freqrange[exchs] = False
        x = kidfq[freqrange]
        y = value[freqrange]
        ax.step(x[np.argsort(x)], y[np.argsort(x)], where="mid", **kwargs)
    elif xtick == "id":
        ax.step(cube.kidid.values, value, where="mid", **kwargs)
    else:
        raise KeyError(xtick)
    ax.set_xlabel("{}".format(xlabeldict[xtick]))
    ax.set_ylabel("{} ({})".format(datatype.values, ytick))
    ax.set_title("spectrum")


def plot_chmap(cube, kidid, ax=None, **kwargs):
    """Plot an intensity map.

    Args:
        cube (xarray.DataArray): Cube which the spectrum information is included.
        kidid (int): Kidid.
        ax (matplotlib.axes): Axis the figure is plotted on.
        kwargs (optional): Plot options passed to ax.imshow().
    """
    if ax is None:
        ax = plt.gca()

    index = np.where(cube.kidid == kidid)[0]
    if len(index) == 0:
        raise KeyError("Such a kidid does not exist.")
    index = int(index)

    im = ax.pcolormesh(cube.x, cube.y, cube[:, :, index].T, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("intensity map ch #{}".format(kidid))
    return im


def plotpsd(data, dt, ndivide=1, window=hanning, overlap_half=False, ax=None, **kwargs):
    """Plot PSD (Power Spectral Density).

    Args:
        data (np.ndarray): Input data.
        dt (float): Time between each data.
        ndivide (int): Do averaging (split data into ndivide,
            get psd of each, and average them).
        overlap_half (bool): Split data to half-overlapped regions.
        ax (matplotlib.axes): Axis the figure is plotted on.
        kwargs (optional): Plot options passed to ax.plot().
    """
    if ax is None:
        ax = plt.gca()
    vk, psddata = psd(data, dt, ndivide, window, overlap_half)
    ax.loglog(vk, psddata, **kwargs)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
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
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Allan Variance")
    ax.legend()


# alias
@deprecation_warning(
    "Use plot_tcoords() instead. plotcoords() will be removed in the future."
    " The order of the arguments has been changed in plot_tcoords()."
    " For a while, De:code properly passes the arguments"
    " in plotcoords() to plot_tcoords()."
)
def plotcoords(array, ax, coords, scantypes=None, **kwargs):
    plot_tcoords(array, coords, scantypes=scantypes, ax=ax, **kwargs)


@deprecation_warning(
    "Use plot_timestream() instead. plottimestream() has been removed."
    "The arguments has been changed in plot_timestream().",
    DeprecationWarning,
)
def plottimestream(array, ax=None, xtick="time", **kwargs):
    pass


@deprecation_warning(
    "Use plot_spectrum() instead. plotspectrum() will be removed in the future."
    " The order of the arguments has been changed in plot_spectrum()."
    " For a while, De:code properly passes the arguments"
    " in plotspectrum() to plot_spectrum()."
)
def plotspectrum(cube, ax, xtick, ytick, aperture, **kwargs):
    plot_spectrum(cube, xtick, ytick, aperture, ax=ax, **kwargs)
