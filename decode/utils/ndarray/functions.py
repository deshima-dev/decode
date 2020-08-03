# coding: utf-8

# public items
__all__ = ["psd", "allan_variance"]


# standard library
from logging import getLogger


# dependent packages
import numpy as np
from scipy.fftpack import fftfreq, fft
from scipy.signal import hanning


# functions
def psd(data, dt, ndivide=1, window=hanning, overlap_half=False):
    """Calculate power spectrum density of data.

    Args:
        data (np.ndarray): Input data.
        dt (float): Time between each data.
        ndivide (int): Do averaging (split data into ndivide,
            get psd of each, and average them).
        ax (matplotlib.axes): Axis you want to plot on.
        doplot (bool): Plot how averaging works.
        overlap_half (bool): Split data to half-overlapped regions.

    Returns:
        vk (np.ndarray): Frequency.
        psd (np.ndarray): PSD
    """
    logger = getLogger("decode.utils.ndarray.psd")

    if overlap_half:
        step = int(len(data) / (ndivide + 1))
        size = step * 2
    else:
        step = int(len(data) / ndivide)
        size = step

    if bin(len(data)).count("1") != 1:
        logger.warning(
            "warning: length of data is not power of 2: {}".format(len(data))
        )
    size = int(len(data) / ndivide)
    if bin(size).count("1") != 1.0:
        if overlap_half:
            logger.warning(
                "warning: ((length of data) / (ndivide+1)) * 2"
                " is not power of 2: {}".format(size)
            )
        else:
            logger.warning(
                "warning: (length of data) / ndivide is not power of 2: {}".format(size)
            )
    psd = np.zeros(size)
    vk_ = fftfreq(size, dt)
    vk = vk_[np.where(vk_ >= 0)]

    for i in range(ndivide):
        d = data[i * step : i * step + size]
        if window is None:
            w = np.ones(size)
            corr = 1.0
        else:
            w = window(size)
            corr = np.mean(w ** 2)
        psd = psd + 2 * (np.abs(fft(d * w))) ** 2 / size * dt / corr

    return vk, psd[: len(vk)] / ndivide


def allan_variance(data, dt, tmax=10):
    """Calculate Allan variance.

    Args:
        data (np.ndarray): Input data.
        dt (float): Time between each data.
        tmax (float): Maximum time.

    Returns:
        vk (np.ndarray): Frequency.
        allanvar (np.ndarray): Allan variance.
    """
    allanvar = []
    nmax = len(data) if len(data) < tmax / dt else int(tmax / dt)
    for i in range(1, nmax + 1):
        databis = data[len(data) % i :]
        y = databis.reshape(len(data) // i, i).mean(axis=1)
        allanvar.append(((y[1:] - y[:-1]) ** 2).mean() / 2)
    return dt * np.arange(1, nmax + 1), np.array(allanvar)
