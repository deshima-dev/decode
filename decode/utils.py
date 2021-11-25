# coding: utf-8

# public items
__all__ = [
    "allan_variance",
    "copy_function",
    "deprecation_warning",
    "one_thread_per_process",
    "psd",
    "slicewhere",
]


# standard library
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from types import CodeType, FunctionType


# dependent packages
import numpy as np
from scipy import ndimage
from scipy.fftpack import fftfreq, fft
from scipy.signal import hanning


# function
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


def copy_function(func, name=None):
    """Copy a function object with different name.

    Args:
        func (function): Function to be copied.
        name (string, optional): Name of the new function.
            If not spacified, the same name of `func` will be used.

    Returns:
        newfunc (function): New function with different name.

    """
    code = func.__code__
    newname = name or func.__name__
    newcode = CodeType(
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_stacksize,
        code.co_flags,
        code.co_code,
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_filename,
        newname,
        code.co_firstlineno,
        code.co_lnotab,
        code.co_freevars,
        code.co_cellvars,
    )
    newfunc = FunctionType(
        newcode,
        func.__globals__,
        newname,
        func.__defaults__,
        func.__closure__,
    )
    newfunc.__dict__.update(func.__dict__)
    return newfunc


def deprecation_warning(message, cls=PendingDeprecationWarning):
    import warnings

    warnings.filterwarnings("always", category=PendingDeprecationWarning)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, cls, stacklevel=2)
            func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def one_thread_per_process():
    """Return a context manager where only one thread is allocated to a process.

    This function is intended to be used as a with statement like::

        >>> with process_per_thread():
        ...     do_something() # one thread per process

    Notes:
        This function only works when MKL (Intel Math Kernel Library)
        is installed and used in, for example, NumPy and SciPy.
        Otherwise this function does nothing.

    """
    try:
        import mkl

        is_mkl = True
    except ImportError:
        is_mkl = False

    if is_mkl:
        n_threads = mkl.get_max_threads()
        mkl.set_num_threads(1)
        try:
            # block nested in the with statement
            yield
        finally:
            # revert to the original value
            mkl.set_num_threads(n_threads)
    else:
        yield


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


def slicewhere(condition):
    """Return slices of regions that fulfill condition.

    Example:
        >>> cond = [False, True, True, False, False, True, False]
        >>> fm.utils.slicewhere(cond)
        [slice(1L, 3L, None), slice(5L, 6L, None)]

    Args:
        condition (numpy.ndarray): Array of booleans.

    Returns:
        slices (list of slice): List of slice objects.
    """
    return [region[0] for region in ndimage.find_objects(ndimage.label(condition)[0])]
