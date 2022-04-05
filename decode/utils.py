__all__ = [
    "allan_variance",
    "chunk",
    "deprecation_warning",
    "one_thread_per_process",
    "psd",
    "slicewhere",
    "xarrayfunc",
]


# standard library
from concurrent.futures import ProcessPoolExecutor as Pool
from contextlib import contextmanager
from functools import wraps
from inspect import Parameter, signature, stack
from logging import getLogger
from multiprocessing import cpu_count
from sys import _getframe as getframe


# dependencies
import numpy as np
import xarray as xr
from morecopy import copy
from scipy import ndimage
from scipy.fftpack import fftfreq, fft
from scipy.signal import hanning


# constants
DEFAULT_N_CHUNKS = 1
try:
    MAX_WORKERS = cpu_count() - 1
except NotImplementedError:
    MAX_WORKERS = 1


# runtime functions
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


def chunk(*argnames, concatfunc=None):
    """Make a function compatible with multicore chunk processing.

    This function is intended to be used as a decorator like::

        >>> @dc.chunk('array')
        >>> def func(array):
        ...     # do something
        ...     return newarray
        >>>
        >>> result = func(array, timechunk=10)

    or you can set a global chunk parameter outside the function::

        >>> timechunk = 10
        >>> result = func(array)
    """

    def _chunk(func):
        depth = [s.function for s in stack()].index("<module>")
        f_globals = getframe(depth).f_globals

        # original (unwrapped) function
        orgfunc = copy(func)
        orgfunc.__name__ += "_org"
        f_globals[orgfunc.__name__] = orgfunc

        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = [s.function for s in stack()].index("<module>")
            f_globals = getframe(depth).f_globals

            # parse args and kwargs
            params = signature(func).parameters
            for i, (key, val) in enumerate(params.items()):
                if not val.kind == Parameter.POSITIONAL_OR_KEYWORD:
                    break

                try:
                    kwargs.update({key: args[i]})
                except IndexError:
                    kwargs.setdefault(key, val.default)

            # n_chunks and n_processes
            n_chunks = DEFAULT_N_CHUNKS
            n_processes = MAX_WORKERS

            if argnames:
                length = len(kwargs[argnames[0]])

                if "numchunk" in kwargs:
                    n_chunks = kwargs.pop("numchunk")
                elif "timechunk" in kwargs:
                    n_chunks = round(length / kwargs.pop("timechunk"))
                elif "numchunk" in f_globals:
                    n_chunks = f_globals["numchunk"]
                elif "timechunk" in f_globals:
                    n_chunks = round(length / f_globals["timechunk"])

                if "n_processes" in kwargs:
                    n_processes = kwargs.pop("n_processes")
                elif "n_processes" in f_globals:
                    n_processes = f_globals["n_processes"]

            # make chunked args
            chunks = {}
            for name in argnames:
                arg = kwargs.pop(name)
                try:
                    chunks.update({name: np.array_split(arg, n_chunks)})
                except TypeError:
                    chunks.update({name: np.tile(arg, n_chunks)})

            # run the function
            futures = []
            results = []
            with one_thread_per_process(), Pool(n_processes) as p:
                for i in range(n_chunks):
                    chunk = {key: val[i] for key, val in chunks.items()}
                    futures.append(p.submit(orgfunc, **{**chunk, **kwargs}))

                for future in futures:
                    results.append(future.result())

            # make an output
            if concatfunc is not None:
                return concatfunc(results)

            try:
                return xr.concat(results, "t")
            except TypeError:
                return np.concatenate(results, 0)

        return wrapper

    return _chunk


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
            corr = np.mean(w**2)
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


def xarrayfunc(func):
    """Make a function compatible with xarray.DataArray.

    This function is intended to be used as a decorator like::

        >>> @dc.xarrayfunc
        >>> def func(array):
        ...     # do something
        ...     return newarray
        >>>
        >>> result = func(array)

    Args:
        func (function): Function to be wrapped. The first argument
            of the function must be an array to be processed.

    Returns:
        wrapper (function): Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(arg, xr.DataArray) for arg in args):
            newargs = []
            for arg in args:
                if isinstance(arg, xr.DataArray):
                    newargs.append(arg.values)
                else:
                    newargs.append(arg)

            return xr.zeros_like(args[0]) + func(*newargs, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper
