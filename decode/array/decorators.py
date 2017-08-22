# coding: utf-8

# public items
__all__ = [
    'chunk',
    'numpyfunc',
]

# standard library
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from inspect import Parameter, signature, stack
from multiprocessing import cpu_count

# dependent packages
import decode as dc
import numpy as np
import xarray as xr

# module constants
POS_OR_KWD = Parameter.POSITIONAL_OR_KEYWORD
try:
    MAX_WORKERS = cpu_count() - 1
except:
    MAX_WORKERS = 1


# decorators
def numpyfunc(func):
    """Make a function compatible with xarray.DataArray.

    This function is intended to be used as a decorator like::

        >>> @dc.arrayfunc
        >>> def func(array):
        ...     # do something
        ...     return newarray
        >>>
        >>> result = func(array)

    Args:
        func (function): A function to be wrapped. The first argument
            of the function must be an array to be processed.

    Returns:
        wrapper (function): A wrapped function.

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

            return dc.full_like(args[0], func(*newargs, **kwargs))
        else:
            return func(*args, **kwargs)

    return wrapper


def chunk(*argnames):
    """Make a function compatible with multicore chunk processing.

    This function is intended to be used as a decorator like::

        >>> @dc.chunk('array')
        >>> def func(array):
        ...     # do something
        ...     return newarray
        >>>
        >>> result = func(array, timechunk=10)

    """
    def _chunk(func):
        orgname = '_original_' + func.__name__
        orgfunc = dc.utils.copy_function(func, orgname)
        depth = [s.function for s in stack()].index('<module>')
        sys._getframe(depth).f_globals[orgname] = orgfunc

        @wraps(func)
        def wrapper(*args, **kwargs):
            # parse args and kwargs
            params = signature(func).parameters
            for i, (key, val) in enumerate(params.items()):
                if not val.kind == POS_OR_KWD:
                    break

                try:
                    kwargs.update({key: args[i]})
                except IndexError:
                    kwargs.setdefault(key, val.default)

            # n_chunks and n_processes
            if 'numchunk' in kwargs:
                n_chunks = kwargs.pop('numchunk', 1)
            elif 'timechunk' in kwargs:
                length = len(kwargs[argnames[0]])
                tchunk = kwargs.pop('timechunk', length)
                n_chunks = round(length / tchunk)
            else:
                n_chunks = 1

            n_processes = kwargs.pop('n_processes', MAX_WORKERS)

            # make chunked args
            chunks = {}
            for name in argnames:
                arg = kwargs.pop(name)
                try:
                    nargs = np.array_split(arg, n_chunks)
                except TypeError:
                    nargs = np.tile(arg, n_chunks)

                chunks.update({name: nargs})

            # run the function
            with dc.utils.one_thread_per_process(), \
                    ProcessPoolExecutor(n_processes) as e:
                futures = []
                for i in range(n_chunks):
                    chunk = {key: val[i] for key, val in chunks.items()}
                    futures.append(e.submit(orgfunc, **{**chunk, **kwargs}))

                results = [future.result() for future in futures]

            # make an output
            try:
                return xr.concat(results, 't')
            except TypeError:
                return np.concatenate(results, 0)

        return wrapper

    return _chunk
