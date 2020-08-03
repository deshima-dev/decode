# coding: utf-8

# public items
__all__ = [
    "OrderedLoader",
    "deprecation_warning",
    "copy_function",
    "one_thread_per_process",
    "slicewhere",
]

# standard library
from collections import OrderedDict
from contextlib import contextmanager
from types import CodeType, FunctionType
from functools import wraps

# dependent packages
import yaml
from scipy import ndimage


# classes
class OrderedLoader(yaml.Loader):
    """YAML loader for keeping order of items.

    This class is intended to be used like::

        >>> with open(filename) as f:
        >>> ... yaml.load(f, Loader=OrderedLoader)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

        def constructor(loader, node):
            return OrderedDict(loader.construct_pairs(node))

        self.add_constructor(tag, constructor)


# function
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
        newcode, func.__globals__, newname, func.__defaults__, func.__closure__,
    )
    newfunc.__dict__.update(func.__dict__)
    return newfunc


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
