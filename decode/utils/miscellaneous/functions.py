# coding: utf-8

# public items
__all__ = [
    'copy_function',
]

# standard library
from types import CodeType, FunctionType


# function
def copy_function(func, name=None):
    """Copy a function object with different name.

    Args:
        func (function): A function to be copied.
        name (string, optional): A name of the new function.
            If not spacified, the same name of `func` will be used.

    Returns:
        newfunc (function): A new function with different name.

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
