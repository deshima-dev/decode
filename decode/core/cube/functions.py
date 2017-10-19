# coding: utf-8

# public items
__all__ = [
    'cube',
    'fromcube',
    'tocube',
]

# dependent packages
import decode as dc
import xarray as xr


# functions
def cube(data, xcoords=None, ycoords=None, chcoords=None, scalarcoords=None, attrs=None, name=None):
    """Create a cube as an instance of xarray.DataArray with Decode accessor.

    Args:
        data (numpy.ndarray): A 3D (x x y x channel) array.
        xcoords (dict, optional): A dictionary of arrays that label x axis.
        ycoords (dict, optional): A dictionary of arrays that label y axis.
        chcoords (dict, optional): A dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): A dictionary of values that don't label any axes (point-like).
        attrs (dict, optional): A dictionary of attributes to add to the instance.
        name (str, optional): A string that names the instance.

    Returns:
        cube (xarray.DataArray): a cube.

    """
    # initialize coords with default values
    cube = xr.DataArray(data, dims=('x', 'y', 'ch'), attrs=attrs, name=name)
    cube.dcc._initcoords()

    # update coords with input values (if any)
    if xcoords is not None:
        cube.coords.update({key: ('x', xcoords[key]) for key in xcoords})

    if ycoords is not None:
        cube.coords.update({key: ('y', ycoords[key]) for key in ycoords})

    if chcoords is not None:
        cube.coords.update({key: ('ch', chcoords[key]) for key in chcoords})

    if scalarcoords is not None:
        cube.coords.update(scalarcoords)

    return cube


def fromcube(cube):
    """cube to array"""
    return xr.DataArray.dcc.fromcube(cube)


def tocube(array, **kwargs):
    """array to cube"""
    return xr.DataArray.dcc.tocube(array, **kwargs)
