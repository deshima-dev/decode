# coding: utf-8

# public items
__all__ = [
    'cube',
    'fromcube',
    'tocube',
    'makecontinuum',
]

# dependent packages
import decode as dc
import xarray as xr


# functions
def cube(data, xcoords=None, ycoords=None, chcoords=None, scalarcoords=None, datacoords=None, attrs=None, name=None):
    """Create a cube as an instance of xarray.DataArray with Decode accessor.

    Args:
        data (numpy.ndarray): 3D (x x y x channel) array.
        xcoords (dict, optional): Dictionary of arrays that label x axis.
        ycoords (dict, optional): Dictionary of arrays that label y axis.
        chcoords (dict, optional): Dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): Dictionary of values that don't label any axes (point-like).
        attrs (dict, optional): Dictionary of attributes to add to the instance.
        name (str, optional): String that names the instance.

    Returns:
        decode cube (decode.cube): Decode cube.

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

    if datacoords is not None:
        cube.coords.update({key: (('x', 'y', 'ch'), datacoords[key]) for key in datacoords})

    if scalarcoords is not None:
        cube.coords.update(scalarcoords)

    return cube


def fromcube(cube):
    """Covert a decode cube to a decode array.

    Args:
        cube (decode.cube): Decode cube which will be converted.

    Returns:
        decode array (decode.array): Decode array.

    Notes:
        This functions is under development.
    """
    return xr.DataArray.dcc.fromcube(cube)


def tocube(array, **kwargs):
    """Convert a decode array to decode cube.

    Args:
        array (decode.array): Decode array which will be converted.
        kwargs (optional): Other arguments.
            xarr (list or numpy.ndarray): Grid array of x direction.
            yarr (list or numpy.ndarray): Grid array of y direction.
            gx (float): The size of grid of x.
            gy (float): The size of grid of y.
            nx (int): The number of grid of x direction.
            ny (int): The number of grid of y direction.
            xmin (float): Minimum value of x.
            xmax (float): Maximum value of x.
            ymin (float): Minimum value of y.
            ymax (float): Maximum value of y.
            xc (float): Center of x.
            yc (float): Center of y.

    Returns:
        decode cube (decode.cube): Decode cube.

    Notes:
        Available combination of kwargs are
            (1) xarr/yarr and xc/yc
            (2) gx/gy and xmin/xmax/ymin/ymax and xc/yc
            (3) nx/ny and xmin/xmax/ymin/ymax

    """
    return xr.DataArray.dcc.tocube(array, **kwargs)


def makecontinuum(cube, kidtp, **kwargs):
    """Make a continuum array.

    Args:
        cube (decode.cube): Decode cube which will be averaged over channels.
        kidtp (int): Kid types which will be used
            0: wideband
            1: filter
            2: blind
        kwargs (optional): Other arguments.
            exchs (list): Excluded channel kidids

    Returns:
        decode cube (decode.cube): Decode cube (2d).
    """
    return xr.DataArray.dcc.makecontinuum(cube, kidtp, **kwargs)
