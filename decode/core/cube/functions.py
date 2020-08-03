# coding: utf-8


# public items
__all__ = ["cube", "fromcube", "tocube", "makecontinuum"]


# standard library
from logging import getLogger


# dependent packages
import numpy as np
import decode as dc
import xarray as xr
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import map_coordinates


# module logger
logger = getLogger(__name__)


# functions
def cube(
    data,
    xcoords=None,
    ycoords=None,
    chcoords=None,
    scalarcoords=None,
    datacoords=None,
    attrs=None,
    name=None,
):
    """Create a cube as an instance of xarray.DataArray with Decode accessor.

    Args:
        data (numpy.ndarray): 3D (x x y x channel) array.
        xcoords (dict, optional): Dictionary of arrays that label x axis.
        ycoords (dict, optional): Dictionary of arrays that label y axis.
        chcoords (dict, optional): Dictionary of arrays that label channel axis.
        scalarcoords (dict, optional): Dictionary of values
            that don't label any axes (point-like).
        datacoords (dict, optional): Dictionary of arrays
            that label x, y, and channel axes.
        attrs (dict, optional): Dictionary of attributes to add to the instance.
        name (str, optional): String that names the instance.

    Returns:
        decode cube (decode.cube): Decode cube.
    """
    # initialize coords with default values
    cube = xr.DataArray(data, dims=("x", "y", "ch"), attrs=attrs, name=name)
    cube.dcc._initcoords()

    # update coords with input values (if any)
    if xcoords is not None:
        cube.coords.update({key: ("x", xcoords[key]) for key in xcoords})

    if ycoords is not None:
        cube.coords.update({key: ("y", ycoords[key]) for key in ycoords})

    if chcoords is not None:
        cube.coords.update({key: ("ch", chcoords[key]) for key in chcoords})

    if datacoords is not None:
        cube.coords.update(
            {key: (("x", "y", "ch"), datacoords[key]) for key in datacoords}
        )

    if scalarcoords is not None:
        cube.coords.update(scalarcoords)

    return cube


def fromcube(cube, template):
    """Covert a decode cube to a decode array.

    Args:
        cube (decode.cube): Decode cube to be cast.
        template (decode.array): Decode array whose shape the cube is cast on.

    Returns:
        decode array (decode.array): Decode array.

    Notes:
        This functions is under development.
    """
    array = dc.zeros_like(template)

    y, x = array.y.values, array.x.values
    gy, gx = cube.y.values, cube.x.values
    iy = interp1d(gy, np.arange(len(gy)))(y)
    ix = interp1d(gx, np.arange(len(gx)))(x)

    for ch in range(len(cube.ch)):
        array[:, ch] = map_coordinates(cube.values[:, :, ch], (ix, iy))

    return array


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
            unit (str): Unit of x/y.
                'deg' or 'degree': Degree (Default).
                'arcmin': Arcminute.
                'arcsec': Arcsecond.

    Returns:
        decode cube (decode.cube): Decode cube.

    Notes:
        Available combinations of kwargs are
            (1) xarr/yarr and xc/yc
            (2) gx/gy and xmin/xmax/ymin/ymax and xc/yc
            (3) nx/ny and xmin/xmax/ymin/ymax
    """
    # pick up kwargs
    unit = kwargs.pop("unit", "deg")
    unit2deg = getattr(u, unit).to("deg")

    xc = kwargs.pop("xc", float(array.xref)) * unit2deg
    yc = kwargs.pop("yc", float(array.yref)) * unit2deg
    xarr = kwargs.pop("xarr", None)
    yarr = kwargs.pop("yarr", None)
    xmin = kwargs.pop("xmin", None)
    xmax = kwargs.pop("xmax", None)
    ymin = kwargs.pop("ymin", None)
    ymax = kwargs.pop("ymax", None)
    gx = kwargs.pop("gx", None)
    gy = kwargs.pop("gy", None)
    nx = kwargs.pop("nx", None)
    ny = kwargs.pop("ny", None)
    if None not in [xarr, yarr]:
        x_grid = xr.DataArray(xarr * unit2deg, dims="grid")
        y_grid = xr.DataArray(yarr * unit2deg, dims="grid")
    else:
        if None not in [xmin, xmax, ymin, ymax]:
            xmin = xmin * unit2deg
            xmax = xmax * unit2deg
            ymin = ymin * unit2deg
            ymax = ymax * unit2deg
        else:
            xmin = array.x.min()
            xmax = array.x.max()
            ymin = array.y.min()
            ymax = array.y.max()
        logger.info("xmin xmax ymin ymax")
        logger.info("{} {} {} {}".format(xmin, xmax, ymin, ymax))

        if None not in [gx, gy]:
            gx = gx * unit2deg
            gy = gy * unit2deg
            logger.info("xc yc gx gy")
            logger.info("{} {} {} {}".format(xc, yc, gx, gy))

            gxmin = np.floor((xmin - xc) / gx)
            gxmax = np.ceil((xmax - xc) / gx)
            gymin = np.floor((ymin - yc) / gy)
            gymax = np.ceil((ymax - yc) / gy)
            xmin = gxmin * gx + xc
            xmax = gxmax * gx + xc
            ymin = gymin * gy + yc
            ymax = gymax * gy + yc

            x_grid = xr.DataArray(np.arange(xmin, xmax + gx, gx), dims="grid")
            y_grid = xr.DataArray(np.arange(ymin, ymax + gy, gy), dims="grid")
        elif None not in [nx, ny]:
            logger.info("nx ny")
            logger.info("{} {}".format(nx, ny))
            # nx/ny does not support xc/yc
            xc = 0
            yc = 0

            x_grid = xr.DataArray(np.linspace(xmin, xmax, nx), dims="grid")
            y_grid = xr.DataArray(np.linspace(ymin, ymax, ny), dims="grid")
        else:
            raise KeyError("Arguments are wrong.")

    # reverse the direction of x when coordsys == 'RADEC'
    if array.coordsys == "RADEC":
        x_grid = x_grid[::-1]

    # compute gridding
    nx, ny, nch = len(x_grid), len(y_grid), len(array.ch)
    i = np.abs(array.x - x_grid).argmin("grid").compute()
    j = np.abs(array.y - y_grid).argmin("grid").compute()
    index = i + nx * j

    array.coords.update({"index": index})
    groupedarray = array.groupby("index")
    groupedones = dc.ones_like(array).groupby("index")

    gridarray = groupedarray.mean("t")
    stdarray = groupedarray.std("t")
    numarray = groupedones.sum("t")

    logger.info("Gridding started.")
    gridarray = gridarray.compute()
    noisearray = (stdarray / numarray ** 0.5).compute()
    logger.info("Gridding finished.")

    # create cube
    mask = gridarray.index.values

    temp = np.full([ny * nx, nch], np.nan)
    temp[mask] = gridarray.values
    data = temp.reshape((ny, nx, nch)).swapaxes(0, 1)

    temp = np.full([ny * nx, nch], np.nan)
    temp[mask] = noisearray.values
    noise = temp.reshape((ny, nx, nch)).swapaxes(0, 1)

    xcoords = {"x": x_grid.values}
    ycoords = {"y": y_grid.values}
    chcoords = {
        "masterid": array.masterid.values,
        "kidid": array.kidid.values,
        "kidfq": array.kidfq.values,
        "kidtp": array.kidtp.values,
    }
    scalarcoords = {
        "coordsys": array.coordsys.values,
        "datatype": array.datatype.values,
        "xref": array.xref.values,
        "yref": array.yref.values,
    }
    datacoords = {"noise": noise}

    return dc.cube(
        data,
        xcoords=xcoords,
        ycoords=ycoords,
        chcoords=chcoords,
        scalarcoords=scalarcoords,
        datacoords=datacoords,
    )


def makecontinuum(cube, **kwargs):
    """Make a continuum array.

    Args:
        cube (decode.cube): Decode cube which will be averaged over channels.
        kwargs (optional): Other arguments.
            inchs (list): Included channel kidids.
            exchs (list): Excluded channel kidids.

    Returns:
        decode cube (decode.cube): Decode cube (2d).
    """
    # pick up kwargs
    inchs = kwargs.pop("inchs", None)
    exchs = kwargs.pop("exchs", None)
    weight = kwargs.pop("weight", None)

    if (inchs is not None) or (exchs is not None):
        raise KeyError("Inchs and exchs are no longer supported. Use weight instead.")

    # if inchs is not None:
    #     logger.info('inchs')
    #     logger.info('{}'.format(inchs))
    #     subcube = cube[:, :, inchs]
    # else:
    #     mask = np.full(len(cube.ch), True)
    #     if exchs is not None:
    #         logger.info('exchs')
    #         logger.info('{}'.format(exchs))
    #         mask[exchs] = False
    #     subcube = cube[:, :, mask]

    if weight is None:
        weight = 1.0
    # else:
    # cont = (subcube * (1 / subcube.noise**2)).sum(dim='ch') \
    #        / (1 / subcube.noise**2).sum(dim='ch')
    # cont = cont.expand_dims(dim='ch', axis=2)
    cont = (cube * (1 / weight ** 2)).sum(dim="ch") / (1 / weight ** 2).sum(dim="ch")

    # define coordinates
    xcoords = {"x": cube.x.values}
    ycoords = {"y": cube.y.values}
    chcoords = {
        "masterid": np.array([0]),  # np.array([int(subcube.masterid.mean(dim='ch'))]),
        "kidid": np.array([0]),  # np.array([int(subcube.kidid.mean(dim='ch'))]),
        "kidfq": np.array([0]),  # np.array([float(subcube.kidfq.mean(dim='ch'))]),
        "kidtp": np.array([1]),
    }  # np.array([1])}
    scalarcoords = {
        "coordsys": cube.coordsys.values,
        "datatype": cube.datatype.values,
        "xref": cube.xref.values,
        "yref": cube.yref.values,
    }

    return dc.cube(
        cont.values,
        xcoords=xcoords,
        ycoords=ycoords,
        chcoords=chcoords,
        scalarcoords=scalarcoords,
    )
