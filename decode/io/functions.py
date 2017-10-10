# coding: utf-8

# public items
__all__ = [
    'loaddfits',
    'savefits',
    'loadnetcdf',
    'savenetcdf',
]

# dependent packages
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits


def loaddfits(fitsname, pixelids='all', scantype='all'):
    """DFITS to array"""
    hdulist = fits.open(fitsname)

    ### readout
    readout   = hdulist['READOUT'].data
    t_out     = np.array(readout['starttime']).astype(np.datetime64)
    pixelid   = readout['pixelid']
    arraydata = readout['arraydata']

    ### antenna
    antlog = hdulist['ANTENNA'].data
    t_ant  = np.array(antlog['starttime']).astype(np.datetime64)
    az     = antlog['az']
    el     = antlog['el']
    try:
        az_c = antlog['az-prog(center)']
        el_c = antlog['el-prog(center)']
    except KeyError:
        az_c = 0
        el_c = 0
    raz    = az - az_c
    rel    = el - el_c

    ### interpolation
    t_ant_last = np.where(t_ant >= t_out[-1])[0][0]
    t_ant_sub  = t_ant[:t_ant_last+1]
    raz_sub    = raz[:t_ant_last+1]
    rel_sub    = rel[:t_ant_last+1]
    dt_out     = (t_out - t_out[0]).astype(np.float64)
    dt_ant_sub = (t_ant_sub - t_out[0]).astype(np.float64)
    raz_sub_i  = np.interp(dt_out, dt_ant_sub, raz_sub)
    rel_sub_i  = np.interp(dt_out, dt_ant_sub, rel_sub)

    ### coordinates
    tcoords  = {'x': raz_sub_i, 'y': rel_sub_i, 'time': t_out}

    ### make array
    array = dc.array(arraydata, tcoords=tcoords)

    ### close hdu
    hdulist.close()

    return array


def savefits(dataarray, fitsname, clobber=False):
    if dataarray.type == 'dca':
        pass
    elif dataarray.type == 'dcc':
        xr.DataArray.dcc.savefits(dataarray, fitsname, clobber=clobber)
    elif dataarray.type == 'dcs':
        pass
    else:
        pass


def loadnetcdf(filename, copy=True):
    """Load a dataarray from a NetCDF file.

    Args:
        filename (str): A file name (*.nc).
        copy (bool): If True, dataarray is copied in memory. Default is True.

    Returns:
        dataarray (xarray.DataArray): A loaded dataarray.

    """
    if copy:
        dataarray = xr.open_dataarray(filename).copy()
    else:
        dataarray = xr.open_dataarray(filename)

    if dataarray.name is None:
        dataarray.name = filename.rstrip('.nc')

    for key, val in dataarray.coords.items():
        if val.dtype.kind == 'S':
            dataarray[key] = val.astype('U')
        elif val.dtype == np.int32:
            dataarray[key] = val.astype('i8')

    return dataarray


def savenetcdf(dataarray, filename=None):
    """Save a dataarray to a NetCDF file.

    Args:
        dataarray (xarray.DataArray): A dataarray to be saved.
        filename (str): A filename (used as <filename>.nc).
            If not spacified, random 8-character name will be used.

    """
    if filename is None:
        if dataarray.name is not None:
            filename = dataarray.name
        else:
            filename = uuid4().hex[:8]

    if not filename.endswith('.nc'):
        filename += '.nc'

    dataarray.to_netcdf(filename)
