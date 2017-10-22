# coding: utf-8

# public items
__all__ = [
    'loaddfits',
    'savefits',
    'loadnetcdf',
    'savenetcdf',
    'loaddfits_2017oct',
]

# standard library
from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits


def loaddfits(fitsname, coordtype='azel', starttime=None, endtime=None, pixelids=None, scantype=None):
    """Load a decode array from a DFITS file.

    Please use `loaddfits_2017oct` until Az/El origins in antenna log are fixed.

    Args:
        fitsname (str): Name of DFITS file.
        coordtype (str): Coordinate type included into a decode array, azel or radec.
        starttime (int, str or numpy.datetime64): Start time of loaded data.
            It can be specified by the start index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            from the first record.
        endtime (int, str or numpy.datetime64): End time of loaded data.
            It can be specified by the end index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            until the last record.
        pixelids (int or list): Under development.
        scantype (str): Under development.

    Returns:
        decode array (decode.array): Loaded decode array.

    """
    logger = getLogger('decode.io.loaddfits')
    logger.warning('please use loaddfits_2017oct before Az/El origins in antenna log are fixed')

    hdulist = fits.open(fitsname)

    ### obsinfo
    obsinfo  = hdulist['OBSINFO'].data
    kidids   = obsinfo['kidids'][0]
    kidfreqs = obsinfo['kidfreqs'][0]

    ### antenna
    antlog = hdulist['ANTENNA'].data
    t_ant  = np.array(antlog['time']).astype(np.datetime64)
    if coordtype == 'azel':
        _x = antlog['az']
        _y = antlog['el']
        try:
            x_c = antlog['az_center']
            y_c = antlog['el_center']
        except KeyError:
            logger.warning('Az_center/el_center are not included in the antenna log! They are asuumed as 0.')
            x_c = 0
            y_c = 0
        x = _x - x_c
        y = _y - y_c
    elif coordtype == 'radec':
        x = antlog['ra']
        y = antlog['dec']

    ### readout
    readout = hdulist['READOUT'].data
    t_out   = np.array(readout['starttime']).astype(np.datetime64)

    if starttime is None:
        startindex = 0
    elif isinstance(starttime, int):
        startindex = starttime
    elif isinstance(starttime, str):
        startindex = np.where(t_out >= np.datetime64(starttime))[0][0]
    elif isinstance(starttime, np.datetime64):
        startindex = np.where(t_out >= starttime)[0][0]

    if endtime is None:
        endindex = t_out.shape[0]
    elif isinstance(endtime, int):
        endindex = endtime
    elif isinstance(endtime, str):
        endindex = np.where(t_out <= np.datetime64(endtime))[0][-1]
    elif isinstance(endtime, np.datetime64):
        endindex = np.where(t_out <= endtime)[0][-1]
    endindex_ant = np.where(t_out <= t_ant[-1])[0][-1]
    if endindex > endindex_ant:
        logger.warning('Endtime of readout is adjusted to the one of anttena log.')
        endindex = endindex_ant

    t_out      = t_out[startindex:endindex]
    pixelid    = readout['pixelid'][startindex:endindex]
    arraydata  = readout['arraydata'][startindex:endindex]

    ### interpolation
    # t_ant_last = np.where(t_ant[-1] > t_out)[0][-1]
    # t_ant_sub  = t_ant[:t_ant_last]
    # x_sub      = x[:t_ant_last]
    # y_sub      = y[:t_ant_last]
    dt_out     = (t_out - t_out[0]).astype(np.float64)
    dt_ant     = (t_ant - t_out[0]).astype(np.float64)
    # x_sub_i    = np.interp(dt_out, dt_ant_sub, x_sub)
    # y_sub_i    = np.interp(dt_out, dt_ant_sub, y_sub)
    x_i = np.interp(dt_out, dt_ant, x)
    y_i = np.interp(dt_out, dt_ant, y)

    ### coordinates
    tcoords  = {'x': x_i, 'y': y_i, 'time': t_out}
    chcoords = {'kidid': kidids, 'kidfq': kidfreqs}

    ### make array
    array = dc.array(arraydata, tcoords=tcoords, chcoords=chcoords)

    ### close hdu
    hdulist.close()

    return array


def savefits(dataarray, fitsname, **kwargs):
    """Save a dataarray to a 3D-cube FITS file.

    Args:
        dataarray (xarray.DataArray): Dataarray to be saved.
        fitsname (str): Name of output FITS file.
        kwargs (optional): Other arguments common with astropy.io.fits.writeto().

    """
    if dataarray.type == 'dca':
        pass
    elif dataarray.type == 'dcc':
        xr.DataArray.dcc.savefits(dataarray, fitsname, **kwargs)
    elif dataarray.type == 'dcs':
        pass
    else:
        pass


def loadnetcdf(filename, copy=True):
    """Load a dataarray from a NetCDF file.

    Args:
        filename (str): Filename (*.nc).
        copy (bool): If True, dataarray is copied in memory. Default is True.

    Returns:
        dataarray (xarray.DataArray): Loaded dataarray.

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
        dataarray (xarray.DataArray): Dataarray to be saved.
        filename (str): Filename (used as <filename>.nc).
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


def loaddfits_2017oct(fitsname, coordtype='azel', starttime=None, endtime=None, pixelids=None, scantype=None):
    """Load a decode array from a DFITS file.

    Args:
        fitsname (str): Name of DFITS file.
        coordtype (str): Coordinate type included into a decode array, azel or radec.
        starttime (int, str or numpy.datetime64): Start time of loaded data.
            It can be specified by the start index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            from the first record.
        endtime (int, str or numpy.datetime64): End time of loaded data.
            It can be specified by the end index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            until the last record.
        pixelids (int or list): Under development.
        scantype (str): Under development.

    Returns:
        decode array (decode.array): Loaded decode array.

    """
    logger = getLogger('decode.io.loaddfits_2017oct')

    with fits.open(fitsname) as hdulist:
        obsinfo = hdulist['OBSINFO'].data
        antlog  = hdulist['ANTENNA'].data
        readout = hdulist['READOUT'].data

        ### obsinfo
        kidids   = obsinfo['kidids'][0]
        kidfreqs = obsinfo['kidfreqs'][0]

        ### start/end time
        t_ant = np.array(antlog['time']).astype(np.datetime64)
        t_out = np.array(readout['starttime']).astype(np.datetime64)

        if starttime is None:
            startindex = 0
        elif isinstance(starttime, int):
            startindex = starttime
        elif isinstance(starttime, str):
            startindex = np.searchsorted(t_out, np.datetime64(starttime))
        elif isinstance(starttime, np.datetime64):
            startindex = np.searchsorted(t_out, starttime)
        else:
            raise ValueError(starttime)

        if endtime is None:
            endindex = len(t_out)
        elif isinstance(endtime, int):
            endindex = endtime
        elif isinstance(endtime, str):
            endindex = np.searchsorted(t_out, np.datetime64(endtime), 'right')
        elif isinstance(endtime, np.datetime64):
            endindex = np.searchsorted(t_out, endtime, 'right')
        else:
            raise ValueError(endtime)

        if t_out[endindex-1] > t_ant[-1]:
            logger.warning('endtime of readout is adjusted to the one of anttena log.')
            endindex = np.searchsorted(t_out, t_ant[-1], 'right')

        logger.debug('startindex: {}'.format(startindex))
        logger.debug('endindex: {}'.format(endindex))
        t_out = t_out[startindex:endindex]

        ### readout
        pixelid   = readout['pixelid'][startindex:endindex]
        arraydata = readout['arraydata'][startindex:endindex]

        ### antenna
        if coordtype == 'azel':
            x = antlog['az'].copy()
            y = antlog['el'].copy()
            try:
                x -= antlog['az_center']
                y -= antlog['el_center']
            except KeyError:
                logger.warning('az/el_center are not included in the antenna log')
                logger.warning('--> they are assumed as 0')
            # finally:
            #     x *= np.cos(np.deg2rad(antlog['el']))
        elif coordtype == 'radec':
            x = antlog['ra'].copy()
            y = antlog['dec'].copy()

        dt_out = (t_out - t_out[0]).astype(np.float64)
        dt_ant = (t_ant - t_out[0]).astype(np.float64)
        x_i = np.interp(dt_out, dt_ant, x)
        y_i = np.interp(dt_out, dt_ant, y)

        ### temporal correction of az/el origins
        el_i = np.interp(dt_out, dt_ant, antlog['el'])
        x_i -= np.median(x_i)
        y_i -= np.median(y_i)
        x_i *= np.cos(np.deg2rad(el_i))

        ### make array
        tcoords  = {'x': x_i, 'y': y_i, 'time': t_out}
        chcoords = {'kidid': kidids, 'kidfq': kidfreqs}
        return dc.array(arraydata, tcoords=tcoords, chcoords=chcoords)
