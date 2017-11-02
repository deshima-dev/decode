# coding: utf-8

# public items
__all__ = [
    'loaddfits',
    'savefits',
    'loadnetcdf',
    'savenetcdf',
    'calcazelorigin',
]

# standard library
from logging import getLogger

# dependent packages
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits


def loaddfits(fitsname, coordtype='azel', starttime=None, endtime=None, pixelids=None,
              scantype=None, mode=0, **kwargs):
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
        mode (int):
            0: The origin of relative az/el is corrected with cosine projection (RECOMMENDED).
            1: The origin of relative az/el is corrected without cosine projection.
            2: The origin of relative az/el is not corrected.
            3: Absolute az/el.

    Returns:
        decode array (decode.array): Loaded decode array.

    """
    if mode in [0, 1, 2, 3]:
        logger = getLogger('decode.io.loaddfits (mode={})'.format(mode))
    else:
        raise KeyError(mode)

    ### open hdulist
    hdulist = fits.open(fitsname)

    ### load data
    obsinfo = hdulist['OBSINFO'].data
    obshdr  = hdulist['OBSINFO'].header
    antlog  = hdulist['ANTENNA'].data
    readout = hdulist['READOUT'].data

    ### obsinfo
    masterids = obsinfo['masterids'][0].astype(np.int64)
    kidids    = obsinfo['kidids'][0].astype(np.int64)
    kidfreqs  = obsinfo['kidfreqs'][0].astype(np.float64)
    kidtypes  = obsinfo['kidtypes'][0].astype(np.int64)

    ### parse start/end time
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
        endindex = t_out.shape[0]
    elif isinstance(endtime, int):
        endindex = endtime
    elif isinstance(endtime, str):
        endindex = np.searchsorted(t_out, np.datetime64(endtime), 'right')
    elif isinstance(endtime, np.datetime64):
        endindex = np.searchsorted(t_out, endtime, 'right')
    else:
        raise ValueError(starttime)

    if t_out[endindex-1] > t_ant[-1]:
        logger.warning('Endtime of readout is adjusted to that of ANTENNA HDU.')
        endindex = np.searchsorted(t_out, t_ant[-1], 'right')

    logger.debug('startindex: {}'.format(startindex))
    logger.debug('endindex: {}'.format(endindex))
    t_out = t_out[startindex:endindex]

    ### readout
    pixelid   = readout['pixelid'][startindex:endindex]
    # phase     = readout['phase'][startindex:endindex]
    # amplitude = readout['amplitude'][startindex:endindex]
    Tsignal   = readout['Tsignal'][startindex:endindex].astype(np.float64)
    Psignal   = readout['Psignal'][startindex:endindex].astype(np.float64)

    ### antenna
    if coordtype == 'azel':
        x = antlog['az'].copy()
        y = antlog['el'].copy()
        try:
            if mode == 0 or mode == 1 or mode == 2:
                x -= antlog['az_center']
                y -= antlog['el_center']
            elif mode == 3:
                pass
        except KeyError:
            logger.warning('Az_center/el_center are not included in the ANTENNA HDU.')
    elif coordtype == 'radec':
        x  = antlog['ra'].copy()
        y  = antlog['dec'].copy()
        x -= obshdr['RA']
        y -= obshdr['DEC']

    ### interpolation
    dt_out  = (t_out - t_out[0]).astype(np.float64)
    dt_ant  = (t_ant - t_out[0]).astype(np.float64)
    x_i     = np.interp(dt_out, dt_ant, x)
    y_i     = np.interp(dt_out, dt_ant, y)

    ### temporal correction of az/el origins
    ### relative az/el原点の問題が解消するまでの暫定的な処置
    if (mode == 0 or mode == 1) and coordtype == 'azel':
        if 'x_m' in kwargs and 'y_m' in kwargs:
            x_m = kwargs['x_m']
            y_m = kwargs['y_m']
        else:
            x_m  = np.median(x_i)
            y_m  = np.median(y_i)
        logger.debug('x_median: {}'.format(x_m))
        logger.debug('y_median: {}'.format(y_m))
        x_i -= x_m
        y_i -= y_m
        if mode == 0:
            el_i = np.interp(dt_out, dt_ant, antlog['el'])
            x_i *= np.cos(np.deg2rad(el_i))

    ### coordinates
    tcoords    = {'x': x_i, 'y': y_i, 'time': t_out}
    chcoords   = {'masteid': masterids, 'kidid': kidids, 'kidfq': kidfreqs, 'kidtp': kidtypes}
    datacoords = {'Psignal': Psignal}

    ### make array
    array = dc.array(Tsignal, tcoords=tcoords, chcoords=chcoords, datacoords=datacoords)

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


def calcazelorigin(fitsnames, starttimes=None, endtimes=None, pixelids=None, scantype=None):
    """Calculate az/el origins from DFITS.

    Args:
        fitsnames (list(str)): Names of DFITS file.
        starttimes (list(int, str or numpy.datetime64)): Start times of loaded data.
            It can be specified by the start index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            from the first record.
        endtimes (list(int, str or numpy.datetime64)): End times of loaded data.
            It can be specified by the end index (int), the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64). Default is None and it means the data will be loaded
            until the last record.
        pixelids (int or list): Under development.
        scantype (str): Under development.

    Returns:
        x_m, y_m (float, float): Averaged median of coordinates
    """
    logger = getLogger('decode.io.calcazelorigin')

    if starttimes is None:
        starttimes = [None for fitsname in fitsnames]
    if endtimes is None:
        endtimes   = [None for fitsname in fitsnames]

    x_ms = np.array([])
    y_ms = np.array([])
    for fitsname, starttime, endtime in zip(fitsnames, starttimes, endtimes):
        ### open hdulist
        hdulist = fits.open(fitsname)

        ### load data
        antlog  = hdulist['ANTENNA'].data
        readout = hdulist['READOUT'].data

        ### parse start/end time
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
            endindex = t_out.shape[0]
        elif isinstance(endtime, int):
            endindex = endtime
        elif isinstance(endtime, str):
            endindex = np.searchsorted(t_out, np.datetime64(endtime), 'right')
        elif isinstance(endtime, np.datetime64):
            endindex = np.searchsorted(t_out, endtime, 'right')
        else:
            raise ValueError(starttime)

        if t_out[endindex-1] > t_ant[-1]:
            logger.warning('Endtime of readout is adjusted to that of ANTENNA HDU.')
            endindex = np.searchsorted(t_out, t_ant[-1], 'right')

        t_out = t_out[startindex:endindex]

        ### antenna
        x = antlog['az'].copy()
        y = antlog['el'].copy()
        try:
            x -= antlog['az_center']
            y -= antlog['el_center']
        except KeyError:
            logger.warning('Az_center/el_center are not included in the ANTENNA HDU.')

        ### interpolation
        dt_out  = (t_out - t_out[0]).astype(np.float64)
        dt_ant  = (t_ant - t_out[0]).astype(np.float64)
        x_i     = np.interp(dt_out, dt_ant, x)
        y_i     = np.interp(dt_out, dt_ant, y)

        ### temporal correction of az/el origins
        ### relative az/el原点の問題が解消するまでの暫定的な処置
        x_ms = np.append(x_ms, np.median(x_i))
        y_ms = np.append(y_ms, np.median(y_i))

        ### close hdu
        hdulist.close()

    return x_ms.mean(), y_ms.mean()
