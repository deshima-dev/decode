# coding: utf-8


# public items
__all__ = ["loaddfits", "savefits", "loadnetcdf", "savenetcdf"]


# standard library
from datetime import datetime
from pytz import timezone
from logging import getLogger
from uuid import uuid4
from pathlib import Path
from pkgutil import get_data


# dependent packages
import yaml
import decode as dc
import numpy as np
import xarray as xr
from astropy.io import fits
from scipy.interpolate import interp1d


# module logger
logger = getLogger(__name__)


def loaddfits(
    fitsname,
    coordtype="azel",
    loadtype="temperature",
    starttime=None,
    endtime=None,
    pixelids=None,
    scantypes=None,
    mode=0,
    **kwargs
):
    """Load a decode array from a DFITS file.

    Args:
        fitsname (str): Name of DFITS file.
        coordtype (str): Coordinate type included into a decode array.
            'azel': Azimuth / elevation.
            'radec': Right ascension / declination.
        loadtype (str): Data unit of xarray.
            'Tsignal': Temperature [K].
            'Psignal': Power [W].
            'amplitude': Amplitude.
            'phase': Phase.
            'linphase': Linear phase.
        starttime (int, str or numpy.datetime64): Start time of loaded data.
            It can be specified by the start index (int),
            the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64).
            Default is None and it means the data will be loaded from the first record.
        endtime (int, str or numpy.datetime64): End time of loaded data.
            It can be specified by the end index (int),
            the time compatible with numpy.datetime64 (str),
            or numpy.datetime64 (numpy.datetime64).
            Default is None and it means the data will be loaded until the last record.
        pixelids (int or list): Under development.
        scantypes (list(str)): Scan types, such as 'GRAD', 'SCAN', 'OFF', 'R'.
        mode (int): Loading mode.
            0: Relative coordinates with cosine projection (RECOMMENDED).
            1: Relative coordinates without cosine projection.
            2: Absolute coordinates.
        kwargs (optional):
            findR (bool): Automatically find R positions.
                ch (int): Representative channel id used for finding R.
                Rth (float): Threshold of R.
                skyth (flaot): Threshold of sky.
                cutnum (int): The number of points of unused data at the edge.
            still (bool): When it is true, scantypes of on/off are manually assigned.
                period (float): On/off period in second for still data.
            shuttle (bool): For shuttle observations.
            xmin_off (float): Minimum x of off-point data.
            xmax_off (float): Maximum x of off-point data.
            xmin_on (float): Minimum x of on-point data.
            xmax_on (float): Maximum x of on-point data.

    Returns:
        decode array (decode.array): Loaded decode array.
    """
    if mode not in [0, 1, 2]:
        raise KeyError(mode)

    logger.info("coordtype starttime endtime mode loadtype")
    logger.info("{} {} {} {} {}".format(coordtype, starttime, endtime, mode, loadtype))

    # pick up kwargs
    # for findR
    findR = kwargs.pop("findR", False)
    ch = kwargs.pop("ch", 0)
    Rth = kwargs.pop("Rth", 280)
    skyth = kwargs.pop("skyth", 150)
    cutnum = kwargs.pop("cutnum", 1)
    # for still
    still = kwargs.pop("still", False)
    period = kwargs.pop("period", 2)
    # for shuttle
    shuttle = kwargs.pop("shuttle", False)
    xmin_off = kwargs.pop("xmin_off", 0)
    xmax_off = kwargs.pop("xmax_off", 0)
    xmin_on = kwargs.pop("xmin_on", 0)
    xmax_on = kwargs.pop("xmax_on", 0)

    # load data
    fitsname = str(Path(fitsname).expanduser())

    with fits.open(fitsname) as hdulist:
        obsinfo = hdulist["OBSINFO"].data
        obshdr = hdulist["OBSINFO"].header
        antlog = hdulist["ANTENNA"].data
        readout = hdulist["READOUT"].data
        wealog = hdulist["WEATHER"].data

    # obsinfo
    masterids = obsinfo["masterids"][0].astype(np.int64)
    kidids = obsinfo["kidids"][0].astype(np.int64)
    kidfreqs = obsinfo["kidfreqs"][0].astype(np.float64)
    kidtypes = obsinfo["kidtypes"][0].astype(np.int64)

    # parse start/end time
    t_ant = np.array(antlog["time"]).astype(np.datetime64)
    t_out = np.array(readout["starttime"]).astype(np.datetime64)
    t_wea = np.array(wealog["time"]).astype(np.datetime64)

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
        endindex = np.searchsorted(t_out, np.datetime64(endtime), "right")
    elif isinstance(endtime, np.datetime64):
        endindex = np.searchsorted(t_out, endtime, "right")
    else:
        raise ValueError(starttime)

    if t_out[endindex - 1] > t_ant[-1]:
        logger.warning("Endtime of readout is adjusted to that of ANTENNA HDU.")
        endindex = np.searchsorted(t_out, t_ant[-1], "right")

    t_out = t_out[startindex:endindex]

    # readout
    if loadtype == "temperature":
        response = readout["Tsignal"][startindex:endindex].astype(np.float64)
    elif loadtype == "power":
        response = readout["Psignal"][startindex:endindex].astype(np.float64)
    elif loadtype == "amplitude":
        response = readout["amplitude"][startindex:endindex].astype(np.float64)
    elif loadtype == "phase":
        response = readout["phase"][startindex:endindex].astype(np.float64)
    elif loadtype == "linphase":
        response = readout["line_phase"][startindex:endindex].astype(np.float64)
    else:
        raise KeyError(loadtype)

    # antenna
    if coordtype == "azel":
        x = antlog["az"].copy()
        y = antlog["el"].copy()
        xref = np.median(antlog["az_center"])
        yref = np.median(antlog["el_center"])
        if mode in [0, 1]:
            x -= antlog["az_center"]
            y -= antlog["el_center"]
            if mode == 0:
                x *= np.cos(np.deg2rad(antlog["el"]))
    elif coordtype == "radec":
        x = antlog["ra"].copy()
        y = antlog["dec"].copy()
        xref = obshdr["RA"]
        yref = obshdr["DEC"]
        if mode in [0, 1]:
            x -= xref
            y -= yref
            if mode == 0:
                x *= np.cos(np.deg2rad(antlog["dec"]))
    else:
        raise KeyError(coordtype)
    scantype = antlog["scantype"]

    # weatherlog
    temp = wealog["temperature"]
    pressure = wealog["pressure"]
    vpressure = wealog["vapor-pressure"]
    windspd = wealog["windspd"]
    winddir = wealog["winddir"]

    # interpolation
    dt_out = (t_out - t_out[0]) / np.timedelta64(1, "s")
    dt_ant = (t_ant - t_out[0]) / np.timedelta64(1, "s")
    dt_wea = (t_wea - t_out[0]) / np.timedelta64(1, "s")
    x_i = np.interp(dt_out, dt_ant, x)
    y_i = np.interp(dt_out, dt_ant, y)

    temp_i = np.interp(dt_out, dt_wea, temp)
    pressure_i = np.interp(dt_out, dt_wea, pressure)
    vpressure_i = np.interp(dt_out, dt_wea, vpressure)
    windspd_i = np.interp(dt_out, dt_wea, windspd)
    winddir_i = np.interp(dt_out, dt_wea, winddir)

    scandict = {t: n for n, t in enumerate(np.unique(scantype))}
    scantype_v = np.zeros(scantype.shape[0], dtype=int)
    for k, v in scandict.items():
        scantype_v[scantype == k] = v
    scantype_vi = interp1d(
        dt_ant,
        scantype_v,
        kind="nearest",
        bounds_error=False,
        fill_value=(scantype_v[0], scantype_v[-1]),
    )(dt_out)
    scantype_i = np.full_like(scantype_vi, "GRAD", dtype="<U8")
    for k, v in scandict.items():
        scantype_i[scantype_vi == v] = k

    # for still data
    if still:
        for n in range(int(dt_out[-1]) // period + 1):
            offmask = (period * 2 * n <= dt_out) & (dt_out < period * (2 * n + 1))
            onmask = (period * (2 * n + 1) <= dt_out) & (dt_out < period * (2 * n + 2))
            scantype_i[offmask] = "OFF"
            scantype_i[onmask] = "SCAN"

    if shuttle:
        offmask = (xmin_off < x_i) & (x_i < xmax_off)
        onmask = (xmin_on < x_i) & (x_i < xmax_on)
        scantype_i[offmask] = "OFF"
        scantype_i[onmask] = "SCAN"
        scantype_i[(~offmask) & (~onmask)] = "JUNK"

    if findR:
        Rindex = np.where(response[:, ch] >= Rth)
        scantype_i[Rindex] = "R"
        movemask = np.hstack(
            [[False] * cutnum, scantype_i[cutnum:] != scantype_i[:-cutnum]]
        ) | np.hstack(
            [scantype_i[:-cutnum] != scantype_i[cutnum:], [False] * cutnum]
        ) & (
            scantype_i == "R"
        )
        scantype_i[movemask] = "JUNK"
        scantype_i[(response[:, ch] > skyth) & (scantype_i != "R")] = "JUNK"
        scantype_i[(response[:, ch] <= skyth) & (scantype_i == "R")] = "JUNK"
        skyindex = np.where(response[:, ch] <= skyth)
        scantype_i_temp = scantype_i.copy()
        scantype_i_temp[skyindex] = "SKY"
        movemask = np.hstack(
            [[False] * cutnum, scantype_i_temp[cutnum:] != scantype_i_temp[:-cutnum]]
        ) | np.hstack(
            [scantype_i_temp[:-cutnum] != scantype_i_temp[cutnum:], [False] * cutnum]
        ) & (
            scantype_i_temp == "SKY"
        )
        scantype_i[movemask] = "JUNK"

    # scanid
    scanid_i = np.cumsum(np.hstack([False, scantype_i[1:] != scantype_i[:-1]]))

    # coordinates
    tcoords = {
        "x": x_i,
        "y": y_i,
        "time": t_out,
        "temp": temp_i,
        "pressure": pressure_i,
        "vapor-pressure": vpressure_i,
        "windspd": windspd_i,
        "winddir": winddir_i,
        "scantype": scantype_i,
        "scanid": scanid_i,
    }
    chcoords = {
        "masterid": masterids,
        "kidid": kidids,
        "kidfq": kidfreqs,
        "kidtp": kidtypes,
    }
    scalarcoords = {
        "coordsys": coordtype.upper(),
        "datatype": loadtype,
        "xref": xref,
        "yref": yref,
    }

    # make array
    array = dc.array(
        response, tcoords=tcoords, chcoords=chcoords, scalarcoords=scalarcoords
    )
    if scantypes is not None:
        mask = np.full(array.shape[0], False)
        for scantype in scantypes:
            mask |= array.scantype == scantype
        array = array[mask]

    return array


def savefits(cube, fitsname, **kwargs):
    """Save a cube to a 3D-cube FITS file.

    Args:
        cube (xarray.DataArray): Cube to be saved.
        fitsname (str): Name of output FITS file.
        kwargs (optional): Other arguments common with astropy.io.fits.writeto().
    """
    # pick up kwargs
    dropdeg = kwargs.pop("dropdeg", False)
    ndim = len(cube.dims)

    # load yaml
    FITSINFO = get_data("decode", "data/fitsinfo.yaml")
    hdrdata = yaml.load(FITSINFO, dc.utils.OrderedLoader)

    # default header
    if ndim == 2:
        header = fits.Header(hdrdata["dcube_2d"])
        data = cube.values.T
    elif ndim == 3:
        if dropdeg:
            header = fits.Header(hdrdata["dcube_2d"])
            data = cube.values[:, :, 0].T
        else:
            header = fits.Header(hdrdata["dcube_3d"])

            kidfq = cube.kidfq.values
            freqrange = ~np.isnan(kidfq)
            orderedfq = np.argsort(kidfq[freqrange])
            newcube = cube[:, :, orderedfq]
            data = newcube.values.T
    else:
        raise TypeError(ndim)

    # update Header
    if cube.coordsys == "AZEL":
        header.update({"CTYPE1": "dAZ", "CTYPE2": "dEL"})
    elif cube.coordsys == "RADEC":
        header.update({"OBSRA": float(cube.xref), "OBSDEC": float(cube.yref)})
    else:
        pass
    header.update(
        {
            "CRVAL1": float(cube.x[0]),
            "CDELT1": float(cube.x[1] - cube.x[0]),
            "CRVAL2": float(cube.y[0]),
            "CDELT2": float(cube.y[1] - cube.y[0]),
            "DATE": datetime.now(timezone("UTC")).isoformat(),
        }
    )
    if (ndim == 3) and (not dropdeg):
        header.update(
            {
                "CRVAL3": float(newcube.kidfq[0]),
                "CDELT3": float(newcube.kidfq[1] - newcube.kidfq[0]),
            }
        )

    fitsname = str(Path(fitsname).expanduser())
    fits.writeto(fitsname, data, header, **kwargs)
    logger.info("{} has been created.".format(fitsname))


def loadnetcdf(filename, copy=True):
    """Load a dataarray from a NetCDF file.

    Args:
        filename (str): Filename (*.nc).
        copy (bool): If True, dataarray is copied in memory. Default is True.

    Returns:
        dataarray (xarray.DataArray): Loaded dataarray.
    """
    filename = str(Path(filename).expanduser())

    if copy:
        dataarray = xr.open_dataarray(filename).copy()
    else:
        dataarray = xr.open_dataarray(filename, chunks={})

    if dataarray.name is None:
        dataarray.name = filename.rstrip(".nc")

    for key, val in dataarray.coords.items():
        if val.dtype.kind == "S":
            dataarray[key] = val.astype("U")
        elif val.dtype == np.int32:
            dataarray[key] = val.astype("i8")

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
    else:
        filename = str(Path(filename).expanduser())

    if not filename.endswith(".nc"):
        filename += ".nc"

    dataarray.to_netcdf(filename)
    logger.info("{} has been created.".format(filename))
