__all__ = ["cube"]


# standard library
from typing import Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Quantity, Unit
from dems.d2 import Cube
from . import convert


# type hints
QuantityLike = Union[Quantity, str]
UnitLike = Union[Unit, str]


def cube(
    dems: xr.DataArray,
    /,
    *,
    skycoord_grid: QuantityLike = "6 arcsec",
    skycoord_units: UnitLike = "arcsec",
) -> xr.DataArray:
    """Make a cube from DEMS.

    Args:
        dems: Input DEMS DataArray to be converted.
        skycoord_grid: Grid size of the sky coordinate axes.
        skycoord_units: Units of the sky coordinate axes.

    Returns:
        Cube DataArray.

    """
    dems = convert.coord_units(dems, "lon", "deg")
    dems = convert.coord_units(dems, "lat", "deg")
    dlon = Quantity(skycoord_grid).to("deg").value
    dlat = Quantity(skycoord_grid).to("deg").value

    lon_min = np.floor(dems.lon.min() / dlon) * dlon
    lon_max = np.ceil(dems.lon.max() / dlon) * dlon
    lat_min = np.floor(dems.lat.min() / dlat) * dlat
    lat_max = np.ceil(dems.lat.max() / dlat) * dlat

    lon = xr.DataArray(np.arange(lon_min, lon_max + dlon, dlon), dims="grid")
    lat = xr.DataArray(np.arange(lat_min, lat_max + dlat, dlat), dims="grid")
    n_lon, n_lat, n_chan = len(lon), len(lat), len(dems.chan)

    i = np.abs(dems.lon - lon).argmin("grid")
    j = np.abs(dems.lat - lat).argmin("grid")
    index = (i + n_lon * j).compute()

    dems = dems.copy(data=dems.data)
    dems.coords.update({"index": index})
    gridded = dems.groupby("index").mean("time")

    data = np.full([n_lat * n_lon, n_chan], np.nan)
    data[gridded.index.values] = gridded.values
    data = data.reshape(n_lat, n_lon, n_chan).transpose(2, 0, 1)

    cube = Cube.new(
        data=data,
        lat=lat,
        lon=lon,
        chan=dems.chan,
        frame=dems.frame,
        d2_mkid_id=dems.d2_mkid_id,
        d2_mkid_frequency=dems.d2_mkid_frequency,
        d2_mkid_type=dems.d2_mkid_type,
    )
    cube = convert.coord_units(cube, "lon", skycoord_units)
    cube = convert.coord_units(cube, "lat", skycoord_units)
    return cube
