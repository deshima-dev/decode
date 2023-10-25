__all__ = ["cube"]


# standard library
from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Quantity
from xarray_dataclasses import AsDataArray, Coord, Data


# type hints
Angle = Union[Quantity, str]
Chan = Literal["chan"]
Lat = Literal["lat"]
Lon = Literal["lon"]
_ = Tuple[()]


@dataclass
class Cube(AsDataArray):
    """Cube of DESHIMA 2.0."""

    data: Data[Tuple[Lon, Lat, Chan], Any]
    lon: Coord[Lon, float] = 0.0
    lat: Coord[Lat, float] = 0.0
    frame: Coord[_, str] = "altaz"
    d2_mkid_frequency: Coord[Chan, float] = 0.0
    d2_mkid_id: Coord[Chan, int] = 0
    d2_mkid_type: Coord[Chan, int] = 0


def cube(
    dems: xr.DataArray,
    /,
    *,
    gridsize_lon: Angle = "3 arcsec",
    gridsize_lat: Angle = "3 arcsec",
) -> xr.DataArray:
    """Make a cube from DEMS.

    Args:
        dems: Input DEMS DataArray.
        gridsize_lon: Grid size of the longitude axis.
        gridsize_lat: Grid size of the latitude axis.

    Returns:
        Cube DataArray.

    """
    dlon = Quantity(gridsize_lon).to("deg").value
    dlat = Quantity(gridsize_lat).to("deg").value
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

    dems = dems.copy()
    dems.coords.update({"index": index})
    grid = dems.groupby("index").mean("time")

    temp = np.full([n_lat * n_lon, n_chan], np.nan)
    temp[grid.index.values] = grid.values
    data = temp.reshape((n_lat, n_lon, n_chan)).swapaxes(0, 1)

    return Cube.new(
        data=data,
        lon=lon.values,
        lat=lat.values,
        frame=dems.frame.values,
        d2_mkid_id=dems.d2_mkid_id.values,
        d2_mkid_frequency=dems.d2_mkid_frequency.values,
        d2_mkid_type=dems.d2_mkid_type.values,
    )
