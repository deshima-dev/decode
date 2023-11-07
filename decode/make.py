__all__ = ["cube"]


# standard library
from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union


# dependencies
import numpy as np
import xarray as xr
from astropy.units import Quantity, Unit
from xarray_dataclasses import AsDataArray, Attr, Coordof, Data
from . import convert


# type hints
QuantityLike = Union[Quantity, str]
UnitLike = Union[Unit, str]
Ln = Literal["lon"]
Lt = Literal["lat"]
Ch = Literal["chan"]


@dataclass
class Weight:
    data: Data[Tuple[Ln, Lt, Ch], float]
    long_name: Attr[str] = "Data weights"


@dataclass
class Lon:
    data: Data[Ln, float]
    long_name: Attr[str] = "Sky longitude"
    units: Attr[str] = "deg"


@dataclass
class Lat:
    data: Data[Lt, float]
    long_name: Attr[str] = "Sky latitude"
    units: Attr[str] = "deg"


@dataclass
class Chan:
    data: Data[Ch, int]
    long_name: Attr[str] = "Channel ID"


@dataclass
class Frame:
    data: Data[Tuple[()], str]
    long_name: Attr[str] = "Sky coordinate frame"


@dataclass
class D2MkidID:
    data: Data[Ch, int]
    long_name: Attr[str] = "[DESHIMA 2.0] MKID ID"


@dataclass
class D2MkidType:
    data: Data[Ch, str]
    long_name: Attr[str] = "[DESHIMA 2.0] MKID type"


@dataclass
class D2MkidFrequency:
    data: Data[Ch, float]
    long_name: Attr[str] = "[DESHIMA 2.0] MKID center frequency"
    units: Attr[str] = "Hz"


@dataclass
class Cube(AsDataArray):
    """Cube of DESHIMA 2.0."""

    data: Data[Tuple[Ln, Lt, Ch], Any]
    weight: Coordof[Weight] = 1.0
    lon: Coordof[Lon] = 0.0
    lat: Coordof[Lat] = 0.0
    chan: Coordof[Chan] = 0
    frame: Coordof[Frame] = "altaz"
    d2_mkid_frequency: Coordof[D2MkidFrequency] = 0.0
    d2_mkid_id: Coordof[D2MkidID] = 0.0
    d2_mkid_type: Coordof[D2MkidType] = ""


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
    dlon = Quantity(skycoord_grid).to(dems.lon.attrs["units"]).value
    dlat = Quantity(skycoord_grid).to(dems.lat.attrs["units"]).value
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
    grid = dems.groupby("index").mean("time")

    temp = np.full([n_lat * n_lon, n_chan], np.nan)
    temp[grid.index.values] = grid.values
    data = temp.reshape((n_lat, n_lon, n_chan)).swapaxes(0, 1)

    cube = Cube.new(
        data=data,
        lon=lon.values,
        lat=lat.values,
        frame=dems.frame.values,
        d2_mkid_id=dems.d2_mkid_id.values,
        d2_mkid_frequency=dems.d2_mkid_frequency.values,
        d2_mkid_type=dems.d2_mkid_type.values,
    )
    cube = convert.units(cube, "lon", skycoord_units)
    cube = convert.units(cube, "lat", skycoord_units)
    return cube
