__all__ = ["units"]


# standard library
from typing import Optional, Union


# dependencies
import xarray as xr
from astropy.units import Equivalency, Quantity, Unit


# type hints
UnitLike = Union[Unit, str]


def units(
    dems: xr.DataArray,
    coord_name: str,
    new_units: UnitLike,
    /,
    *,
    equivalencies: Optional[Equivalency] = None,
    inplace: bool = False,
) -> xr.DataArray:
    """Convert units of a coordinate of DEMS.

    Args:
        dems: Target DEMS DataArray.
        coord_name: Name of the coordinate for the conversion.
        new_units: Units to be converted from the current ones.
        equivalencies: Optional Astropy equivalencies.
        inplace: Whether the units are converted in-place.

    Returns:
        DEMS DataArray with the coordinate units converted.

    """
    if not inplace:
        # deepcopy except for data
        dems = dems.copy(data=dems.data)

    coord = dems[coord_name]

    if (units := coord.attrs.get("units")) is None:
        return dems

    coord.values = Quantity(coord, units).to(new_units, equivalencies)
    coord.attrs["units"] = new_units
    return dems
