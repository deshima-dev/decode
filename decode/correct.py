__all__ = ["for_atmosphere"]


# standard library
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TypeVar, Union


# dependencies
import xarray as xr
from . import select


# type hints
T = TypeVar("T", bool, int, float, str, datetime, timedelta)
Multiple = Union[Sequence[T], T]


def for_atmosphere(
    dems: xr.DataArray,
    include_on: Multiple[T],
    include_off: Multiple[T],
    include_r: Multiple[T],
    /,
    method_off: str = "linear",
    method_r: str = "nearest",
    T_amb: float = 273.0,
) -> xr.DataArray:
    """Correct for the atmospheric transmission of DEMS.

    Args:
        dems: Input DEMS DataArray with a correct state coordinate.
        include_on: State value(s) to be assigned to on-source samples.
        include_off: State value(s) to be assigned to off-source samples.
        include_r: State value(s) to be assigned to hot-load samples.
        method_off: Interpolation method of the off-source samples
            to the measured time of the on-source samples.
        method_r: Interpolation method of the hot-load samples
            to the measured time of the on-source samples.
        T_amb: Ambient temperature assumed for correction.

    Returns:
        DEMS DataArray of the on-source samples in the Ta* scale.

    """
    Tb_on = select.by(dems, "state", include=include_on)
    Tb_off = select.by(dems, "state", include=include_off)
    Tb_r = select.by(dems, "state", include=include_r)

    Tb_off_mean = Tb_off.groupby("scan").map(mean_in_time)
    Tb_r_mean = Tb_r.groupby("scan").map(mean_in_time)

    Tb_off_ip = Tb_off_mean.interp_like(
        Tb_on,
        method=method_off,  # type: ignore
        kwargs={"fill_value": "extrapolate"},
    ).values
    Tb_r_ip = Tb_r_mean.interp_like(
        Tb_on,
        method=method_r,  # type: ignore
        kwargs={"fill_value": "extrapolate"},
    ).values

    return T_amb * (Tb_on - Tb_off_ip) / (Tb_r_ip - Tb_off_ip)


def mean_in_time(dems: xr.DataArray) -> xr.DataArray:
    """Similar to DataArray.mean but keeps middle time."""
    middle = dems[len(dems) // 2 : len(dems) // 2 + 1]
    return xr.zeros_like(middle) + dems.mean("time")
