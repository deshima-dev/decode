# dependencies
import numpy as np
import xarray as xr
from decode import utils
from dems.d2 import MS


def test_mad() -> None:
    dems = MS.new(np.arange(25).reshape(5, 5))
    assert (utils.mad(dems, "time") == 5.0).all()


def test_phaseof() -> None:
    tester = xr.DataArray([0, 1, 1, 2, 2, 2, 1, 0])
    expected = xr.DataArray([0, 1, 1, 2, 2, 2, 3, 4])
    assert (utils.phaseof(tester) == expected).all()
