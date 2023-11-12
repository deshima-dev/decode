# dependencies
import numpy as np
import xarray as xr
from decode import utils
from dems.d2 import MS


def test_mad() -> None:
    dems = MS.new(np.arange(25).reshape(5, 5))
    assert (utils.mad(dems, "time") == 5.0).all()
