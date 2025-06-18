# standard library
from pathlib import Path


# dependencies
import xarray as xr
from decode import load, select


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = load.dems(
    DEMS_DIR / "dems_20171111110002.nc.gz",
    include_mkid_types=None,
)


def test_select_by_min() -> None:
    min = -0.5
    sel = select.by(DEMS[::100], "lon", min=min)
    assert (sel.lon >= min).all()


def test_select_by_max() -> None:
    max = +0.5
    sel = select.by(DEMS[::100], "lon", max=max)
    assert (sel.lon < max).all()


def test_select_by_range() -> None:
    min, max = -0.5, +0.5
    sel = select.by(DEMS[::100], "lon", min=min, max=max)
    assert ((sel.lon >= min) & (sel.lon < max)).all()


def test_select_by_include() -> None:
    include = ["SCAN", "TRAN"]
    sel = select.by(DEMS[::100], "state", include=include)
    assert set(sel.state.values) == set(include)


def test_select_by_exclude() -> None:
    exclude = ["ACC"]
    sel = select.by(DEMS[::100], "state", exclude=exclude)
    assert set(sel.state.values).isdisjoint(exclude)
