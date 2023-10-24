# standard library
from pathlib import Path


# dependencies
from decode import io, select


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = io.open_dems(DEMS_DIR / "dems_20171111110002.nc.gz")


def test_by_min() -> None:
    min = -0.5
    sel = select.by(DEMS[::100], "lon", min=min)
    assert (sel.lon >= min).all()


def test_by_max() -> None:
    max = +0.5
    sel = select.by(DEMS[::100], "lon", max=max)
    assert (sel.lon < max).all()


def test_by_include() -> None:
    include = ["SCAN", "TRAN"]
    sel = select.by(DEMS[::100], "state", include=include)
    assert set(sel.state.values) == set(include)


def test_by_exclude() -> None:
    exclude = ["ACC"]
    sel = select.by(DEMS[::100], "state", exclude=exclude)
    assert set(sel.state.values).isdisjoint(exclude)
