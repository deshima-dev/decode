# standard library
from pathlib import Path


# dependencies
from decode import convert, load


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = load.dems(DEMS_DIR / "dems_20171111110002.nc.gz")


def test_convert_units() -> None:
    converted = convert.units(DEMS, "interval", "hr")
    assert (DEMS.interval / 3600 == converted.interval).all()
