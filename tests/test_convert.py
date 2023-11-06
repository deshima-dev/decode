# standard library
from pathlib import Path


# dependencies
from decode import convert, load


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = load.dems(DEMS_DIR / "dems_20171111110002.nc.gz")


def test_convert_frame() -> None:
    converted = convert.frame(DEMS, "relative")
    assert (converted.lon_origin == 0.0).all()
    assert (converted.lat_origin == 0.0).all()
    assert (converted.frame == "relative-altaz").all()


def test_convert_units() -> None:
    converted = convert.units(DEMS, "interval", "hr")
    assert (DEMS.interval / 3600 == converted.interval).all()
