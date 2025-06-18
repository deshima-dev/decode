# standard library
from pathlib import Path


# dependencies
from decode import assign, load


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = load.dems(
    DEMS_DIR / "dems_20171111110002.nc.gz",
    include_mkid_types=None,
)


def test_assign_scan() -> None:
    assert assign.scan(DEMS).scan.min() == 0
    assert assign.scan(DEMS).scan.max() == 244
