# standard library
from pathlib import Path


# dependencies
from decode import load
from pytest import mark


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS_ALL = map(Path, DEMS_DIR.glob("*.nc.gz"))


# test functions
def test_load_atm() -> None:
    load.atm(type="eta")
    load.atm(type="tau")


@mark.parametrize("dems", DEMS_ALL)
def test_load_dems(dems: Path) -> None:
    load.dems(dems, include_mkid_types=None)
