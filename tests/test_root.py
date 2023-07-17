# dependencies
import decode as dc


# test data
AUTHOR = "Akio Taniguchi"
VERSION = "1.0.0"


# test functions
def test_author() -> None:
    """Make sure that the author is correct."""
    assert dc.__author__ == AUTHOR


def test_version() -> None:
    """Make sure that the version is correct."""
    assert dc.__version__ == VERSION
