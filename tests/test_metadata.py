import decode as dc


def test_version():
    """Make sure the version is valid."""
    assert dc.__version__ == "0.6.1"


def test_author():
    """Make sure the author is valid."""
    assert dc.__author__ == "DESHIMA software team"
