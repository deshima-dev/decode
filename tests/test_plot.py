# standard library
from pathlib import Path


# dependencies
import matplotlib.pyplot as plt
from decode import load, plot


# constants
DEMS_DIR = Path(__file__).parents[1] / "data" / "dems"
DEMS = load.dems(
    DEMS_DIR / "dems_20171111110002.nc.gz",
    include_mkid_types=None,
)


def test_plot_data_1d_time() -> None:
    data = DEMS[:100, 0]
    xlabel = DEMS.time.long_name

    fig, ax = plt.subplots()
    plot.data(data, ax=ax)
    assert ax.get_xlabel().replace("\n", " ") == xlabel


def test_plot_data_1d_chan() -> None:
    data = DEMS[0]
    xlabel = DEMS.chan.long_name

    fig, ax = plt.subplots()
    plot.data(data, ax=ax)
    assert ax.get_xlabel().replace("\n", " ") == xlabel


def test_plot_data_2d() -> None:
    data = DEMS[:100, :100]
    xlabel = DEMS.chan.long_name
    ylabel = DEMS.time.long_name

    fig, ax = plt.subplots()
    plot.data(data, ax=ax)
    assert ax.get_xlabel().replace("\n", " ") == xlabel
    assert ax.get_ylabel().replace("\n", " ") == ylabel


def test_plot_state_sky() -> None:
    data = DEMS[:100, :100]
    xlabel = f"{DEMS.lon.long_name} [{DEMS.lon.units}]"
    ylabel = f"{DEMS.lat.long_name} [{DEMS.lat.units}]"

    fig, ax = plt.subplots()
    plot.state(data, on="sky", ax=ax)
    assert ax.get_xlabel().replace("\n", " ") == xlabel
    assert ax.get_ylabel().replace("\n", " ") == ylabel


def test_plot_state_time() -> None:
    data = DEMS[:100, :100]
    xlabel = DEMS.time.long_name
    ylabel = DEMS.state.long_name

    fig, ax = plt.subplots()
    plot.state(data, on="time", ax=ax)
    assert ax.get_xlabel().replace("\n", " ") == xlabel
    assert ax.get_ylabel().replace("\n", " ") == ylabel
