# dependencies
import numpy as np
import xarray as xr
from decode import stats


DATA = xr.DataArray(
    np.arange(36).reshape(6, 6),
    dims=("x", "y"),
    coords={
        "x": np.arange(0, 6),
        "y": np.arange(6, 12),
        "c1": ("x", np.array(list("abcdef"))),
        "c2": ("x", np.arange(6)),
    },
)


def test_all() -> None:
    expected = xr.DataArray(
        np.array([[False, True], [True, True], [True, True]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.all(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_any() -> None:
    expected = xr.DataArray(
        np.array([[True, True], [True, True], [True, True]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.any(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_count() -> None:
    expected = xr.DataArray(
        np.array([[6, 6], [6, 6], [6, 6]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.count(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_first() -> None:
    expected = xr.DataArray(
        np.array([[0, 3], [12, 15], [24, 27]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.first(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_last() -> None:
    expected = xr.DataArray(
        np.array([[8, 11], [20, 23], [32, 35]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.last(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_max() -> None:
    expected = xr.DataArray(
        np.array([[8, 11], [20, 23], [32, 35]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.max(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_mean() -> None:
    expected = xr.DataArray(
        np.array([[4, 7], [16, 19], [28, 31]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.mean(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_median() -> None:
    expected = xr.DataArray(
        np.array([[4, 7], [16, 19], [28, 31]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.median(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_min() -> None:
    expected = xr.DataArray(
        np.array([[0, 3], [12, 15], [24, 27]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.min(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_prod() -> None:
    expected = xr.DataArray(
        np.array(
            [
                [0, 59400],
                [14938560, 43354080],
                [464256000, 860955480],
            ]
        ),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.prod(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_std() -> None:
    expected = xr.DataArray(
        np.array(
            [
                [3.1091263510296048, 3.1091263510296048],
                [3.1091263510296048, 3.1091263510296048],
                [3.1091263510296048, 3.1091263510296048],
            ]
        ),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.std(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_sum() -> None:
    expected = xr.DataArray(
        np.array([[24, 42], [96, 114], [168, 186]]),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.sum(DATA, dim={"x": 2, "y": 3}),
        expected,
    )


def test_var() -> None:
    expected = xr.DataArray(
        np.array(
            [
                [9.666666666666666, 9.666666666666666],
                [9.666666666666666, 9.666666666666666],
                [9.666666666666666, 9.666666666666666],
            ]
        ),
        dims=("x", "y"),
        coords={
            "x": np.array([0.5, 2.5, 4.5]),
            "y": np.array([7.0, 10.0]),
            "c1": ("x", np.array(list("ace"))),
            "c2": ("x", np.array([0.5, 2.5, 4.5])),
        },
    )
    xr.testing.assert_equal(
        stats.var(DATA, dim={"x": 2, "y": 3}),
        expected,
    )
