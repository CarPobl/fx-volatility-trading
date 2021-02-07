from algorithm.stat_methods import (
    calc_annual_realised_vol,
    sum_squares_moving_window,
    calc_moving_annual_realised_vol,
    calc_moving_percentile,
    forecast_vol,
    gridiserFactory,
)

import unittest
import numpy as np

from . import FILE_DEFS
from tests.test_data import data as test_data


def test_calc_annual_realised_vol():
    expected_result = 0.387
    result = calc_annual_realised_vol(test_data.dummy_levels)
    assert round(result, 3) == expected_result


def test_sum_squares_moving_window():
    arr = np.array(range(1, 10))
    results = sum_squares_moving_window(arr, 4)
    assert all(results == test_data.dummy_moving_squared_sums)


def test_calc_moving_annual_realised_vol():
    results = calc_moving_annual_realised_vol(test_data.dummy_levels, 3, False)
    assert all(np.round(results, 3)[2:] == test_data.expected_moving_reslised_vol[2:])
    results = calc_moving_annual_realised_vol(test_data.dummy_levels, 3, True)
    assert all(np.round(results, 3)[2:] == test_data.expected_moving_reslised_vol[2:])


def test_calc_moving_perentile():
    window_size = 3
    results = calc_moving_percentile(test_data.dummy_implied_vols, window_size)
    assert all(
        results[window_size - 1 :]
        == test_data.expected_implied_vol_percentiles[window_size - 1 :]
    )


def test_ema_forecast():
    vol_0 = 0.210
    results = forecast_vol(test_data.dummy_levels, "ema", vol_0=vol_0, l=0.9)
    assert all(np.round(results, 3) == test_data.expected_ema_forecast)


def test_gridiserFactory():
    shape = (
        {"divisions": 3, "max": 3, "min": -9},
        {"divisions": 2, "max": 6, "min": -2},
    )
    gridise = gridiserFactory(shape)
    assert gridise(-8.7, 0.1) == (0, 0)
    assert gridise(0, 0) == (2, 0)
    assert gridise(-3, 4) == (1, 1)
    with unittest.TestCase.assertRaises(None, ValueError):
        gridise(-10, 0)
    with unittest.TestCase.assertRaises(None, ValueError):
        gridise(0, 10)
