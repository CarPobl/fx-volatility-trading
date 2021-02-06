from algorithm.stat_methods import (
    calc_annual_realised_vol,
    sum_squares_moving_window,
    calc_moving_annual_realised_vol,
)
from algorithm.utils import load_csv_data
import numpy as np

from . import FILE_DEFS

simple_levels = np.array(
    [1.221, 1.25, 1.258, 1.228, 1.226, 1.217, 1.283, 1.231, 1.219, 1.232]
)

expected_moving_reslised_vol = np.array(
    [0, 0, 0.272, 0.199, 0.201, 0.423, 0.536, 0.538, 0.348]
)

def test_calc_annual_realised_vol():
    global simple_levels
    expected_result = 0.387
    result = calc_annual_realised_vol(simple_levels)
    assert round(result, 3) == expected_result


def test_sum_squares_moving_window():
    arr = np.array(range(1, 10))
    expected_results = np.array([0.0, 0.0, 0.0, 30.0, 54.0, 86.0, 126.0, 174.0, 230.0])
    results = sum_squares_moving_window(arr, 4)
    assert all(results == expected_results)


def test_calc_moving_annual_realised_vol():
    global simple_levels, expected_moving_reslised_vol
    results = calc_moving_annual_realised_vol(simple_levels, 3, False)
    assert all(np.round(results, 3) == expected_moving_reslised_vol)
    
    
