import numpy as np
import math


def calc_annual_realised_vol(levels: np.ndarray) -> np.ndarray:
    n = len(levels)
    levels_t_minus1 = np.zeros(n)
    levels_t_minus1[1:] = levels[:-1]
    log_returns = np.log((levels / levels_t_minus1)[1:])
    return math.sqrt(252 * np.matmul(log_returns, log_returns.T) / n)


def sum_squares_moving_window(arr: np.ndarray, window_size: int) -> np.ndarray:
    n = len(arr)
    transition_matrix = np.zeros([n, n])
    i = 0
    while i <= n - window_size:
        transition_matrix[window_size + i - 1, i : window_size + i] = arr[
            i : window_size + i
        ]
        i += 1
    return np.matmul(transition_matrix, arr)


def calc_moving_annual_realised_vol(
    levels: np.ndarray, window_size: int, by_matrix: bool = True
) -> np.ndarray:
    if not by_matrix:
        n = len(levels)
        output = np.zeros(n)
        i = 0
        while i < n - window_size:
            in_levels = levels[i : i + window_size + 1]
            output[i + window_size] = calc_annual_realised_vol(in_levels)
            i += 1
        return output[1:]
    else:
        n = len(levels)
        levels_t_minus1 = np.zeros(n)
        levels_t_minus1[1:] = levels[:-1]
        log_returns = np.log((levels / levels_t_minus1)[1:])
        summed_squares = sum_squares_moving_window(log_returns, window_size)
        return np.sqrt(252 * summed_squares / (window_size + 1))


def calc_percentile(arr: np.ndarray, value: float) -> float:
    return sum(arr < value) / float(len(arr))


def calc_moving_percentile(arr: np.ndarray, window_size: int) -> np.ndarray:
    n = len(arr)
    output = np.zeros(n)
    output[:] = np.NaN
    i = 0
    while i <= n - window_size:
        in_arr = arr[i : i + window_size]
        value = arr[i + window_size - 1]
        output[i + window_size - 1] = calc_percentile(in_arr, value)
        i += 1
    return output
