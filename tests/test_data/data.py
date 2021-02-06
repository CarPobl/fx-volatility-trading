import numpy as np

dummy_moving_squared_sums = np.array(
    [0.0, 0.0, 0.0, 30.0, 54.0, 86.0, 126.0, 174.0, 230.0]
)


dummy_levels = np.array(
    [1.221, 1.25, 1.258, 1.228, 1.226, 1.217, 1.283, 1.231, 1.219, 1.232]
)


expected_moving_reslised_vol = np.array(
    [0, 0, 0.272, 0.199, 0.201, 0.423, 0.536, 0.538, 0.348]
)


dummy_implied_vols = np.array([0.8, 0.6, 0.2, 0.1, 0.9, 0.7, 0.5, 0.4, 0.7, 0.3])


expected_implied_vol_percentiles = np.array(
    [np.NaN, np.NaN, 0, 0, 2 / 3, 1 / 3, 0, 0, 2 / 3, 0]
)


expected_ema_forecast = np.array(
    [0.21, 0.199, 0.189, 0.180, 0.170, 0.162, 0.154, 0.147, 0.139, 0.132]
)