from algorithm import SPOT_DATA_FILE
from algorithm import VOL_DATA_FILE

from algorithm.utils import load_csv_data, FileDef
from algorithm.stat_methods import (
    calc_moving_annual_realised_vol,
    calc_moving_percentile,
)

import numpy as np


window_size = 252  # 1Y


# Load Market data
file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1y_atmf_vol"),
]
df = load_csv_data(*file_defs)


# Clean data
df.sort_values(by="date", ascending=True, inplace=True)
df["1y_atmf_vol"] = df["1y_atmf_vol"] / 100
df.dropna(inplace=True)
df.reset_index(inplace=True)


# Calculate realised volatility using a 1-year window
df["1y_realised_vol"] = np.NaN
df["1y_realised_vol"][1:] = calc_moving_annual_realised_vol(
    np.array(df["spot"]), window_size
)


# Calculate implied volatility percentile for a 1-year window
df["1y_implied_vol_percentile"] = np.NaN
df["1y_implied_vol_percentile"] = calc_moving_percentile(
    np.array(df["1y_atmf_vol"]), window_size
)
df.dropna(inplace=True)
print(df)


# Calculate forecasted 1Y implied EMA vol

# Calculate Vol Carry

# Create a trade to be traded every day at the fair strike K,
# with a variance notional of 1, and maturing in 1Y.
# For this, calculate K_fair using a rule-of-thumb described
# Bassu-Strasser-Guichard Varswap paper.


# Calculate the payoff at maturity of each trade, distinguishing
# between those that are profitable and those that not.
