from . import SPOT_DATA_FILE
from . import VOL_DATA_FILE

from utils import load_csv_data, FileDef


# Load Market data
file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1y_atmf_vol"),
]
market_data = load_csv_data(*file_defs)


# Calculate realised volatility using a 1-year window


# Calculate volatility percentile for a 1-year window


# Calculate skew slope


# Remove rows with Null values


# Create a trade to be traded every day at the fair strike K,
# with a variance notional of 1, and maturing in 1Y.
# For this, calculate K_fair using a rule-of-thumb described
# Bassu-Strasser-Guichard Varswap paper.


# Calculate the payoff at maturity of each trade, distinguishing
# between those that are profitable and those that not.
