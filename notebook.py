#%%
from algorithm import SPOT_DATA_FILE
from algorithm import VOL_DATA_FILE

from algorithm.utils import load_csv_data, FileDef, pandas_to_heatmap_matrix
from algorithm.stat_methods import (
    calc_moving_annual_realised_vol,
    calc_moving_percentile,
    forecast_vol,
    gridiserFactory,
)
from algorithm.trade_classes import VarianceSwap

import numpy as np
import pandas as pd
from datetime import timedelta


swap_window_size = 21  # 1M (in business_days)
percentile_window_size = 252  # 1Y (in business_days)
ema_lambda = 0.97
X_CELLS_IN_PLOT = 20
Y_CELLS_IN_PLOT = 30

T_swap = swap_window_size / 252  # (in years)


# Load Market data
file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1m_annualised_atmf_vol"),
]
df = load_csv_data(*file_defs)


# Clean data
df.sort_values(by="date", ascending=True, inplace=True)
df["1m_annualised_atmf_vol"] = df["1m_annualised_atmf_vol"] / 100
df["1m_atmf_vol"] = df["1m_annualised_atmf_vol"] * (swap_window_size / 252) ** 0.5
df.dropna(inplace=True)
# df["1m_atmf_vol_1m_prior"] = np.NaN
# df["1m_atmf_vol_1m_prior"][swap_window_size:] = df["1m_atmf_vol"][:-swap_window_size]


# Calculate realised volatility using a 1-year window
df["1m_realised_annualised_vol"] = np.NaN
df["1m_realised_annualised_vol"][1:] = calc_moving_annual_realised_vol(
    np.array(df["spot"]), swap_window_size
)


# Create a trade to be traded every day at the fair strike K,
# with a variance notional of 1, and maturing in 1M.
# For this, calculate K_fair using a rule-of-thumb described
# Bassu-Strasser-Guichard Varswap paper.

# Although not a fair assumption, there is no enough market data
# to estimatethis value. It would have been possible if volatility
# data for different deltas was provided.
skew_slope = 0

# We also calculate the payoff at maturity of each trade, distinguishing
# between those that are profitable and those that are not.

latest_date = df["date"].max()
fair_trades = [np.NaN] * df.shape[0]
payoffs = [np.NaN] * df.shape[0]
# TODO: loop below is very inefficient. Implement efficient algorithm
for indx, pair in enumerate(df.iterrows()):
    row = pair[1]
    if np.isnan(row["1m_atmf_vol"]) or np.isnan(row["spot"]):
        continue
    fair_strike = VarianceSwap.estimate_fair_strike(
        row["1m_annualised_atmf_vol"], T=T_swap, skew_slope=skew_slope
    )

    trade = VarianceSwap(
        direction="buy",
        underlying="EURUSD",
        trade_date=row["date"],
        value_date=row["date"] + timedelta(days=round(365 * T_swap)),
        strike=fair_strike,
        vega_amount=1,
    )
    fair_trades[indx] = trade

    # Take the exact dates at which the trade was valued. Not just a fixed window
    # We assume that the trade date does not count as valuation, but the value date
    # does.
    dates_in_trade = (df["date"] > trade.trade_date) & (df["date"] <= trade.value_date)
    levels = np.array(df.loc[dates_in_trade, "spot"])

    # If the expiry date later than the latest date in the dataframe, then break
    if latest_date < trade.value_date:
        break
    final_realised_vol = VarianceSwap.calc_final_realised_vol(levels)
    payoffs[indx] = trade.payoff(final_realised_vol)

df["fair_trade"] = fair_trades
df["payoff"] = payoffs


# Calculate implied volatility percentile for a 1-year window
df["1y_implied_vol_percentile"] = np.NaN
df["1y_implied_vol_percentile"] = calc_moving_percentile(
    np.array(df["1m_atmf_vol"]), percentile_window_size
)
df.dropna(inplace=True)


# Calculate forecasted 1M realised EMA vol
vol_0 = df["1m_realised_annualised_vol"].iloc[0] / (
    12 ** 0.5
)  # From annualised to monthly
df["1m_realised_ema_vol_forecast"] = forecast_vol(
    np.array(df["spot"]), "ema", vol_0=vol_0, l=ema_lambda
)


# Calculate Vol Carry
df["vol_carry"] = df["1m_atmf_vol"] - df["1m_realised_ema_vol_forecast"]


# Clasify each row in grid cell:
# TODO: Encapsulate the below code and make more efficient.
shape = (
    {
        "divisions": X_CELLS_IN_PLOT,
        "max": df["1y_implied_vol_percentile"].max(),
        "min": df["1y_implied_vol_percentile"].min(),
    },
    {"divisions": Y_CELLS_IN_PLOT, "max": df["vol_carry"].max(), "min": df["vol_carry"].min()},
)
gridise = gridiserFactory(shape)
df["cell"] = [
    str(gridise(row["1y_implied_vol_percentile"], row["vol_carry"]))
    for _, row in df.iterrows()
]
df.dropna(inplace=True)

df["profitable"] = df["payoff"] > 0
grouping_cols = ["cell", "1y_implied_vol_percentile", "vol_carry", "profitable"]
aggreg_df = df[grouping_cols]

avg_df = pd.pivot_table(
    aggreg_df,
    index=["cell"],
    values=["1y_implied_vol_percentile", "vol_carry"],
    aggfunc=np.average,
)

count_df = pd.pivot_table(
    aggreg_df,
    index=["cell"],
    values="profitable",
    aggfunc=len,
)
count_df.columns = ["total_count"]

sum_df = pd.pivot_table(
    aggreg_df,
    index=["cell"],
    values="profitable",
    aggfunc=sum,
)
sum_df.columns = ["positive_count"]

grouped_df = pd.concat([avg_df, count_df, sum_df], axis=1)
grouped_df["hit_rate"] = grouped_df["positive_count"] / grouped_df["total_count"]

parse_tuple = lambda val: tuple(
    int(num)
    for num in val.replace("'", "")
    .replace('"', "")
    .replace("(", "")
    .replace(")", "")
    .split(",")
)
grouped_df["coordinates"] = [parse_tuple(val) for val in grouped_df.index]

print(grouped_df.head())


# Plot heatmap
# TODO: Improve plot (turn upside down, label axes...)

import matplotlib.pyplot as plt

heat_matrix = pandas_to_heatmap_matrix(grouped_df, "coordinates", "hit_rate")
plt.imshow(heat_matrix.T, cmap='hot', interpolation='nearest')
plt.show()

# %%
