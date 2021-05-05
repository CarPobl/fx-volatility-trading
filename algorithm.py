#%%
# Packages, constants and utils:
from algorithm import SPOT_DATA_FILE
from algorithm import VOL_DATA_FILE

from algorithm.utils import (
    FileDef,
    load_csv_data,
    get_index_of_first,
)
from algorithm.stat_methods import (
    calc_annual_realised_vol,
    calc_moving_percentile,
    forecast_ema_vol,
)
from algorithm.trade_classes import VarianceSwap
from algorithm.graphics import PandasHeatMapPlot

import os
import numpy as np
from datetime import timedelta
import logging

from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.providers.ibmq import IBMQAccountError


not_nan = lambda val: not np.isnan(val)
YEAR_WINDOW = 252  # 1Y (in business_days)


#%%
# Imputs
swap_window_size = 21  # 1M (in business_days)
percentile_window_size = YEAR_WINDOW
ema_lambda = 0.97

x_cells_in_plot = 20
y_cells_in_plot = 30

T_swap = swap_window_size / YEAR_WINDOW  # (in years)


#%%
# Load Market data
file_defs = [
    FileDef(filename=SPOT_DATA_FILE, colname="spot"),
    FileDef(filename=VOL_DATA_FILE, colname="1m_annualised_atmf_vol"),
]
df = load_csv_data(*file_defs)


#%%
# Sort loaded data
df.sort_values(by="date", ascending=True, inplace=True)

#%%
# Volatilities

# Convert annualised implied vol to monthly vol
df["1m_annualised_atmf_vol"] = df["1m_annualised_atmf_vol"] / 100
df["1m_atmf_vol"] = df["1m_annualised_atmf_vol"] * (swap_window_size / 252) ** 0.5

# Calculate implied volatility percentile for a 1-year window
df["1y_implied_vol_percentile"] = np.NaN
df["1y_implied_vol_percentile"] = calc_moving_percentile(
    np.array(df["1m_atmf_vol"]), percentile_window_size
)
print(df.head())

# Calculate forecasted 1M realised vol using EMA
start = get_index_of_first(np.array(df["spot"]), not_nan)
first_year_spots = np.array(np.array(df["spot"].iloc[start : start + YEAR_WINDOW]))
vol_0_monthly = calc_annual_realised_vol(first_year_spots) / (
    12 ** 0.5
)  # From annualised to monthly

df["1m_realised_ema_vol_forecast"] = np.NaN
df["1m_realised_ema_vol_forecast"][start:] = forecast_ema_vol(
    levels=np.array(df["spot"].iloc[start:]),
    vol_0=vol_0_monthly,
    window_size=swap_window_size,
    _lambda=ema_lambda,
)
print(df.head())

#%%
# Remove null values araising from window discrepancies
df.dropna(inplace=True)
print(df.head())

# %%
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
df["profitable"] = df["payoff"] > 0

#%%
# Calculate Vol Carry and mark trades that are profitable for each trade date
df["vol_carry"] = df["1m_atmf_vol"] - df["1m_realised_ema_vol_forecast"]


#%%
# Plot heatmap
plot_cols = ["1y_implied_vol_percentile", "vol_carry", "profitable"]
plot = PandasHeatMapPlot(df[plot_cols], x_cells_in_plot, y_cells_in_plot, *plot_cols)
plot.show(
    xlabel="Vol Percentile(%)",
    ylabel="Vol Carry(%)",
)
# %%
# Feeding results to QSVM to train the model
seed = 0

# Prepare the training data and test data
ml_cols = [col for col in plot_cols if col != "profitable"]
msk = np.random.rand(len(df)) < 0.75  # Train with 75% of data
train_df = df[msk]
test_df = df[~msk]

to_ndarr_dict = lambda x: {
    "profitable": x[x.profitable][ml_cols].values,
    "non_profitable": x[~x.profitable][ml_cols].values,
}

training_input = to_ndarr_dict(train_df)
test_input = to_ndarr_dict(test_df)

feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement="full")
qsvm = QSVM(feature_map, training_input, test_input)

api_token = os.environ["IBM_API_TOKEN"]  # Obtain the IBM API token from env variables

# %%
logging.getLogger("qiskit.providers.ibmq").setLevel(logging.WARNING)
from qiskit import IBMQ

try:
    IBMQ.enable_account(api_token)
except IBMQAccountError:
    pass

provider = IBMQ.get_provider(hub="ibm-q")
backend = provider.backends()[0]
quantum_instance = QuantumInstance(
    backend, shots=1024, seed_simulator=seed, seed_transpiler=seed
)

result = qsvm.run(quantum_instance)
