from datetime import date
import math

from algorithm.trade_classes import Trade, VarianceSwap


def test_trade_parent_class():
    kwargs = {
        "direction": "buy",
        "underlying": "EURUSD",
        "trade_date": date(2020, 1, 1),
        "value_date": date(2021, 1, 1),
        "dummy_param": "dummy",
    }
    expected_str = (
        f'{kwargs["direction"]} {kwargs["underlying"]} '
        f'| {kwargs["trade_date"]} - {kwargs["value_date"]}'
    )
    trade = Trade(**kwargs)
    assert str(trade) == expected_str
    assert trade.dummy_param == "dummy"


def test_variance_swap():
    kwargs = {
        "direction": "buy",
        "underlying": "EURUSD",
        "trade_date": date(2020, 1, 1),
        "value_date": date(2021, 1, 1),
        "strike": 20,
        "var_amount": 5_000,
    }
    vega_amount = kwargs["var_amount"] * (2 * kwargs["strike"])
    expected_str = (
        f"{kwargs['underlying']}: {kwargs['direction']} {vega_amount}"
        f"@{kwargs['strike']} | {kwargs['trade_date']} - {kwargs['value_date']}"
    )
    trade = VarianceSwap(**kwargs)
    assert str(trade) == expected_str

    assert trade.payoff(realised_vol=kwargs["strike"]) == 0

    # Examples as per varswap paper.

    valuation_date = date(2020, 4, 1)  # 3M after issuance
    mtm = trade.calc_mtm(
        realised_vol=15, fair_strike=19, r=0.02, valuation_date=valuation_date
    )
    assert round(mtm) == -357_247

    params = {"vol_atmf": 0.2, "T": 2}
    skew_slope = 2 / (100 - 90)
    linear_skew_fair_strike = VarianceSwap.estimate_fair_strike(
        **params, skew_slope=skew_slope
    )
    assert round(linear_skew_fair_strike * 100, 1) == 22.3

    skew_slope = -2 / 100 / math.log(0.9)
    log_linear_skew_fair_strike = VarianceSwap.estimate_fair_strike(
        **params, skew_slope=skew_slope, linear_skew=False
    )
    assert round(log_linear_skew_fair_strike * 100, 1) == 22.8
