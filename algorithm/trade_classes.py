from uuid import uuid4
from datetime import date
import math

import numpy as np
from algorithm.stat_methods import calc_annual_realised_vol

ALLOWED_DIRECTIONS = ("buy", "sell")


class Trade:
    _trade_id = None
    _direction = None
    _underlying = None
    _trade_date = None
    _value_date = None

    def __init__(
        self,
        direction: str,
        underlying: str,
        trade_date: date,
        value_date: date,
        **kwargs,
    ) -> None:
        self._trade_id = uuid4()
        direction = direction.lower()
        if direction not in ALLOWED_DIRECTIONS:
            raise ValueError(
                f"direction must be an allowed direction {ALLOWED_DIRECTIONS}"
            )
        self._direction = direction
        self._underlying = underlying
        self._trade_date = trade_date
        self._value_date = value_date
        [setattr(self, k, v) for k, v in kwargs.items()]

    def __str__(self) -> str:
        return f"{self.direction} {self.underlying} | {self.trade_date} - {self.value_date}"

    def __repr__(self) -> str:
        return str(self)

    def payoff(self, *args, **kwargs) -> float:
        raise NotImplementedError()

    def calc_mtm(self, *args, **kwargs) -> float:
        raise NotImplementedError()

    # Implement further methods ...

    @property
    def direction(self):
        return self._direction

    @property
    def underlying(self):
        return self._underlying

    @property
    def trade_id(self):
        return self._trade_id

    @property
    def trade_date(self):
        return self._trade_date

    @property
    def value_date(self):
        return self._value_date


class VarianceSwap(Trade):
    _strike = None
    _vega_amount = None

    def __init__(
        self,
        direction: str,
        underlying: str,
        trade_date: date,
        value_date: date,
        strike: float,
        vega_amount: float = None,
        var_amount: float = None,
    ) -> None:
        if vega_amount is None and var_amount is None:
            raise ValueError("Either _vega_amount or var_amount is required")
        elif vega_amount is None:
            vega_amount = var_amount * strike * 2
        super().__init__(
            direction,
            underlying,
            trade_date,
            value_date,
            _strike=strike,
            _vega_amount=vega_amount,
        )

    def __str__(self) -> str:
        return (
            f"{self.underlying}: {self.direction} {self.vega_amount}"
            f"@{self.strike} | {self.trade_date} - {self.value_date}"
        )

    def payoff(self, realised_vol: float) -> float:
        """Calculate the payoff at trade maturity

        Args:
            realised_vol: the annualised realised volatility
                calculated from trade inception
                (see `VarianceSwap.calc_final_realised_vol`)
        """
        return self.var_amount * (realised_vol ** 2 - self.strike ** 2)

    def calc_mtm(
        self, realised_vol: float, fair_strike: float, r: float, valuation_date: date
    ) -> float:
        """Calculate the mark-to-maket.

        Args:
            realised_vol: The annualised realised volatility from trade date
                to valuation_date.
            fair_strike: The fair strike of a variance swap of same maturity
                date as current swap, issued at the same date as current swap.
            r: The annualised, continuously compounded discount rate.
            valuation_date: date at which the mtm is calculated.
        """
        T = ((self.value_date - self.trade_date).days - 1) / 365
        t = ((valuation_date - self.trade_date).days - 1) / 365
        mtm = (
            self.var_amount
            * math.exp(-r * (T - t))
            * (
                t / T * (realised_vol ** 2)
                + (T - t) / T * (fair_strike ** 2)
                - (self.strike ** 2)
            )
        )
        return mtm

    @staticmethod
    def estimate_fair_strike(
        vol_atmf: float, T: float, skew_slope: float, linear_skew: bool = True
    ) -> float:
        """Calculate the fair strike that would be traded.

        Args:
            vol_atmf: At-the-money forward volatility for
                the duration of the trade.
            T: The duration of the trade in years.
            skew_slope: The slope of the skew curve. If the
                curve is log-linear, it is the slope of the
                log skew curve.
        Kwargs:
            linear_skew (default: True): False for log-linear skew
        """
        if linear_skew:
            return vol_atmf * (1 + 3 * T * skew_slope ** 2) ** 0.5
        else:
            # NOTE: Assume the user will calculate the slope
            # numerically and provide the right input
            β = skew_slope
            return math.sqrt(
                vol_atmf ** 2
                + β * (vol_atmf ** 3) * T
                + (β / 2) ** 2
                * (12 * (vol_atmf ** 2) * T + 5 * (vol_atmf ** 4) * T ** 2)
            )

    @staticmethod
    def calc_final_realised_vol(levels: np.ndarray) -> float:
        """Calculates the final realised volatility

        Args:
            levels: an array with the daily underlying asset
                prices. It must contain all the observations
                from trade inception to trade maturity.
        """
        return calc_annual_realised_vol(levels)

    @property
    def strike(self):
        return self._strike

    @property
    def vega_amount(self):
        return self._vega_amount

    @property
    def var_amount(self):
        return self._vega_amount / (2 * self.strike)


# Implement more trade classes below ...
