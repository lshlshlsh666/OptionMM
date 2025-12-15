from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """
    Strategy interface.

    A strategy should:
    - Select which contracts it wants to quote at a timestamp (universe)
    - Generate quotes for those contracts (my_bid / my_ask)
    """

    @abstractmethod
    def select_universe(
        self, market_at_ts: pd.DataFrame
    ) -> pd.Index:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def generate_quotes(  # pragma: no cover
        self,
        market_at_ts: pd.DataFrame,
        universe: pd.Index,
        positions: dict[tuple[float, str], int],
        predicted_at_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleImproveInsideStrategy(BaseStrategy):
    """
    Minimal baseline strategy:
    - Universe: N strikes around ATM, for selected option_types
    - Quotes: improve inside the market spread by a fixed fraction

    Notes:
    - This class is intentionally simple and meant as a template for more complex strategies.
    - market_at_ts can contain additional columns (e.g., iv/predicted_price/greeks) for richer strategies.
    """

    n_strikes: int = 3
    option_types: tuple[str, ...] = ("C", "P")
    improvement_frac: float = 0.25  # 0 => join market; 0.5 => midpoint
    min_half_spread: float = 0.01  # absolute minimum half-spread in price units

    def select_universe(self, market_at_ts: pd.DataFrame) -> pd.Index:
        if market_at_ts.empty:
            return market_at_ts.index

        # Determine ATM strike. Prefer the 'is_atm' flag if present.
        atm_strike: float | None = None
        if "is_atm" in market_at_ts.columns:
            atm_rows = market_at_ts[market_at_ts["is_atm"].astype(bool)]
            if not atm_rows.empty:
                try:
                    atm_strike = float(atm_rows.index.get_level_values("strike")[0])
                except Exception:
                    atm_strike = None

        if atm_strike is None:
            # Fallback: choose strike closest to underlying mid if available.
            if {"underlying_bid", "underlying_ask"}.issubset(market_at_ts.columns):
                s_mid = float(
                    (
                        market_at_ts["underlying_bid"].iloc[0]
                        + market_at_ts["underlying_ask"].iloc[0]
                    )
                    / 2
                )
                strikes = (
                    market_at_ts.index.get_level_values("strike").unique().astype(float)
                )
                atm_strike = float(strikes[np.argmin(np.abs(strikes - s_mid))])
            else:
                strikes = (
                    market_at_ts.index.get_level_values("strike").unique().astype(float)
                )
                atm_strike = float(np.median(strikes))

        strikes_all = (
            market_at_ts.index.get_level_values("strike").unique().astype(float)
        )
        nearest = strikes_all[np.argsort(np.abs(strikes_all - atm_strike))][
            : max(int(self.n_strikes), 1)
        ]
        nearest_set = set(float(x) for x in nearest)

        mask = (
            market_at_ts.index.get_level_values("strike")
            .astype(float)
            .map(lambda k: k in nearest_set)
        )
        idx = market_at_ts.index[mask]

        if "option_type" in market_at_ts.index.names:
            idx = idx[idx.get_level_values("option_type").isin(self.option_types)]

        return idx

    def generate_quotes(
        self,
        market_at_ts: pd.DataFrame,
        universe: pd.Index,
        positions: dict[tuple[float, str], int],
        predicted_at_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        # This baseline strategy currently ignores positions and predicted prices,
        # but the signature supports more complex strategies.
        if universe.empty:
            return pd.DataFrame(columns=["my_bid", "my_ask"])

        mkt = market_at_ts.loc[universe, ["bid", "ask"]].copy()
        spread = (mkt["ask"] - mkt["bid"]).clip(lower=0.0)

        my_bid = mkt["bid"] + self.improvement_frac * spread
        my_ask = mkt["ask"] - self.improvement_frac * spread

        mid = (mkt["bid"] + mkt["ask"]) / 2
        half = np.maximum((my_ask - my_bid) / 2, self.min_half_spread)
        my_bid = (mid - half).clip(lower=0.0)
        my_ask = (mid + half).clip(lower=0.0)

        out = pd.DataFrame({"my_bid": my_bid, "my_ask": my_ask})
        out.index = mkt.index
        return out
