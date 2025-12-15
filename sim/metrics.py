from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    total_pnl: float
    total_trades: int
    fill_rate: float
    max_drawdown: float
    avg_abs_position: float
    avg_quote_spread: float
    # Hedging metrics
    total_hedge_trades: int
    total_hedge_cost: float
    max_delta_exposure: float
    avg_delta_exposure: float


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return float(dd.min())


def compute_metrics(
    *,
    timeline: pd.DataFrame,
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    hedge_trades: pd.DataFrame | None = None,
    initial_cash: float,
) -> Metrics:
    """
    Compute a small set of headline metrics from backtest logs.

    Args:
        timeline: DataFrame with columns: quote_datetime, equity, net_position, total_delta, etc.
        quotes: DataFrame with columns: my_bid, my_ask
        trades: DataFrame of option trades
        hedge_trades: DataFrame of hedge trades (underlying)
        initial_cash: Initial cash amount
    """
    if timeline.empty:
        return Metrics(
            total_pnl=float("nan"),
            total_trades=0,
            fill_rate=float("nan"),
            max_drawdown=float("nan"),
            avg_abs_position=float("nan"),
            avg_quote_spread=float("nan"),
            total_hedge_trades=0,
            total_hedge_cost=float("nan"),
            max_delta_exposure=float("nan"),
            avg_delta_exposure=float("nan"),
        )

    equity = timeline.set_index("quote_datetime")["equity"].astype(float)
    total_pnl = float(equity.iloc[-1] - initial_cash)
    total_trades = int(len(trades)) if trades is not None and not trades.empty else 0

    total_quotes = int(len(quotes)) if quotes is not None and not quotes.empty else 0
    fill_rate = float(total_trades / total_quotes) if total_quotes > 0 else float("nan")

    max_dd = _max_drawdown(equity)
    avg_abs_pos = (
        float(np.mean(np.abs(timeline["net_position"].astype(float))))
        if "net_position" in timeline
        else float("nan")
    )
    avg_q_spread = (
        float(np.mean((quotes["my_ask"] - quotes["my_bid"]).astype(float)))
        if total_quotes > 0
        else float("nan")
    )

    # Hedging metrics
    total_hedge_trades = (
        int(len(hedge_trades)) if hedge_trades is not None and not hedge_trades.empty else 0
    )

    # Estimate hedge cost as the spread paid on hedge trades
    total_hedge_cost = 0.0
    if hedge_trades is not None and not hedge_trades.empty and "qty" in hedge_trades.columns:
        # Approximate cost as 0.01% of notional (can be refined)
        total_hedge_cost = float(
            (hedge_trades["qty"].abs() * hedge_trades["price"] * 0.0001).sum()
        )

    # Delta exposure from timeline
    max_delta_exposure = float("nan")
    avg_delta_exposure = float("nan")
    if "total_delta" in timeline.columns:
        delta_series = timeline["total_delta"].astype(float)
        max_delta_exposure = float(delta_series.abs().max())
        avg_delta_exposure = float(delta_series.abs().mean())

    return Metrics(
        total_pnl=total_pnl,
        total_trades=total_trades,
        fill_rate=fill_rate,
        max_drawdown=max_dd,
        avg_abs_position=avg_abs_pos,
        avg_quote_spread=avg_q_spread,
        total_hedge_trades=total_hedge_trades,
        total_hedge_cost=total_hedge_cost,
        max_delta_exposure=max_delta_exposure,
        avg_delta_exposure=avg_delta_exposure,
    )
