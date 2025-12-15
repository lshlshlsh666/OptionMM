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
    initial_cash: float,
) -> Metrics:
    """
    Compute a small set of headline metrics from backtest logs.
    """
    if timeline.empty:
        return Metrics(
            total_pnl=float("nan"),
            total_trades=0,
            fill_rate=float("nan"),
            max_drawdown=float("nan"),
            avg_abs_position=float("nan"),
            avg_quote_spread=float("nan"),
        )

    equity = timeline.set_index("quote_datetime")["equity"].astype(float)
    total_pnl = float(equity.iloc[-1] - initial_cash)
    total_trades = int(len(trades)) if trades is not None else 0

    total_quotes = int(len(quotes)) if quotes is not None else 0
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

    return Metrics(
        total_pnl=total_pnl,
        total_trades=total_trades,
        fill_rate=fill_rate,
        max_drawdown=max_dd,
        avg_abs_position=avg_abs_pos,
        avg_quote_spread=avg_q_spread,
    )
