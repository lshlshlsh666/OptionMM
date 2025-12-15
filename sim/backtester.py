from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np
import pandas as pd

from sim.metrics import Metrics, compute_metrics
from dataclasses import field

from sim.strategy import BaseStrategy, SimpleImproveInsideStrategy
from sim.trade_simulator import FillModelConfig, simulate_fills

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 0.0
    seed: int = 7

    strategy: BaseStrategy = field(default_factory=SimpleImproveInsideStrategy)
    fills: FillModelConfig = FillModelConfig()

    max_fills_per_timestamp: int | None = None
    log_every: int = 50  # log progress every N timestamps (0/negative disables)


@dataclass(frozen=True)
class BacktestResult:
    quotes: pd.DataFrame
    trades: pd.DataFrame
    timeline: pd.DataFrame
    metrics: Metrics


def _mark_to_market_value(
    positions: dict[tuple[float, str], int], market_at_ts: pd.DataFrame
) -> float:
    if not positions:
        return 0.0
    if market_at_ts.empty:
        return 0.0

    mv = 0.0
    for (strike, option_type), qty in positions.items():
        try:
            row = market_at_ts.loc[(strike, option_type)]
        except KeyError:
            continue
        mid = float((row["bid"] + row["ask"]) / 2)
        if np.isfinite(mid):
            mv += float(qty) * mid
    return mv


def run_backtest(snapshot: pd.DataFrame, *, cfg: BacktestConfig) -> BacktestResult:
    """
    Sequential backtester.

    Input:
    - snapshot: multi-indexed by (quote_datetime, strike, option_type), with columns bid/ask (and optionally is_atm).
    """
    if "quote_datetime" not in snapshot.index.names:
        raise ValueError("snapshot index must include level: quote_datetime")
    if (
        "strike" not in snapshot.index.names
        or "option_type" not in snapshot.index.names
    ):
        raise ValueError("snapshot index must include levels: strike, option_type")
    if not {"bid", "ask"}.issubset(snapshot.columns):
        raise ValueError("snapshot must have columns: bid, ask")

    rng = np.random.default_rng(cfg.seed)

    timestamps = (
        snapshot.index.get_level_values("quote_datetime").unique().sort_values()
    )
    positions: dict[tuple[float, str], int] = {}
    cash = float(cfg.initial_cash)

    quote_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    timeline_rows: list[dict] = []

    logger.info(
        "Backtest start: timestamps=%d, initial_cash=%.2f, seed=%d, max_fills_per_timestamp=%s",
        len(timestamps),
        cfg.initial_cash,
        cfg.seed,
        cfg.max_fills_per_timestamp,
    )

    for ts in timestamps:
        market_at_ts = snapshot.xs(ts, level="quote_datetime").copy()
        market_at_ts.index = market_at_ts.index.set_names(["strike", "option_type"])

        universe_idx = cfg.strategy.select_universe(market_at_ts)
        predicted_at_ts = (
            market_at_ts[["predicted_price"]].copy()
            if "predicted_price" in market_at_ts.columns
            else pd.DataFrame(index=market_at_ts.index)
        )
        quotes_at_ts = cfg.strategy.generate_quotes(
            market_at_ts, universe_idx, positions, predicted_at_ts
        )
        quote_log_at_ts = quotes_at_ts.copy()
        quote_log_at_ts["quote_datetime"] = ts
        quote_rows.append(quote_log_at_ts.reset_index())

        fills = simulate_fills(
            ts=ts,
            market=market_at_ts,
            quotes=quotes_at_ts[["my_bid", "my_ask"]],
            rng=rng,
            cfg=cfg.fills,
            max_fills_per_timestamp=cfg.max_fills_per_timestamp,
        )

        if fills is not None and not fills.empty:
            # Update positions and cash.
            for _, tr in fills.iterrows():
                key = (float(tr["strike"]), str(tr["option_type"]))
                qty = int(tr["qty"])
                price = float(tr["fill_price"])
                positions[key] = positions.get(key, 0) + qty
                cash -= (
                    qty * price
                )  # buy reduces cash; sell (qty negative) increases cash

            trade_rows.append(fills)

        mv = _mark_to_market_value(positions, market_at_ts)
        total_capital = cash + mv
        net_pos = int(sum(positions.values())) if positions else 0
        trades_count = int(len(fills)) if fills is not None else 0
        contracts_quoted = int(len(quotes_at_ts))

        timeline_rows.append(
            {
                "quote_datetime": ts,
                "cash": cash,
                "mtm_value": mv,
                "equity": total_capital,
                "net_position": net_pos,
                "trades": trades_count,
                "contracts_quoted": contracts_quoted,
            }
        )

        if cfg.log_every > 0 and (len(timeline_rows) % cfg.log_every == 0):
            logger.info(
                "t=%s | quoted=%d | trades=%d | cash=%.2f | mtm=%.2f | total_capital=%.2f | net_pos=%d",
                str(ts),
                contracts_quoted,
                trades_count,
                cash,
                mv,
                total_capital,
                net_pos,
            )

    quotes_df = (
        pd.concat(quote_rows, ignore_index=True) if quote_rows else pd.DataFrame()
    )
    trades_df = (
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    )
    timeline_df = pd.DataFrame(timeline_rows)

    metrics = compute_metrics(
        timeline=timeline_df,
        quotes=quotes_df,
        trades=trades_df,
        initial_cash=cfg.initial_cash,
    )

    logger.info(
        "Backtest done: total_pnl=%.4f, total_trades=%d, fill_rate=%.4f, max_drawdown=%.4f",
        metrics.total_pnl,
        metrics.total_trades,
        metrics.fill_rate,
        metrics.max_drawdown,
    )

    return BacktestResult(
        quotes=quotes_df, trades=trades_df, timeline=timeline_df, metrics=metrics
    )
