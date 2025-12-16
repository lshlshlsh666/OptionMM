from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np
import pandas as pd

from sim.metrics import Metrics, compute_metrics
from dataclasses import field

from sim.strategy import BaseStrategy, SimpleImproveInsideStrategy
from sim.trade_simulator import FillModelConfig, simulate_fills
from sim.hedger import HedgerConfig, DeltaHedger
from utils.greeks import compute_portfolio_delta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 0.0
    seed: int = 7

    strategy: BaseStrategy = field(default_factory=SimpleImproveInsideStrategy)
    fills: FillModelConfig = FillModelConfig()
    hedger: HedgerConfig = HedgerConfig()

    max_fills_per_timestamp: int | None = None
    log_every: int = 50  # log progress every N timestamps (0/negative disables)


@dataclass(frozen=True)
class BacktestResult:
    quotes: pd.DataFrame
    trades: pd.DataFrame
    hedge_trades: pd.DataFrame
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


def run_backtest(
    snapshot: pd.DataFrame,
    *,
    cfg: BacktestConfig,
    expiration_date: pd.Timestamp | None = None,
) -> BacktestResult:
    """
    Sequential backtester with delta hedging support.

    Input:
    - snapshot: multi-indexed by (quote_datetime, strike, option_type), with columns bid/ask (and optionally is_atm).
    - expiration_date: Option expiration date (required if hedging is enabled)
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

    if cfg.hedger.enabled and expiration_date is None:
        raise ValueError("expiration_date is required when hedging is enabled")

    rng = np.random.default_rng(cfg.seed)

    timestamps = (
        snapshot.index.get_level_values("quote_datetime").unique().sort_values()
    )
    positions: dict[tuple[float, str], int] = {}
    cash = float(cfg.initial_cash)

    # Initialize hedger
    hedger = DeltaHedger(cfg=cfg.hedger)

    quote_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    timeline_rows: list[dict] = []

    logger.info(
        "Backtest start: timestamps=%d, initial_cash=%.2f, seed=%d, max_fills_per_timestamp=%s, hedging=%s",
        len(timestamps),
        cfg.initial_cash,
        cfg.seed,
        cfg.max_fills_per_timestamp,
        cfg.hedger.enabled,
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

        # Delta hedging
        portfolio_delta = 0.0
        hedge_trade = None
        if cfg.hedger.enabled and expiration_date is not None:
            portfolio_delta = compute_portfolio_delta(
                positions=positions,
                market_data=market_at_ts,
                expiration_date=expiration_date,
                current_ts=ts,
                n_steps=50,
            )
            # Get underlying mid price for hedging
            if {"underlying_bid", "underlying_ask"}.issubset(market_at_ts.columns):
                underlying_mid = float(
                    (market_at_ts["underlying_bid"].iloc[0] + market_at_ts["underlying_ask"].iloc[0]) / 2
                )
                hedge_trade = hedger.execute_hedge(ts, portfolio_delta, underlying_mid)
                if hedge_trade is not None:
                    # Update cash for hedge trade
                    cash -= hedge_trade.qty * hedge_trade.price

        # Mark-to-market
        mv = _mark_to_market_value(positions, market_at_ts)
        # Include underlying position in equity
        underlying_mtm = 0.0
        if cfg.hedger.enabled and {"underlying_bid", "underlying_ask"}.issubset(market_at_ts.columns):
            underlying_mid = float(
                (market_at_ts["underlying_bid"].iloc[0] + market_at_ts["underlying_ask"].iloc[0]) / 2
            )
            underlying_mtm = hedger.get_underlying_mtm(underlying_mid)
        total_capital = cash + mv + underlying_mtm

        net_pos = int(sum(positions.values())) if positions else 0
        trades_count = int(len(fills)) if fills is not None else 0
        contracts_quoted = int(len(quotes_at_ts))
        total_delta = portfolio_delta + hedger.underlying_position

        timeline_rows.append(
            {
                "quote_datetime": ts,
                "cash": cash,
                "mtm_value": mv,
                "underlying_mtm": underlying_mtm,
                "equity": total_capital,
                "net_position": net_pos,
                "trades": trades_count,
                "contracts_quoted": contracts_quoted,
                "portfolio_delta": portfolio_delta,
                "underlying_position": hedger.underlying_position,
                "total_delta": total_delta,
            }
        )

        if cfg.log_every > 0 and (len(timeline_rows) % cfg.log_every == 0):
            logger.info(
                "t=%s | quoted=%d | trades=%d | cash=%.2f | mtm=%.2f | equity=%.2f | net_pos=%d | delta=%.2f",
                str(ts),
                contracts_quoted,
                trades_count,
                cash,
                mv,
                total_capital,
                net_pos,
                total_delta,
            )

    quotes_df = (
        pd.concat(quote_rows, ignore_index=True) if quote_rows else pd.DataFrame()
    )
    trades_df = (
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    )
    hedge_trades_df = hedger.get_hedge_trades_df()
    timeline_df = pd.DataFrame(timeline_rows)

    metrics = compute_metrics(
        timeline=timeline_df,
        quotes=quotes_df,
        trades=trades_df,
        hedge_trades=hedge_trades_df,
        initial_cash=cfg.initial_cash,
    )

    logger.info(
        "Backtest done: total_pnl=%.4f, total_trades=%d, fill_rate=%.4f, max_drawdown=%.4f, hedge_trades=%d",
        metrics.total_pnl,
        metrics.total_trades,
        metrics.fill_rate,
        metrics.max_drawdown,
        metrics.total_hedge_trades,
    )

    return BacktestResult(
        quotes=quotes_df,
        trades=trades_df,
        hedge_trades=hedge_trades_df,
        timeline=timeline_df,
        metrics=metrics,
    )
