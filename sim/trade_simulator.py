from __future__ import annotations

from dataclasses import dataclass

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillModelConfig:
    """
    Probabilistic fill model for market making simulation.

    Aggressiveness is measured relative to the current market spread:
    - For bids: 0 at market bid, 1 at market ask (crossing)
    - For asks: 0 at market ask, 1 at market bid (crossing)
    """

    p_base: float = (
        0.02  # fill probability when joining best bid/ask (aggressiveness=0)
    )
    p_max: float = (
        0.40  # fill probability when quoting at the opposite side (aggressiveness=1)
    )
    gamma: float = (
        1.5  # curvature: >1 makes probability ramp up faster when more aggressive
    )
    size_decay: float = (
        0.01  # negative correlation: larger queue size ahead => lower fill prob
    )
    trade_size: int = 1  # if hit, fill exactly 1 contract


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _fill_prob_from_aggressiveness(aggr: float, cfg: FillModelConfig) -> float:
    aggr01 = _clip01(aggr)
    # Ensure monotonic in aggressiveness and within [0, 1].
    p = cfg.p_base + (cfg.p_max - cfg.p_base) * (aggr01**cfg.gamma)
    return _clip01(p)


def simulate_fills(
    *,
    ts: pd.Timestamp,
    market: pd.DataFrame,
    quotes: pd.DataFrame,
    rng: np.random.Generator,
    cfg: FillModelConfig,
    max_fills_per_timestamp: int | None = None,
) -> pd.DataFrame:
    """
    Simulate fills for one timestamp.

    Rules:
    1) If our bid crosses market ask -> immediate buy fill at market ask (taker).
    2) If our ask crosses market bid -> immediate sell fill at market bid (taker).
    3) Otherwise, compute aggressiveness vs market quote and use it to produce a fill probability.
       If hit, fill exactly 1 contract at our quote price.

    Inputs:
    - market: indexed by (strike, option_type) with columns: bid, ask
      Optional size columns (best level): bid_size / ask_size (or similar)
    - quotes: indexed by (strike, option_type) with columns: my_bid, my_ask
    """
    required_mkt_cols = {"bid", "ask"}
    required_quote_cols = {"my_bid", "my_ask"}
    missing_mkt = required_mkt_cols - set(market.columns)
    missing_q = required_quote_cols - set(quotes.columns)
    if missing_mkt:
        raise ValueError(f"market is missing columns: {sorted(missing_mkt)}")
    if missing_q:
        raise ValueError(f"quotes is missing columns: {sorted(missing_q)}")

    # Optional size columns: if present, we'll use them to penalize fills when we're not more aggressive.
    bid_size_col = next(
        (
            c
            for c in ("bid_size", "bid_sz", "bid_qty", "bid_quantity")
            if c in market.columns
        ),
        None,
    )
    ask_size_col = next(
        (
            c
            for c in ("ask_size", "ask_sz", "ask_qty", "ask_quantity")
            if c in market.columns
        ),
        None,
    )

    mkt_cols = ["bid", "ask"]
    if bid_size_col is not None:
        mkt_cols.append(bid_size_col)
    if ask_size_col is not None:
        mkt_cols.append(ask_size_col)

    joined = market[mkt_cols].join(quotes[["my_bid", "my_ask"]], how="inner")
    out_columns = [
        "quote_datetime",
        "strike",
        "option_type",
        "side",
        "qty",
        "fill_price",
        "fill_type",
        "p_buy",
        "p_sell",
        "u",
        "queue_ahead_buy",
        "queue_ahead_sell",
        "market_bid",
        "market_ask",
        "my_bid",
        "my_ask",
    ]
    if joined.empty:
        return pd.DataFrame(columns=out_columns)

    fills: list[dict] = []
    for (strike, option_type), row in joined.iterrows():
        mkt_bid = float(row["bid"])
        mkt_ask = float(row["ask"])
        my_bid = float(row["my_bid"])
        my_ask = float(row["my_ask"])
        mkt_bid_size = (
            float(row[bid_size_col])
            if bid_size_col is not None and bid_size_col in row
            else 0.0
        )
        mkt_ask_size = (
            float(row[ask_size_col])
            if ask_size_col is not None and ask_size_col in row
            else 0.0
        )

        if (
            not np.isfinite(mkt_bid)
            or not np.isfinite(mkt_ask)
            or not np.isfinite(my_bid)
            or not np.isfinite(my_ask)
        ):
            continue
        if not np.isfinite(mkt_bid_size):
            mkt_bid_size = 0.0
        if not np.isfinite(mkt_ask_size):
            mkt_ask_size = 0.0
        mkt_bid_size = max(mkt_bid_size, 0.0)
        mkt_ask_size = max(mkt_ask_size, 0.0)
        if mkt_bid <= 0 or mkt_ask <= 0 or mkt_ask < mkt_bid:
            continue
        if my_bid <= 0 or my_ask <= 0 or my_ask < my_bid:
            continue

        # Crossing (taker) logic first.
        if my_bid >= mkt_ask:
            logger.debug(
                "CROSS buy t=%s %s %s | my_bid=%.4f >= mkt_ask=%.4f",
                str(ts),
                str(strike),
                str(option_type),
                my_bid,
                mkt_ask,
            )
            fills.append(
                {
                    "quote_datetime": ts,
                    "strike": strike,
                    "option_type": option_type,
                    "side": "buy",
                    "qty": int(cfg.trade_size),
                    "fill_price": mkt_ask,
                    "fill_type": "cross",
                    "p_buy": 1.0,
                    "p_sell": 0.0,
                    "u": 0.0,
                    "queue_ahead_buy": 0.0,
                    "queue_ahead_sell": 0.0,
                    "market_bid": mkt_bid,
                    "market_ask": mkt_ask,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                }
            )
            continue

        if my_ask <= mkt_bid:
            logger.debug(
                "CROSS sell t=%s %s %s | my_ask=%.4f <= mkt_bid=%.4f",
                str(ts),
                str(strike),
                str(option_type),
                my_ask,
                mkt_bid,
            )
            fills.append(
                {
                    "quote_datetime": ts,
                    "strike": strike,
                    "option_type": option_type,
                    "side": "sell",
                    "qty": -int(cfg.trade_size),
                    "fill_price": mkt_bid,
                    "fill_type": "cross",
                    "p_buy": 0.0,
                    "p_sell": 1.0,
                    "u": 0.0,
                    "queue_ahead_buy": 0.0,
                    "queue_ahead_sell": 0.0,
                    "market_bid": mkt_bid,
                    "market_ask": mkt_ask,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                }
            )
            continue

        spread = mkt_ask - mkt_bid
        if spread <= 0:
            continue

        # Aggressiveness:
        # - More aggressive than market: use aggressiveness-only probability (no size penalty)
        # - Otherwise (join or worse): penalize by the queue size ahead (negative correlation)
        bid_aggr = 0.0 if my_bid <= mkt_bid else (my_bid - mkt_bid) / spread
        ask_aggr = 0.0 if my_ask >= mkt_ask else (mkt_ask - my_ask) / spread

        queue_ahead_buy = mkt_bid_size if my_bid <= mkt_bid else 0.0
        queue_ahead_sell = mkt_ask_size if my_ask >= mkt_ask else 0.0

        if bid_aggr > 0:
            p_buy = _fill_prob_from_aggressiveness(bid_aggr, cfg)
        else:
            p_buy = cfg.p_base * float(np.exp(-cfg.size_decay * queue_ahead_buy))

        if ask_aggr > 0:
            p_sell = _fill_prob_from_aggressiveness(ask_aggr, cfg)
        else:
            p_sell = cfg.p_base * float(np.exp(-cfg.size_decay * queue_ahead_sell))

        # Ensure probabilities are valid.
        p_buy = _clip01(p_buy)
        p_sell = _clip01(p_sell)
        total = p_buy + p_sell
        if total > 1.0:
            p_buy /= total
            p_sell /= total

        # At most one fill per contract per timestamp: sample side using a single u.
        u = float(rng.random())
        if u < p_buy:
            logger.debug(
                "FILL buy t=%s %s %s | p_buy=%.4f p_sell=%.4f u=%.4f | q_ahead_buy=%.2f",
                str(ts),
                str(strike),
                str(option_type),
                p_buy,
                p_sell,
                u,
                queue_ahead_buy,
            )
            fills.append(
                {
                    "quote_datetime": ts,
                    "strike": strike,
                    "option_type": option_type,
                    "side": "buy",
                    "qty": int(cfg.trade_size),
                    "fill_price": my_bid,  # we get hit at our bid
                    "fill_type": "prob",
                    "p_buy": p_buy,
                    "p_sell": p_sell,
                    "u": u,
                    "queue_ahead_buy": queue_ahead_buy,
                    "queue_ahead_sell": queue_ahead_sell,
                    "market_bid": mkt_bid,
                    "market_ask": mkt_ask,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                }
            )
        elif u < p_buy + p_sell:
            logger.debug(
                "FILL sell t=%s %s %s | p_buy=%.4f p_sell=%.4f u=%.4f | q_ahead_sell=%.2f",
                str(ts),
                str(strike),
                str(option_type),
                p_buy,
                p_sell,
                u,
                queue_ahead_sell,
            )
            fills.append(
                {
                    "quote_datetime": ts,
                    "strike": strike,
                    "option_type": option_type,
                    "side": "sell",
                    "qty": -int(cfg.trade_size),
                    "fill_price": my_ask,  # we get lifted at our ask
                    "fill_type": "prob",
                    "p_buy": p_buy,
                    "p_sell": p_sell,
                    "u": u,
                    "queue_ahead_buy": queue_ahead_buy,
                    "queue_ahead_sell": queue_ahead_sell,
                    "market_bid": mkt_bid,
                    "market_ask": mkt_ask,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                }
            )

    fills_df = pd.DataFrame(fills)
    if fills_df.empty:
        return pd.DataFrame(columns=out_columns)

    if max_fills_per_timestamp is not None and max_fills_per_timestamp >= 0:
        fills_df = fills_df.head(int(max_fills_per_timestamp)).copy()

    return fills_df
