"""
Delta hedging module for options market making.

Implements threshold-based delta hedging policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


__all__ = ["HedgerConfig", "DeltaHedger", "HedgeTrade"]


@dataclass(frozen=True)
class HedgerConfig:
    """
    Configuration for delta hedging.

    Attributes:
        enabled: Whether hedging is active
        delta_threshold: Hedge when |portfolio_delta| > threshold
        hedge_spread: Spread paid on underlying hedge trades (as fraction of mid)
    """

    enabled: bool = True
    delta_threshold: float = 10.0
    hedge_spread: float = 0.0001  # 1 bps spread on underlying


@dataclass
class HedgeTrade:
    """Record of a single hedge trade."""

    quote_datetime: pd.Timestamp
    qty: int  # Positive = buy underlying, negative = sell
    price: float
    portfolio_delta_before: float
    portfolio_delta_after: float


@dataclass
class DeltaHedger:
    """
    Threshold-based delta hedger.

    Hedges to delta-neutral only when |portfolio_delta| exceeds the threshold.
    """

    cfg: HedgerConfig
    underlying_position: int = field(default=0, init=False)
    hedge_trades: list[HedgeTrade] = field(default_factory=list, init=False)

    def should_hedge(self, portfolio_delta: float) -> bool:
        """Check if hedging is needed based on threshold."""
        if not self.cfg.enabled:
            return False
        # Total delta includes option delta + underlying position
        total_delta = portfolio_delta + self.underlying_position
        return abs(total_delta) > self.cfg.delta_threshold

    def compute_hedge_quantity(self, portfolio_delta: float) -> int:
        """
        Compute number of underlying shares to trade to achieve delta-neutral.

        Args:
            portfolio_delta: Current delta from options portfolio

        Returns:
            Number of shares to trade (positive = buy, negative = sell)
        """
        # Target: underlying_position = -portfolio_delta (to offset delta)
        target_underlying = -portfolio_delta
        hedge_qty = int(round(target_underlying - self.underlying_position))
        return hedge_qty

    def execute_hedge(
        self,
        ts: pd.Timestamp,
        portfolio_delta: float,
        underlying_mid: float,
    ) -> HedgeTrade | None:
        """
        Execute hedge trade if needed.

        Args:
            ts: Current timestamp
            portfolio_delta: Current delta from options
            underlying_mid: Mid price of underlying

        Returns:
            HedgeTrade if executed, None otherwise
        """
        if not self.should_hedge(portfolio_delta):
            return None

        hedge_qty = self.compute_hedge_quantity(portfolio_delta)
        if hedge_qty == 0:
            return None

        # Apply spread: pay more when buying, receive less when selling
        if hedge_qty > 0:
            price = underlying_mid * (1 + self.cfg.hedge_spread / 2)
        else:
            price = underlying_mid * (1 - self.cfg.hedge_spread / 2)

        delta_before = portfolio_delta + self.underlying_position
        self.underlying_position += hedge_qty
        delta_after = portfolio_delta + self.underlying_position

        trade = HedgeTrade(
            quote_datetime=ts,
            qty=hedge_qty,
            price=price,
            portfolio_delta_before=delta_before,
            portfolio_delta_after=delta_after,
        )
        self.hedge_trades.append(trade)
        return trade

    def get_underlying_mtm(self, underlying_mid: float) -> float:
        """Get mark-to-market value of underlying position."""
        return self.underlying_position * underlying_mid

    def get_hedge_trades_df(self) -> pd.DataFrame:
        """Return hedge trades as DataFrame."""
        if not self.hedge_trades:
            return pd.DataFrame(
                columns=[
                    "quote_datetime",
                    "qty",
                    "price",
                    "portfolio_delta_before",
                    "portfolio_delta_after",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "quote_datetime": t.quote_datetime,
                    "qty": t.qty,
                    "price": t.price,
                    "portfolio_delta_before": t.portfolio_delta_before,
                    "portfolio_delta_after": t.portfolio_delta_after,
                }
                for t in self.hedge_trades
            ]
        )

    def reset(self) -> None:
        """Reset hedger state."""
        self.underlying_position = 0
        self.hedge_trades = []
