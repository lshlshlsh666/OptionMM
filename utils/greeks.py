"""
Greeks calculation module for options risk management.

Provides delta, gamma calculations for Black-Scholes and American options.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.pricing import BinomialTree


__all__ = ["delta_bs", "gamma_bs", "delta_american_put", "compute_portfolio_delta"]


def delta_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    dividend: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes delta for European options.

    For American calls without dividends, BS delta is appropriate since
    early exercise is never optimal.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        dividend: Continuous dividend yield (default 0)

    Returns:
        Delta value
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        # At expiration or invalid inputs
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r - dividend + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return float(np.exp(-dividend * T) * norm.cdf(d1))
    else:
        return float(np.exp(-dividend * T) * (norm.cdf(d1) - 1))


def gamma_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    dividend: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes gamma (same for calls and puts).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        dividend: Continuous dividend yield (default 0)

    Returns:
        Gamma value
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - dividend + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    return float(np.exp(-dividend * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def delta_american_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 100,
    bump: float = 0.01,
) -> float:
    """
    Calculate delta for American put using finite difference on binomial tree.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        n_steps: Number of steps in binomial tree
        bump: Relative bump size for finite difference

    Returns:
        Delta value (negative for puts)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return -1.0 if S < K else 0.0

    # Price at S + dS
    bt_up = BinomialTree(
        S0=S * (1 + bump),
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        dividend=0,
        n_steps=n_steps,
        option_type="put",
        exercise_type="american",
    )
    price_up = bt_up.price()

    # Price at S - dS
    bt_down = BinomialTree(
        S0=S * (1 - bump),
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        dividend=0,
        n_steps=n_steps,
        option_type="put",
        exercise_type="american",
    )
    price_down = bt_down.price()

    # Central difference
    dS = 2 * S * bump
    return float((price_up - price_down) / dS)


def compute_option_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type_str: str,
    n_steps: int = 100,
) -> float:
    """
    Compute delta for a single option, choosing appropriate method.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type_str: 'C' for call, 'P' for put
        n_steps: Binomial tree steps for American puts

    Returns:
        Delta value
    """
    if pd.isna(sigma) or sigma <= 0:
        return 0.0

    if option_type_str == "C":
        # American calls without dividends: use BS delta
        return delta_bs(S, K, T, r, sigma, "call")
    else:
        # American puts: use numerical delta
        return delta_american_put(S, K, T, r, sigma, n_steps)


def compute_portfolio_delta(
    positions: dict[tuple[float, str], int],
    market_data: pd.DataFrame,
    expiration_date: pd.Timestamp,
    current_ts: pd.Timestamp,
    n_steps: int = 50,
) -> float:
    """
    Compute total portfolio delta from option positions.

    Args:
        positions: Dict mapping (strike, option_type) -> quantity
        market_data: DataFrame with columns: underlying_bid, underlying_ask, iv, r
                    Indexed by (strike, option_type)
        expiration_date: Option expiration date
        current_ts: Current timestamp
        n_steps: Binomial tree steps for American puts

    Returns:
        Total portfolio delta (sum of position * option_delta)
    """
    if not positions:
        return 0.0

    T = (expiration_date - current_ts).total_seconds() / (365 * 24 * 60 * 60)
    if T <= 0:
        return 0.0

    total_delta = 0.0

    for (strike, option_type), qty in positions.items():
        if qty == 0:
            continue

        try:
            row = market_data.loc[(strike, option_type)]
            S = (row["underlying_bid"] + row["underlying_ask"]) / 2
            sigma = row["iv"] if "iv" in row else 0.2
            r = row["r"] if "r" in row else 0.05

            if pd.isna(S) or S <= 0:
                continue

            option_delta = compute_option_delta(S, strike, T, r, sigma, option_type, n_steps)
            total_delta += qty * option_delta

        except (KeyError, IndexError):
            continue

    return total_delta
