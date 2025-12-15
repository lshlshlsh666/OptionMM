from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

from config import Config
from utils.pricing import BinomialTree, BlackScholes

CONFIG = Config()

__all__ = ["estimate_iv", "estimate_rate"]


def _solve_single_iv(
    row_data: dict,
    idx: tuple,
    expiration_date: pd.Timestamp,
    rate: float,
    binomial_tree_steps: int,
    initial_guess: float = 0.2,
) -> tuple:
    """
    Solve IV for a single option contract (used for multiprocessing).

    Args:
        row_data: Dictionary containing row data (bid, ask, underlying_bid, underlying_ask)
        idx: Index tuple (quote_datetime, strike, option_type)
        expiration_date: Expiration date
        rate: Risk-free rate
        binomial_tree_steps: Number of steps in binomial tree
        initial_guess: Initial guess for IV

    Returns:
        Tuple of (idx, iv_value)
    """
    market_price = (row_data['bid'] + row_data['ask']) / 2

    # Skip if market price is invalid
    if market_price <= 0 or np.isnan(market_price):
        return (idx, np.nan)

    S0 = (row_data['underlying_bid'] + row_data['underlying_ask']) / 2
    quote_datetime, K, option_type_str = idx
    option_type = 'call' if option_type_str == 'C' else 'put'

    T = (expiration_date - pd.Timestamp(quote_datetime)).total_seconds() / (365 * 24 * 60 * 60)

    # Skip if time to expiration is invalid
    if T <= 0 or np.isnan(T) or not np.isfinite(T):
        return (idx, np.nan)

    def objective(sigma):
        """Objective function for minimization: (price(sigma) - market_price)^2."""
        try:
            sigma_val = sigma[0] if isinstance(sigma, (list, np.ndarray)) else sigma
            if sigma_val <= 0 or not np.isfinite(sigma_val):
                return np.inf

            if option_type == "put":
                bt = BinomialTree(
                    S0=S0,
                    K=K,
                    T=T,
                    r=rate,
                    sigma=sigma_val,
                    dividend=0,
                    n_steps=binomial_tree_steps,
                    option_type=option_type,
                    exercise_type="american",
                )
                px = bt.price()
            else:
                bs = BlackScholes(S0, K, T, rate, sigma_val, 0, option_type)
                px = bs.price()

            if not np.isfinite(px) or px < 0:
                return np.inf
            return (px - market_price) ** 2
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            return np.inf

    # Optimize using L-BFGS-B
    try:
        result = minimize(
            objective,
            x0=[initial_guess],
            method='L-BFGS-B',
            bounds=[(0.001, 10.0)],
        )
        if result.success and np.isfinite(result.x[0]) and 0.001 <= result.x[0] <= 10.0:
            return (idx, result.x[0])
        return (idx, np.nan)
    except (ValueError, RuntimeError, AttributeError):
        return (idx, np.nan)


def estimate_rate(snapshot: pd.DataFrame, expiration_date: pd.Timestamp, cache: bool = False) -> pd.Series:
    """
    Estimate the rate of the dividend.
    """
    r_path = CONFIG.root_path / "data" / "processed" / "r.parquet"
    if cache and r_path.exists():
        return pd.read_parquet(r_path)

    r = (
        snapshot.groupby(level="quote_datetime", group_keys=False)
        .apply(_estimate_rate, expiration_date)
        .ewm(span=60)
        .mean()
    ).r
    r_path.parent.mkdir(parents=True, exist_ok=True)
    r.to_frame("r").to_parquet(r_path)
    return r


def _estimate_rate(snapshot: pd.DataFrame, expiration_date: pd.Timestamp) -> float:
    """
    Estimate the rate of the dividend.
    """
    quote_datetime = snapshot.index.get_level_values("quote_datetime")[0]

    call_snap_shot = snapshot[["bid", "ask"]].loc[quote_datetime].xs("C", level="option_type")
    put_snap_shot = snapshot[["bid", "ask"]].loc[quote_datetime].xs("P", level="option_type")
    forward_synthetic_ask = call_snap_shot['ask'] - put_snap_shot['bid']
    forward_synthetic_bid = call_snap_shot['bid'] - put_snap_shot['ask']

    box_spread_ask = (
        pd.DataFrame(
            forward_synthetic_ask.values.reshape(-1, 1) - forward_synthetic_bid.values.reshape(1, -1),
            index=forward_synthetic_ask.index.rename("strike_small"),
            columns=forward_synthetic_ask.index.rename("strike_large"),
        )
        .unstack()
        .to_frame("ask")
    )
    box_spread_ask['strike_diff'] = box_spread_ask.index.get_level_values(
        "strike_large"
    ) - box_spread_ask.index.get_level_values("strike_small")
    box_spread_ask = (
        box_spread_ask.reset_index(drop=True).set_index("strike_diff").pipe(_filter_strike_diff_by_quantiles)
    )

    box_spread_bid = (
        pd.DataFrame(
            forward_synthetic_bid.values.reshape(-1, 1) - forward_synthetic_ask.values.reshape(1, -1),
            index=forward_synthetic_bid.index.rename("strike_small"),
            columns=forward_synthetic_bid.index.rename("strike_large"),
        )
        .unstack()
        .to_frame("bid")
    )
    box_spread_bid['strike_diff'] = box_spread_bid.index.get_level_values(
        "strike_large"
    ) - box_spread_bid.index.get_level_values("strike_small")
    box_spread_bid = (
        box_spread_bid.reset_index(drop=True).set_index("strike_diff").pipe(_filter_strike_diff_by_quantiles)
    )

    box_spread = pd.concat([box_spread_ask, box_spread_bid], axis=1)
    box_spread = box_spread.groupby("strike_diff").mean().reset_index()
    x = box_spread['strike_diff']
    y = box_spread[['ask', 'bid']].mean(axis=1).values

    # ols
    beta = (y / x).clip(0, 1).median()
    tau = (expiration_date - quote_datetime).total_seconds() / (365 * 24 * 60 * 60)
    r = -np.log(beta) / tau
    return pd.DataFrame(r, index=[quote_datetime], columns=["r"])


def _filter_strike_diff_by_quantiles(
    df: pd.DataFrame, lower_quantile: float = 0.3, upper_quantile: float = 0.7
) -> pd.DataFrame:
    """
    Filter strike_diff using quantiles to remove extreme values.

    This is more robust than hard-coded thresholds as it adapts to the data distribution.

    Args:
        df: DataFrame with 'strike_diff' as index
        lower_quantile: Lower quantile threshold (default 0.3, i.e., 30th percentile)
        upper_quantile: Upper quantile threshold (default 0.7, i.e., 70th percentile)

    Returns:
        Filtered DataFrame
    """
    strike_diffs = df.index.values
    strike_diffs = strike_diffs[strike_diffs > 0]
    if len(strike_diffs) == 0:
        return df

    # Only filter if we have enough data points
    if len(strike_diffs) > 10:
        lower_bound = np.quantile(strike_diffs, lower_quantile)
        upper_bound = np.quantile(strike_diffs, upper_quantile)
        # Ensure we filter out negative values and very small differences
        lower_bound = max(lower_bound, 0)
        return df.query(f"strike_diff >= {lower_bound} and strike_diff <= {upper_bound}")
    else:
        # For small datasets, use a simple positive filter
        return df.query("strike_diff > 0")


def estimate_iv(
    snapshot: pd.DataFrame,
    expiration_date: pd.Timestamp,
    r: pd.Series,
    binomial_tree_steps: int = 100,
    n_workers: int = 10,
    atm_guess: pd.Series | None = None,
    cache: bool = False,
) -> pd.Series:
    """
    Estimate the implied volatility.

    Args:
        snapshot: DataFrame with option data
        expiration_date: Expiration date of the options
        r: Risk-free rate
        binomial_tree_steps: Number of steps in the binomial tree (default 100)
        n_workers: Number of workers for parallel processing (default 10)
        atm_guess: Optional Series of ATM IV values to use as initial guess (indexed by (quote_datetime, option_type))
    Returns:
        Series of implied volatilities indexed by snapshot.index
    """
    iv_path = CONFIG.root_path / "data" / "processed" / "iv.parquet"
    if cache and iv_path.exists():
        return pd.read_parquet(iv_path)

    # Calculate ATM IV first if available (this provides better initial guesses)
    if atm_guess is None and 'is_atm' in snapshot.columns:
        atm_snapshot = snapshot[snapshot['is_atm']].copy()
        if len(atm_snapshot) > 0:
            # Remove is_atm column to prevent infinite recursion
            atm_snapshot_no_flag = atm_snapshot.drop(columns=['is_atm'], errors='ignore')
            # Recursively calculate ATM IV without atm_guess to avoid infinite recursion
            atm_iv = estimate_iv(atm_snapshot_no_flag, expiration_date, r, binomial_tree_steps, atm_guess=None)
            # Create a Series indexed by (quote_datetime, option_type) for easy lookup
            atm_guess = atm_iv.droplevel('strike') if 'strike' in atm_iv.index.names else atm_iv

    # Prepare data for parallel processing
    indices = list(snapshot.index)
    row_data_list = []
    initial_guesses = []

    for idx in indices:
        row = snapshot.loc[idx]
        row_data = {
            'bid': row['bid'],
            'ask': row['ask'],
            'underlying_bid': row['underlying_bid'],
            'underlying_ask': row['underlying_ask'],
        }
        row_data_list.append(row_data)

        # Get initial guess: prefer ATM IV, fallback to time value heuristic
        quote_datetime, K, option_type_str = idx
        S0 = (row['underlying_bid'] + row['underlying_ask']) / 2
        market_price = (row['bid'] + row['ask']) / 2
        T = (expiration_date - pd.Timestamp(quote_datetime)).total_seconds() / (365 * 24 * 60 * 60)

        if atm_guess is not None:
            try:
                guess_key = (quote_datetime, option_type_str)
                if guess_key in atm_guess.index:
                    initial_guesses.append(atm_guess.loc[guess_key])
                elif quote_datetime in atm_guess.index:
                    initial_guesses.append(atm_guess.loc[quote_datetime])
                else:
                    initial_guesses.append(0.2)
            except (KeyError, IndexError):
                initial_guesses.append(0.2)
        else:
            # Use time value heuristic as fallback
            option_type = 'call' if option_type_str == 'C' else 'put'
            intrinsic_value = max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
            time_value = market_price - intrinsic_value
            if time_value <= 0 or T < 1e-6:
                initial_guesses.append(0.2)
            else:
                initial_guesses.append(min(max(np.sqrt(2 * np.pi / T) * time_value / S0, 0.05), 2.0))

    # Use multiprocessing if there are enough items to process
    use_multiprocessing = len(indices) > 50  # Threshold for using multiprocessing

    if use_multiprocessing:
        # Prepare arguments for parallel processing
        args_list = [
            (
                row_data_list[i],
                indices[i],
                expiration_date,
                r.loc[indices[i][0]].item(),  # Get rate for this quote_datetime
                binomial_tree_steps,
                initial_guesses[i],
            )
            for i in range(len(indices))
        ]

        # Process in parallel
        iv_dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(_solve_single_iv, *args): args[1] for args in args_list}

            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Calculating IV"):
                try:
                    idx, iv_value = future.result()
                    iv_dict[idx] = iv_value
                except Exception:
                    idx = future_to_idx[future]
                    iv_dict[idx] = np.nan

        # Create Series from dictionary
        iv_series = pd.Series(iv_dict)
        # Reindex to match original snapshot index order
        iv_series = iv_series.reindex(snapshot.index)
    else:
        # Sequential processing for small datasets
        iv_series = pd.Series(index=snapshot.index, dtype=float)
        for i, idx in enumerate(tqdm(indices)):
            result = _solve_single_iv(
                row_data_list[i],
                idx,
                expiration_date,
                r.loc[idx[0]],
                binomial_tree_steps,
                initial_guesses[i],
            )
            iv_series.loc[idx] = result[1]

    iv_path.parent.mkdir(parents=True, exist_ok=True)
    iv_series.to_frame("iv").to_parquet(iv_path)

    return iv_series
