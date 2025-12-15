from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from utils.pricing import BinomialTree, BlackScholes

CONFIG = Config()

__all__ = ["predict_all_next_prices", "predict_next_price"]


def _predict_single_option(
    row_data: dict,
    idx: tuple,
    next_ts: pd.Timestamp,
    expiration_date: pd.Timestamp,
    current_iv: float,
    binomial_steps: int,
    exercise_type: str,
) -> tuple:
    """
    Predict price for a single option (used for multiprocessing).

    Args:
        row_data: Dictionary with next timestamp data (underlying_bid, underlying_ask, r)
        idx: Index tuple (strike, option_type)
        next_ts: Next timestamp
        expiration_date: Expiration date
        current_iv: Current implied volatility
        binomial_steps: Number of steps for binomial tree
        exercise_type: 'american' or 'european'

    Returns:
        Tuple of (next_ts, strike, option_type, predicted_price)
    """
    strike, option_type_str = idx
    option_type = 'call' if option_type_str == 'C' else 'put'

    if pd.isna(current_iv) or current_iv <= 0:
        return (next_ts, strike, option_type_str, np.nan)

    S0_next = (row_data['underlying_bid'] + row_data['underlying_ask']) / 2
    r_next = row_data.get('r', np.nan)

    # Calculate time to expiration at next timestamp
    T_next = (expiration_date - pd.Timestamp(next_ts)).total_seconds() / (365 * 24 * 60 * 60)

    if T_next <= 0 or pd.isna(T_next) or not np.isfinite(T_next):
        return (next_ts, strike, option_type_str, np.nan)

    if pd.isna(r_next) or not np.isfinite(r_next):
        return (next_ts, strike, option_type_str, np.nan)

    # Predict price using current IV
    try:
        if option_type == 'call':
            # Use Black-Scholes for calls
            bs = BlackScholes(
                S0=S0_next,
                K=strike,
                T=T_next,
                r=r_next,
                sigma=current_iv,
                dividend=0,
                option_type=option_type,
            )
            predicted_price = bs.price()
        else:
            # Use Binomial Tree for puts
            bt = BinomialTree(
                S0=S0_next,
                K=strike,
                T=T_next,
                r=r_next,
                sigma=current_iv,
                dividend=0,
                n_steps=binomial_steps,
                option_type=option_type,
                exercise_type=exercise_type,
            )
            predicted_price = bt.price()

        if not np.isfinite(predicted_price) or predicted_price < 0:
            predicted_price = np.nan

    except (ValueError, TypeError, AttributeError, ZeroDivisionError):
        predicted_price = np.nan

    return (next_ts, strike, option_type_str, predicted_price)


def predict_next_price(
    snapshot: pd.DataFrame,
    expiration_date: pd.Timestamp,
    binomial_steps: int = 100,
    exercise_type: str = "american",
    n_workers: int = 10,
) -> pd.Series:
    """
    Predict option prices for the next timestamp using current IV.

    The prediction uses:
    - Current timestamp's IV as the volatility estimate for next timestamp
    - Next timestamp's underlying price (S0)
    - Next timestamp's time to expiration (T)
    - Current or next timestamp's risk-free rate (r)

    Note: Uses Black-Scholes for calls and Binomial Tree for puts.
          American calls without dividends don't need early exercise, so BS is sufficient.
          American puts may need early exercise, so Binomial Tree is required.

    Args:
        snapshot: DataFrame with multi-index (quote_datetime, strike, option_type)
                 Must have columns: 'iv', 'r', 'underlying_bid', 'underlying_ask'
        expiration_date: Expiration date of the options
        binomial_steps: Number of steps for binomial tree (default 100)
        exercise_type: 'american' or 'european' (default 'american')
        n_workers: Number of workers for parallel processing (default 10)

    Returns:
        Series of predicted prices indexed by snapshot.index
    """
    if 'iv' not in snapshot.columns:
        raise ValueError("Snapshot must have 'iv' column (implied volatility)")
    if 'r' not in snapshot.columns:
        raise ValueError("Snapshot must have 'r' column (risk-free rate)")

    # Get unique timestamps sorted
    timestamps = snapshot.index.get_level_values("quote_datetime").unique().sort_values()

    if len(timestamps) < 2:
        raise ValueError("Need at least 2 timestamps to predict next price")

    # Prepare all tasks for parallel processing
    all_tasks = []
    for i, current_ts in enumerate(timestamps[:-1]):
        next_ts = timestamps[i + 1]
        current_snapshot = snapshot.loc[current_ts]
        next_snapshot = snapshot.loc[next_ts]

        for idx in current_snapshot.index:
            strike = idx[0]  # Extract strike for filtering

            # Get current IV
            try:
                current_iv = current_snapshot.loc[idx, 'iv']
                if pd.isna(current_iv):
                    strike_mask = current_snapshot.index.get_level_values('strike') == strike
                    iv_candidates = current_snapshot.loc[strike_mask, 'iv']
                    current_iv = iv_candidates.iloc[0] if len(iv_candidates) > 0 else np.nan
                if isinstance(current_iv, pd.Series):
                    current_iv = current_iv.iloc[0] if len(current_iv) > 0 else np.nan
            except (KeyError, IndexError):
                current_iv = np.nan

            # Get next timestamp's parameters
            try:
                next_row = next_snapshot.loc[idx]
                row_data = {
                    'underlying_bid': next_row['underlying_bid'],
                    'underlying_ask': next_row['underlying_ask'],
                    'r': next_row['r'] if 'r' in next_row else current_snapshot.loc[idx, 'r'],
                }
            except (KeyError, IndexError):
                row_data = None

            if row_data is not None:
                all_tasks.append(
                    (
                        row_data,
                        idx,
                        next_ts,
                        expiration_date,
                        current_iv,
                        binomial_steps,
                        exercise_type,
                    )
                )

    # Use multiprocessing if there are enough tasks
    use_multiprocessing = len(all_tasks) > 50

    if use_multiprocessing:
        predicted_prices = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(_predict_single_option, *task): task for task in all_tasks}

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Predicting prices",
                miniters=1,
                mininterval=0.1,
            ):
                try:
                    result = future.result()
                    predicted_prices.append(result)
                except Exception:
                    task = future_to_task[future]
                    predicted_prices.append((task[1], task[2], task[1][1], np.nan))
    else:
        # Sequential processing for small datasets
        predicted_prices = []
        for task in tqdm(all_tasks, desc="Predicting prices", miniters=1, mininterval=0.1):
            result = _predict_single_option(*task)
            predicted_prices.append(result)

    # Create result Series with same index structure as snapshot
    if not predicted_prices:
        return pd.Series(dtype=float, index=snapshot.index)

    # Create MultiIndex for predicted prices
    predicted_index = pd.MultiIndex.from_tuples(
        [(ts, strike, opt_type) for ts, strike, opt_type, _ in predicted_prices],
        names=snapshot.index.names,
    )
    predicted_values = [price for _, _, _, price in predicted_prices]

    predicted_series = pd.Series(predicted_values, index=predicted_index)

    # Reindex to match original snapshot structure (only for timestamps we predicted)
    # Only include predictions for timestamps that exist in snapshot
    valid_indices = predicted_series.index.intersection(snapshot.index)
    predicted_series = predicted_series.reindex(valid_indices)

    return predicted_series


def predict_all_next_prices(
    snapshot: pd.DataFrame,
    expiration_date: pd.Timestamp,
    binomial_steps: int = 100,
    exercise_type: str = "american",
    n_workers: int = 10,
    cache: bool = False,
) -> pd.DataFrame:
    """
    Predict option prices for all next timestamps and add as a column.

    Args:
        snapshot: DataFrame with option data
        expiration_date: Expiration date
        binomial_steps: Number of steps for binomial tree
        exercise_type: 'american' or 'european' (default 'american')
        n_workers: Number of workers for parallel processing (default 10)

    Returns:
        DataFrame with added 'predicted_price' column
    """
    nvda_path = CONFIG.root_path / "data" / "processed" / "NVDA.parquet"
    if cache and nvda_path.exists():
        return pd.read_parquet(nvda_path)

    predicted = predict_next_price(snapshot, expiration_date, binomial_steps, exercise_type, n_workers)

    result = snapshot.copy()
    result['predicted_price'] = predicted

    result.to_parquet(nvda_path)

    return result
