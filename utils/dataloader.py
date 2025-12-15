from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

CONFIG = Config()

__all__ = ["load_data", "load_raw_data"]


def load_data(raw_file_path: Path = CONFIG.root_path / "data" / "raw") -> tuple[pd.DataFrame, pd.Timestamp]:
    snapshot, expiration_date = load_raw_data(raw_file_path)
    r = pd.read_parquet(CONFIG.root_path / "data" / "processed" / "r.parquet")
    r.index.name = "quote_datetime"
    iv = pd.read_parquet(CONFIG.root_path / "data" / "processed" / "iv.parquet")
    return snapshot.join(r).join(iv), expiration_date


def load_raw_data(file_path: Path = CONFIG.root_path / "data" / "raw") -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Load data from the raw data folder, and do the following preprocessing:
    1. Sort all expiration dates in ascending order.
    2. Keep only the option contracts with the latest expiration date.
    3. Mark the atm option contract.
    4. Add the log moneyness.
    """
    snapshot = pd.read_parquet(file_path)
    snapshot, expiration_date = _filter_expiration(snapshot)
    snapshot = _mark_atm(snapshot)
    snapshot = _add_log_moneyness(snapshot)
    return snapshot.copy(), expiration_date


def _filter_expiration(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Filter the snapshot by the latest expiration date with at least 1 month left.
    Return the snapshot and the expiration date.
    """
    ticker = snapshot.index.get_level_values("root")[0]
    expirations = snapshot.index.get_level_values("expiration").unique().sort_values()
    last_ts = snapshot.index.get_level_values("quote_datetime").max()
    for expiration in expirations:
        if (expiration - last_ts).total_seconds() / (30 * 24 * 60 * 60) > 1:
            return snapshot.xs(expiration, level="expiration").loc[ticker], expiration + pd.Timedelta(hours=16)
    return snapshot.xs(expiration, level="expiration").loc[ticker], expiration + pd.Timedelta(hours=16)


def _mark_atm(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Mark the atm option contract by the smallest absolute difference between the underlying price and the strike price.
    """
    snapshot['S-K'] = (snapshot['underlying_ask'] + snapshot['underlying_bid']) / 2 - snapshot.index.get_level_values(
        "strike"
    )
    snapshot['is_atm'] = snapshot.groupby(level=["quote_datetime", "option_type"], group_keys=False).apply(
        lambda x: x['S-K'].abs() == x['S-K'].abs().min()
    )
    del snapshot['S-K']
    return snapshot


def _add_log_moneyness(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Add the log moneyness to the snapshot.
    """
    snapshot['log_moneyness'] = np.log(
        snapshot.index.get_level_values('strike') / (snapshot['underlying_ask'] + snapshot['underlying_bid']) * 2
    )
    return snapshot
