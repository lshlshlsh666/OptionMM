"""
Vol-fit analysis script.

Analyzes how well implied volatility predictions match market prices.
"""

import logging

import numpy as np
import pandas as pd

from config import Config
from utils.dataloader import load_raw_data
from core.option_price_predictor import predict_all_next_prices

CONFIG = Config()
logger = logging.getLogger(__name__)


def analyze_prediction_errors(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze prediction errors: predicted_price vs market mid.

    Returns DataFrame with error statistics.
    """
    if "predicted_price" not in snapshot.columns:
        raise ValueError("snapshot must have 'predicted_price' column")

    snapshot = snapshot.copy()
    snapshot["market_mid"] = (snapshot["bid"] + snapshot["ask"]) / 2
    snapshot["prediction_error"] = snapshot["predicted_price"] - snapshot["market_mid"]
    snapshot["prediction_error_pct"] = snapshot["prediction_error"] / snapshot["market_mid"] * 100

    # Remove NaN predictions
    valid = snapshot.dropna(subset=["predicted_price", "market_mid"])

    # Overall statistics
    stats = {
        "n_observations": len(valid),
        "mean_error": valid["prediction_error"].mean(),
        "std_error": valid["prediction_error"].std(),
        "median_error": valid["prediction_error"].median(),
        "mean_abs_error": valid["prediction_error"].abs().mean(),
        "rmse": np.sqrt((valid["prediction_error"] ** 2).mean()),
        "mean_error_pct": valid["prediction_error_pct"].mean(),
        "std_error_pct": valid["prediction_error_pct"].std(),
    }

    return pd.Series(stats)


def analyze_by_moneyness(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze prediction errors by moneyness buckets.
    """
    snapshot = snapshot.copy()
    snapshot["market_mid"] = (snapshot["bid"] + snapshot["ask"]) / 2
    snapshot["prediction_error"] = snapshot["predicted_price"] - snapshot["market_mid"]

    # Create moneyness buckets
    if "log_moneyness" in snapshot.columns:
        snapshot["moneyness_bucket"] = pd.cut(
            snapshot["log_moneyness"],
            bins=[-np.inf, -0.1, -0.05, 0.0, 0.05, 0.1, np.inf],
            labels=["Deep ITM", "ITM", "Slight ITM", "Slight OTM", "OTM", "Deep OTM"],
        )
    else:
        return pd.DataFrame()

    valid = snapshot.dropna(subset=["predicted_price", "market_mid", "moneyness_bucket"])

    stats_by_moneyness = valid.groupby("moneyness_bucket", observed=True).agg(
        n_obs=("prediction_error", "count"),
        mean_error=("prediction_error", "mean"),
        std_error=("prediction_error", "std"),
        mean_abs_error=("prediction_error", lambda x: x.abs().mean()),
    )

    return stats_by_moneyness


def analyze_by_option_type(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze prediction errors by option type (call vs put).
    """
    snapshot = snapshot.copy()
    snapshot["market_mid"] = (snapshot["bid"] + snapshot["ask"]) / 2
    snapshot["prediction_error"] = snapshot["predicted_price"] - snapshot["market_mid"]

    option_type = snapshot.index.get_level_values("option_type")

    valid = snapshot.dropna(subset=["predicted_price", "market_mid"])

    stats_by_type = valid.groupby(option_type).agg(
        n_obs=("prediction_error", "count"),
        mean_error=("prediction_error", "mean"),
        std_error=("prediction_error", "std"),
        mean_abs_error=("prediction_error", lambda x: x.abs().mean()),
        rmse=("prediction_error", lambda x: np.sqrt((x**2).mean())),
    )

    return stats_by_type


def analyze_iv_stability(snapshot: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how stable IV is across timestamps.
    """
    if "iv" not in snapshot.columns:
        return pd.DataFrame()

    # Group by (strike, option_type) and analyze IV over time
    iv_stats = []
    for idx in snapshot.index.droplevel("quote_datetime").unique():
        try:
            iv_series = snapshot.xs(idx, level=["strike", "option_type"])["iv"].dropna()
            if len(iv_series) > 1:
                iv_stats.append(
                    {
                        "strike": idx[0],
                        "option_type": idx[1],
                        "mean_iv": iv_series.mean(),
                        "std_iv": iv_series.std(),
                        "cv_iv": iv_series.std() / iv_series.mean() if iv_series.mean() > 0 else np.nan,
                        "n_obs": len(iv_series),
                    }
                )
        except (KeyError, IndexError):
            continue

    if not iv_stats:
        return pd.DataFrame()

    df = pd.DataFrame(iv_stats)

    return pd.DataFrame(
        {
            "mean_iv": df["mean_iv"].mean(),
            "avg_iv_std": df["std_iv"].mean(),
            "avg_iv_cv": df["cv_iv"].mean(),
            "n_contracts": len(df),
        },
        index=[0],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print("Loading data...")
    snapshot, expiration_date = load_raw_data()
    snapshot = predict_all_next_prices(snapshot, expiration_date, cache=True)

    print("\n" + "=" * 60)
    print("VOL-FIT ANALYSIS")
    print("=" * 60)

    # Overall prediction error statistics
    print("\n--- Overall Prediction Error Statistics ---")
    overall_stats = analyze_prediction_errors(snapshot)
    print(overall_stats.to_string())

    # By moneyness
    print("\n--- Prediction Errors by Moneyness ---")
    moneyness_stats = analyze_by_moneyness(snapshot)
    if not moneyness_stats.empty:
        print(moneyness_stats.to_string())
    else:
        print("No moneyness data available")

    # By option type
    print("\n--- Prediction Errors by Option Type ---")
    type_stats = analyze_by_option_type(snapshot)
    print(type_stats.to_string())

    # IV stability
    print("\n--- IV Stability Analysis ---")
    iv_stats = analyze_iv_stability(snapshot)
    if not iv_stats.empty:
        print(iv_stats.to_string())
    else:
        print("No IV data available")

    # Save results
    output_dir = CONFIG.root_path / "data" / "processed" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_stats.to_frame("value").to_csv(output_dir / "overall_prediction_stats.csv")
    if not moneyness_stats.empty:
        moneyness_stats.to_csv(output_dir / "prediction_by_moneyness.csv")
    type_stats.to_csv(output_dir / "prediction_by_option_type.csv")
    if not iv_stats.empty:
        iv_stats.to_csv(output_dir / "iv_stability.csv", index=False)

    print(f"\nAnalysis results saved to {output_dir}/")


if __name__ == "__main__":
    main()
