import logging
import pandas as pd

from sim.backtester import BacktestConfig, run_backtest
from sim.strategy import SimpleImproveInsideStrategy
from sim.trade_simulator import FillModelConfig
from utils.dataloader import load_raw_data
from core.option_price_predictor import predict_all_next_prices


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Minimal demo: only needs market bid/ask snapshots.
    snapshot, _expiration_date = load_raw_data()
    snapshot = predict_all_next_prices(snapshot, _expiration_date, cache=True)



    cfg = BacktestConfig(
        initial_cash=0.0,
        seed=7,
        log_every=50,
        strategy=SimpleImproveInsideStrategy(
            n_strikes=3,
            option_types=("C", "P"),
            improvement_frac=0.25,
            min_half_spread=0.01,
        ),
        fills=FillModelConfig(
            p_base=0.02,
            p_max=0.40,
            gamma=1.5,
            trade_size=1,
        ),
        max_fills_per_timestamp=None,  # set to 1 if you want at most one trade per bar
    )

    result = run_backtest(snapshot, cfg=cfg)

    print("=== Backtest metrics ===")
    print(
        pd.Series(
            {
                "total_pnl": result.metrics.total_pnl,
                "total_trades": result.metrics.total_trades,
                "fill_rate": result.metrics.fill_rate,
                "max_drawdown": result.metrics.max_drawdown,
                "avg_abs_position": result.metrics.avg_abs_position,
                "avg_quote_spread": result.metrics.avg_quote_spread,
            }
        )
    )

    # Save logs for later analysis.
    result_dir = "data/processed"
    result.quotes.to_parquet(f"{result_dir}/backtest_quotes.parquet", index=False)
    result.trades.to_parquet(f"{result_dir}/backtest_trades.parquet", index=False)
    result.timeline.to_parquet(f"{result_dir}/backtest_timeline.parquet", index=False)


if __name__ == "__main__":
    main()
