# Option Pricing and Prediction System

Final Project for MTH 9871 - Option Price Prediction System Based on Implied Volatility

## Project Overview

This project implements a complete option pricing and prediction system, including:

- **Risk-Free Rate Estimation**: Estimates risk-free rates from option market data using the Box Spread method
- **Implied Volatility Calculation**: Calculates implied volatility for American options using Black-Scholes and Binomial Tree models
- **Price Prediction**: Predicts option prices for the next timestamp based on current implied volatility
- **Multiprocessing Acceleration**: Supports parallel computation to improve processing speed

## Project Structure

```
.
├── core/                          # Core functionality modules
│   ├── implied_vol_calculator.py # Implied volatility calculation
│   └── option_price_predictor.py # Option price prediction
├── utils/                         # Utility modules
│   ├── dataloader.py             # Data loading and preprocessing
│   ├── pricing.py                # Option pricing models (BS, Binomial Tree)
│   └── greeks.py                 # Greeks calculation (delta, gamma)
├── sim/                           # Simulation and backtesting
│   ├── backtester.py             # Sequential backtester with hedging
│   ├── strategy.py               # Strategy interface and implementations
│   ├── trade_simulator.py        # Fill model simulation
│   ├── hedger.py                 # Delta hedging module
│   └── metrics.py                # Performance metrics
├── scripts/                       # Script files
│   ├── preprocess_data.py        # Data preprocessing script
│   ├── predict_iv.py             # Prediction script
│   ├── run_backtest.py           # Run market-making backtest
│   └── analyze_vol_fit.py        # Vol-fit analysis
├── data/                          # Data directory
│   ├── raw/                      # Raw OPRA snapshots
│   └── processed/                # Processed data and results
├── config.py                      # Configuration file
```

## Installation

### Requirements

- Python >= 3.11
- Virtual environment recommended

### Installation Steps

1. Clone the project (if applicable)
```bash
git clone <repository-url>
cd Final
```

2. Create virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install -e .
```

Or using uv (if installed):
```bash
uv pip install -e .
```

## Usage

### 1. Data Preprocessing

First, calculate the risk-free rate and implied volatility:

```python
from scripts.preprocess_data import main

main()  # Calculate r and IV, results saved in data/processed/
```

Or call manually:

```python
from core.implied_vol_calculator import estimate_iv, estimate_rate
from utils.dataloader import load_raw_data

# Load raw data
snapshot, expiration_date = load_raw_data()

# Estimate risk-free rate
r = estimate_rate(snapshot, expiration_date, cache=True)

# Calculate implied volatility
iv = estimate_iv(
    snapshot, 
    expiration_date, 
    r, 
    binomial_tree_steps=100,
    n_workers=10,
    cache=True
)
```

### 2. Price Prediction

Use the calculated IV to predict option prices for the next timestamp:

```python
from core.option_price_predictor import predict_all_next_prices
from utils.dataloader import load_data

# Load data (including r and iv)
snapshot, expiration_date = load_data()

# Predict prices
snapshot_with_predictions = predict_all_next_prices(
    snapshot,
    expiration_date,
    binomial_steps=100,
    exercise_type="american",
    n_workers=10,
    cache=True
)
```

## Core Features

### Risk-Free Rate Estimation (`estimate_rate`)

- **Method**: Box Spread method
- **Principle**: Uses the difference between synthetic forward prices from call and put options to estimate risk-free rate
- **Features**:
  - Uses quantile filtering to remove extreme values
  - Uses Exponential Weighted Moving Average (EWM) to smooth the rate series
  - Supports caching to speed up repeated calculations

### Implied Volatility Calculation (`estimate_iv`)

- **Method**: L-BFGS-B optimization algorithm
- **Pricing Models**:
  - Call options: Black-Scholes (American calls without dividends don't need early exercise)
  - Put options: Binomial Tree (handles early exercise for American puts)
- **Features**:
  - Automatically calculates ATM IV as initial guess to improve convergence speed
  - Supports multiprocessing for parallel computation
  - Supports caching

### Price Prediction (`predict_next_price`)

- **Method**: Uses current timestamp's IV to predict next timestamp's price
- **Pricing Models**:
  - Call: Black-Scholes
  - Put: Binomial Tree (American)
- **Features**:
  - Supports multiprocessing acceleration
  - Automatically handles missing data
  - Supports caching

## Trading Strategies

This repo currently includes a minimal market-making backtester under `sim/`:

- Generate our quotes per timestamp (default: improve inside the market spread by a fixed fraction)
- Simulate fills:
  - Cross -> immediate fill
  - Otherwise -> probabilistic hit based on aggressiveness vs market quotes
- Log quotes/trades/timeline and compute headline metrics

## Risk Management

### Delta Hedging

The system implements **threshold-based delta hedging** to manage directional risk:

- **Portfolio Delta Calculation**: Computes aggregate delta from all option positions
  - Calls: Black-Scholes analytical delta
  - Puts: Numerical delta via finite difference on binomial tree
- **Hedging Policy**: Hedge to delta-neutral when `|portfolio_delta| > threshold`
- **Execution**: Trade underlying shares at mid price (with configurable spread)

Configuration in `HedgerConfig`:
```python
HedgerConfig(
    enabled=True,
    delta_threshold=10.0,  # Hedge when |delta| > 10
    hedge_spread=0.0001,   # 1 bps spread on underlying
)
```

### Position Limits

The strategy enforces position limits to prevent excessive inventory:
- `max_position_per_contract`: Maximum position per (strike, option_type)
- `max_total_position`: Maximum total position across all contracts

When at limit, the strategy stops quoting on the side that would increase exposure.

## Backtesting (Market Making)

Run the minimal backtest demo:

```python
from scripts.run_backtest import main

main()
```

## Strategy Interface (Handover)

Strategies live under `sim/strategy.py` and must implement `BaseStrategy`.

### Required methods

- **`select_universe(market_at_ts) -> pd.Index`**

  - **Input**: `market_at_ts` is the market snapshot at one timestamp (a cross-section).
    - **Index**: `("strike", "option_type")`
    - **Required columns**: `bid`, `ask`
    - **Optional columns**: `is_atm`, `underlying_bid`, `underlying_ask`, `iv`, `r`, `predicted_price`, ...
  - **Output**: an index subset of contracts to quote (same index type as above).

- **`generate_quotes(market_at_ts, universe, positions, predicted_at_ts) -> pd.DataFrame`**
  - **Inputs**:
    - `market_at_ts`: same as above.
    - `universe`: the selected contract index (subset of `market_at_ts.index`).
    - `positions`: `dict[(strike, option_type) -> qty]` representing current inventory.
      - `qty > 0` long; `qty < 0` short.
    - `predicted_at_ts`: DataFrame aligned to `("strike","option_type")`.
      - If available, contains column `predicted_price`.
      - If not available, it will be an empty DataFrame but with the same index.
  - **Output**: a DataFrame with:
    - **Index**: `("strike", "option_type")` (typically exactly `universe`)
    - **Required columns**: `my_bid`, `my_ask`
    - **Constraints**: `my_bid > 0`, `my_ask > 0`, `my_ask >= my_bid` (otherwise fills are skipped).

### How the simulator uses your quotes (important)

The fill model is implemented in `sim/trade_simulator.py`:

- **Crossing**:
  - If `my_bid >= market_ask` -> immediate buy fill at `market_ask`
  - If `my_ask <= market_bid` -> immediate sell fill at `market_bid`
- **Otherwise probabilistic**:
  - If you are **more aggressive** (inside-spread), fill probability increases with aggressiveness.
  - If you are **not more aggressive** (join/worse), fill probability is penalized by queue size ahead if
    `bid_size/ask_size` exists in the data (negative correlation).
  - If hit, fills are **always 1 contract** (configurable via `FillModelConfig.trade_size`).

### Minimal template

```python
import pandas as pd
from sim.strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    def select_universe(self, market_at_ts: pd.DataFrame) -> pd.Index:
        return market_at_ts.index  # or select ATM band

    def generate_quotes(
        self,
        market_at_ts: pd.DataFrame,
        universe: pd.Index,
        positions: dict[tuple[float, str], int],
        predicted_at_ts: pd.DataFrame,
    ) -> pd.DataFrame:
        mkt = market_at_ts.loc[universe, ["bid", "ask"]]
        mid = (mkt["bid"] + mkt["ask"]) / 2

        if "predicted_price" in predicted_at_ts.columns:
            theo = predicted_at_ts.loc[universe, "predicted_price"].fillna(mid)
        else:
            theo = mid

        half = 0.01
        return pd.DataFrame(
            {"my_bid": (theo - half).clip(lower=0.0), "my_ask": (theo + half).clip(lower=0.0)},
            index=universe,
        )
```

## Technical Details

### Pricing Models

- **Black-Scholes**: For European options and American Call options (without dividends)
- **Binomial Tree**: For American Put options, supports early exercise

### Optimization Algorithms

- **L-BFGS-B**: For IV calculation, supports boundary constraints
- **Initial Guess**: Uses ATM IV as initial value to improve convergence speed

### Parallel Computing

- **Multiprocessing**: Uses `ProcessPoolExecutor` for parallel computation
- **Auto Selection**: Automatically chooses sequential or parallel processing based on data size
- **Performance**: Can achieve 2-4x speedup on multi-core CPUs

## Data Format

### Input Data

Raw data should contain the following columns:
- `quote_datetime`: Quote timestamp
- `strike`: Strike price
- `option_type`: Option type ('C' or 'P')
- `bid`, `ask`: Bid and ask prices
- `underlying_bid`, `underlying_ask`: Underlying asset bid and ask prices
- `expiration`: Expiration date

### Output Data

Processed data includes:
- `r`: Risk-free rate
- `iv`: Implied volatility
- `predicted_price`: Predicted price (if prediction was run)

## Assumptions

The following assumptions are made in this implementation:

1. **No transaction costs on options**: Option fills occur at quoted prices without fees
2. **Underlying hedges at mid price**: Hedge trades execute at mid with small spread (configurable)
3. **Continuous market**: No market gaps or halts; prices are always available
4. **IV persistence**: Current IV is a reasonable predictor of next-timestamp fair value
5. **No dividends**: American calls are priced using Black-Scholes (no early exercise benefit)
6. **Single expiry**: All options share the same expiration date
7. **Probabilistic fills**: Non-crossing orders fill probabilistically based on aggressiveness

## Model Limitations

1. **American put pricing**: Uses binomial tree approximation (100 steps default)
2. **Greeks computation**: Numerical delta for puts adds computational overhead
3. **No smile dynamics**: IV is computed independently per contract, no arbitrage-free interpolation
4. **Fill model simplification**: Real market microstructure is more complex
5. **No latency modeling**: Assumes instantaneous quote updates and fills
6. **Risk-free rate estimation**: Box spread method may be noisy for illiquid strikes

## Areas for Improvement

For a production trading system, the following areas would need enhancement:

### High Priority
- **Transaction cost modeling**: Include realistic bid-ask spreads, commissions, and market impact
- **Greeks computation speed**: Pre-compute or cache Greeks, use analytical approximations
- **IV surface interpolation**: Use arbitrage-free methods (SVI, SABR) for consistent pricing

### Medium Priority
- **Multi-expiry support**: Quote across multiple expiries with cross-expiry hedging
- **Vega hedging**: Use options at different strikes to neutralize vega exposure
- **Real-time latency**: Model quote-to-fill delays and quote staleness

### Lower Priority
- **Dividend handling**: Support discrete dividends for accurate American option pricing
- **Jump risk**: Model gap risk and extreme moves
- **Regulatory constraints**: Position limits, margin requirements, reporting

## Vol-Fit Analysis

Run the vol-fit analysis to evaluate prediction quality:

```python
from scripts.analyze_vol_fit import main
main()
```

This produces:
- Overall prediction error statistics
- Prediction errors by moneyness bucket
- Prediction errors by option type (call vs put)
- IV stability analysis over time

## Contributing

Issues and Pull Requests are welcome.

## License

This project is for educational purposes only.

## Contact

For questions, please contact via Issues.
