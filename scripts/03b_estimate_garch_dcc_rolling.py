#!/usr/bin/env python3
"""Script to estimate GARCH-DCC-LRMES on rolling windows.

This script implements rolling window estimation for time-series SRISK analysis:
1. Load processed returns data
2. Initialize RollingWindowEstimator with 5-year windows
3. For each window (stepped weekly):
   - Estimate benchmark GARCH
   - Process all banks in parallel (bivariate GARCH-DCC-LRMES)
   - Forward-fill failed estimations
4. Generate panel data (window_date × bank_ric)
5. Save results for time-varying SRISK calculation

Configuration:
    - Window size: CONFIG.ROLLING_WINDOW_DAYS (default 1260 days = 5 years)
    - Step size: CONFIG.ROLLING_STEP_DAYS (default 5 days = weekly)
    - Parallelization: CONFIG.ROLLING_N_JOBS (default -1 = all CPUs)

Output:
    - data/results/lrmes_rolling_panel.parquet: Panel data with rolling LRMES

Usage:
    uv run scripts/03b_estimate_garch_dcc_rolling.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.rolling_window import RollingWindowEstimator
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd
import numpy as np

# Configure logging to suppress warnings that interfere with progress bar
# Warnings are logged to file instead of printed to stdout
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DATA_DIR / 'rolling_estimation.log'),
        # Only show ERROR and CRITICAL to stdout (not WARNING)
        logging.StreamHandler(sys.stdout)
    ]
)
# Set stdout handler to only show errors
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.setLevel(logging.ERROR)


def load_processed_data():
    """Load processed returns and bank universe."""
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)

    # Load bank returns and benchmarks
    bank_returns = load_dataframe("returns_banks_clean", PROCESSED_DATA_DIR)
    benchmark_returns = load_dataframe("returns_benchmarks_clean", PROCESSED_DATA_DIR)
    universe = load_dataframe("banks_universe_clean", PROCESSED_DATA_DIR)

    print(f"\nBank returns shape: {bank_returns.shape}")
    print(f"Benchmark returns shape: {benchmark_returns.shape}")
    print(f"Date range: {bank_returns.index.min()} to {bank_returns.index.max()}")
    print(f"Banks: {len(universe)}")
    print(f"Benchmarks available: {', '.join(benchmark_returns.columns.tolist())}")

    return bank_returns, benchmark_returns, universe


def main():
    """Main execution function for rolling window estimation."""

    print("\n" + "=" * 80)
    print("ROLLING WINDOW GARCH-DCC-LRMES ESTIMATION")
    print("=" * 80)
    print(f"Logging: Warnings written to {RESULTS_DATA_DIR / 'rolling_estimation.log'}")
    print(f"         (stdout only shows errors to preserve progress bar)")
    print("=" * 80)

    # Check if rolling window is enabled
    if not CONFIG.ROLLING_WINDOW_ENABLED:
        print("\n⚠ Rolling window estimation is DISABLED in config.")
        print("Set CONFIG.ROLLING_WINDOW_ENABLED = True to enable.")
        return

    # Load data
    bank_returns, benchmark_returns, universe = load_processed_data()

    # Select benchmark
    benchmark_name = CONFIG.BENCHMARK_BROAD

    if benchmark_name not in benchmark_returns.columns:
        print(f"\n✗ ERROR: Benchmark {benchmark_name} not found in data")
        print(f"Available benchmarks: {benchmark_returns.columns.tolist()}")
        return

    print(f"\nUsing benchmark: {benchmark_name}")

    # Initialize rolling window estimator
    print("\n" + "=" * 80)
    print("INITIALIZING ROLLING WINDOW ESTIMATOR")
    print("=" * 80)

    estimator = RollingWindowEstimator(
        bank_returns=bank_returns[['FAB.AD', 'ENBD.DU']], # remove this hardcoding!!
        benchmark_returns=benchmark_returns, 
        benchmark_name=benchmark_name,
        window_days=CONFIG.ROLLING_WINDOW_DAYS,
        step_days=CONFIG.ROLLING_STEP_DAYS,
        start_date=CONFIG.ROLLING_WINDOW_START_DATE,
        n_jobs=CONFIG.ROLLING_N_JOBS,
        min_crisis_paths=CONFIG.ROLLING_MIN_CRISIS_PATHS
    )

    # Run rolling estimation
    intermediate_path = RESULTS_DATA_DIR / "lrmes_rolling_panel_intermediate.parquet"

    panel = estimator.run_rolling_estimation(
        save_intermediate=True,
        intermediate_path=intermediate_path
    )

    # Save final results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_path = RESULTS_DATA_DIR / "lrmes_rolling_panel.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"✓ Saved rolling LRMES panel: {output_path}")

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Panel dimensions
    print(f"\nPanel dimensions:")
    print(f"  Windows: {panel['window_date'].nunique()}")
    print(f"  Banks: {panel['bank_ric'].nunique()}")
    print(f"  Total observations: {len(panel)}")
    print(f"  Window date range: {panel['window_date'].min()} to {panel['window_date'].max()}")

    # LRMES statistics
    print(f"\nLRMES statistics:")
    print(f"  Mean: {panel['lrmes'].mean():.4f}")
    print(f"  Std: {panel['lrmes'].std():.4f}")
    print(f"  Min: {panel['lrmes'].min():.4f}")
    print(f"  25th percentile: {panel['lrmes'].quantile(0.25):.4f}")
    print(f"  Median: {panel['lrmes'].median():.4f}")
    print(f"  75th percentile: {panel['lrmes'].quantile(0.75):.4f}")
    print(f"  Max: {panel['lrmes'].max():.4f}")

    # Convergence statistics
    print(f"\nConvergence:")
    print(f"  Convergence rate: {panel['convergence'].mean()*100:.1f}%")
    print(f"  Successful estimations: {panel['convergence'].sum()}/{len(panel)}")

    # Top banks by average LRMES
    print(f"\nTop 10 banks by average LRMES:")
    top_banks = panel.groupby('bank_ric')['lrmes'].mean().sort_values(ascending=False).head(10)
    for i, (ric, lrmes) in enumerate(top_banks.items(), 1):
        print(f"  {i:2d}. {ric:20s} {lrmes:.4f}")

    # Crisis scenario statistics
    print(f"\nCrisis scenario statistics:")
    print(f"  Average crisis scenarios per estimation: {panel['n_crisis_scenarios'].mean():.0f}")
    print(f"  Min crisis scenarios: {panel['n_crisis_scenarios'].min():.0f}")
    print(f"  Max crisis scenarios: {panel['n_crisis_scenarios'].max():.0f}")
    print(f"  Average crisis probability: {panel['crisis_probability'].mean()*100:.2f}%")

    # Correlation statistics
    print(f"\nBank-market correlation statistics:")
    print(f"  Average correlation: {panel['avg_correlation'].mean():.4f}")
    print(f"  Std: {panel['avg_correlation'].std():.4f}")
    print(f"  Range: [{panel['avg_correlation'].min():.4f}, {panel['avg_correlation'].max():.4f}]")

    # GARCH persistence statistics
    print(f"\nGARCH persistence statistics:")
    print(f"  Average bank GARCH persistence: {panel['garch_persistence'].mean():.4f}")
    print(f"  Average DCC persistence: {panel['dcc_persistence'].mean():.4f}")

    # Importance sampling statistics (if enabled)
    if 'efficiency_ratio' in panel.columns:
        print(f"\nImportance sampling statistics:")
        print(f"  Average efficiency ratio: {panel['efficiency_ratio'].mean():.2%}")
        print(f"  Average ESS: {panel['effective_sample_size'].mean():.0f}")
        print(f"  Min efficiency: {panel['efficiency_ratio'].min():.2%}")
        print(f"  Max efficiency: {panel['efficiency_ratio'].max():.2%}")
        print(f"  Std efficiency: {panel['efficiency_ratio'].std():.2%}")

    # Time series analysis
    print(f"\nTime series analysis:")
    lrmes_by_window = panel.groupby('window_date')['lrmes'].mean()
    print(f"  Average LRMES first window: {lrmes_by_window.iloc[0]:.4f}")
    print(f"  Average LRMES last window: {lrmes_by_window.iloc[-1]:.4f}")
    print(f"  Change: {(lrmes_by_window.iloc[-1] - lrmes_by_window.iloc[0]):.4f}")

    # Check for windows with low crisis scenarios
    low_crisis_windows = panel[panel['n_crisis_scenarios'] < CONFIG.ROLLING_MIN_CRISIS_PATHS]
    if len(low_crisis_windows) > 0:
        print(f"\n⚠ WARNING: {len(low_crisis_windows)} estimations have fewer than {CONFIG.ROLLING_MIN_CRISIS_PATHS} crisis scenarios")
        print(f"  Consider:")
        print(f"    - Increasing N_SIMULATIONS (current: {CONFIG.N_SIMULATIONS})")
        print(f"    - Using less extreme CRISIS_THRESHOLD (current: {CONFIG.CRISIS_THRESHOLD*100:.0f}%)")

    print("\n" + "=" * 80)
    print("ROLLING WINDOW ESTIMATION COMPLETE")
    print("=" * 80)
    print(f"\nWarnings logged to: {RESULTS_DATA_DIR / 'rolling_estimation.log'}")
    print(f"\nNext step: Run 04_calculate_srisk.py to compute time-varying SRISK")


if __name__ == "__main__":
    main()
