#!/usr/bin/env python3
"""Script to estimate CoVaR on rolling windows.

This script implements rolling window estimation for time-series CoVaR analysis:
1. Load processed returns, market caps, and state variables
2. Calculate benchmark rolling volatility
3. Initialize RollingCoVaREstimator with 5-year windows
4. For each window (stepped weekly):
   - Prepare state variables (FRED indicators + benchmark)
   - Estimate CoVaR for all banks using quantile regression
   - Forward-fill failed estimations
5. Generate panel data (window_date × bank_ric)
6. Save results for time-varying systemic risk monitoring

Configuration:
    - Window size: CONFIG.ROLLING_WINDOW_DAYS (default 1260 days = 5 years)
    - Step size: CONFIG.ROLLING_STEP_DAYS (default 5 days = weekly)
    - Start date: CONFIG.ROLLING_WINDOW_START_DATE (window end dates >= this date)

Output:
    - data/results/covar_rolling_panel.parquet: Panel data with rolling ΔCoVaR

Usage:
    uv run scripts/05b_estimate_covar_rolling.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.covar_rolling import RollingCoVaREstimator
from src.models.covar import calculate_rolling_volatility
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd
import numpy as np


def load_processed_data():
    """Load all processed data required for rolling CoVaR estimation."""
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)

    # Load bank returns
    bank_returns = load_dataframe("returns_banks_clean", PROCESSED_DATA_DIR)
    print(f"  ✓ Bank returns: {bank_returns.shape}")

    # Load benchmark returns
    benchmark_returns = load_dataframe("returns_benchmarks_clean", PROCESSED_DATA_DIR)
    print(f"  ✓ Benchmark returns: {benchmark_returns.shape}")

    # Load aligned daily data (contains market caps)
    aligned_data = load_dataframe("aligned_daily_data", PROCESSED_DATA_DIR)
    print(f"  ✓ Aligned data: {aligned_data.shape}")

    # Extract market caps
    market_cap_cols = [col for col in aligned_data.columns if col.startswith("mktcap_")]
    market_caps = aligned_data[market_cap_cols]
    market_caps.columns = [col.replace("mktcap_", "") for col in market_caps.columns]
    print(f"  ✓ Market caps: {market_caps.shape}")

    # Load FRED indicators (aligned to trading days)
    try:
        fred_indicators = load_dataframe("fred_indicators_aligned", PROCESSED_DATA_DIR)
        print(f"  ✓ FRED indicators: {fred_indicators.shape}")
    except FileNotFoundError:
        print("  ✗ FRED indicators not found - run scripts/01b_fetch_fred_data.py and scripts/02_clean_and_align_data.py")
        sys.exit(1)

    print(f"\nDate range: {bank_returns.index.min()} to {bank_returns.index.max()}")
    print(f"Banks: {len(bank_returns.columns)}")
    print(f"Benchmarks available: {', '.join(benchmark_returns.columns.tolist())}")

    return bank_returns, benchmark_returns, market_caps, fred_indicators


def main():
    """Main execution function for rolling CoVaR estimation."""

    # Check if rolling window is enabled
    if not CONFIG.ROLLING_WINDOW_ENABLED:
        print("\n⚠ Rolling window estimation is DISABLED in config.")
        print("Set CONFIG.ROLLING_WINDOW_ENABLED = True to enable.")
        return

    try:
        # 1. Load data
        bank_returns, benchmark_returns, market_caps, fred_indicators = load_processed_data()

        # 2. Select benchmark
        benchmark_name = CONFIG.BENCHMARK_BROAD
        if benchmark_name not in benchmark_returns.columns:
            print(f"\n✗ ERROR: Benchmark {benchmark_name} not found")
            print(f"Available benchmarks: {benchmark_returns.columns.tolist()}")
            sys.exit(1)

        benchmark_series = benchmark_returns[benchmark_name]
        print(f"\nUsing benchmark: {benchmark_name}")

        # 3. Calculate benchmark rolling volatility
        print("\n" + "=" * 80)
        print("CALCULATING BENCHMARK ROLLING VOLATILITY")
        print("=" * 80)

        benchmark_vol = calculate_rolling_volatility(
            benchmark_series,
            window=CONFIG.COVAR_VOL_WINDOW
        )
        print(f"  ✓ Rolling volatility calculated (window={CONFIG.COVAR_VOL_WINDOW} days)")
        print(f"  Mean volatility: {benchmark_vol.mean():.4f}")
        print(f"  Std volatility: {benchmark_vol.std():.4f}")

        # 4. Initialize rolling window estimator
        print("\n" + "=" * 80)
        print("INITIALIZING ROLLING CoVaR ESTIMATOR")
        print("=" * 80)

        estimator = RollingCoVaREstimator(
            bank_returns=bank_returns,
            market_caps=market_caps,
            fred_indicators=fred_indicators,
            benchmark_returns=benchmark_series,
            benchmark_vol=benchmark_vol,
            window_days=CONFIG.ROLLING_WINDOW_DAYS,
            step_days=CONFIG.ROLLING_STEP_DAYS,
            start_date=CONFIG.ROLLING_WINDOW_START_DATE
        )

        # 5. Run rolling estimation
        intermediate_path = RESULTS_DATA_DIR / "covar_rolling_panel_intermediate.parquet"

        panel = estimator.run_rolling_estimation(
            save_intermediate=True,
            intermediate_path=intermediate_path
        )

        # 6. Save final results
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = RESULTS_DATA_DIR / "covar_rolling_panel.parquet"

        metadata = {
            "description": "Rolling window CoVaR systemic risk metrics for GCC banks",
            "methodology": "Adrian & Brunnermeier (2016) quantile regression",
            "window_days": CONFIG.ROLLING_WINDOW_DAYS,
            "step_days": CONFIG.ROLLING_STEP_DAYS,
            "start_date": CONFIG.ROLLING_WINDOW_START_DATE,
            "var_quantile": CONFIG.COVAR_VAR_QUANTILE,
            "covar_quantile": CONFIG.COVAR_SYSTEM_QUANTILE,
            "median_quantile": CONFIG.COVAR_MEDIAN_QUANTILE,
            "benchmark": benchmark_name,
            "num_windows": panel['window_date'].nunique(),
            "num_banks": panel['bank_ric'].nunique(),
            "date_range": f"{panel['window_date'].min()} to {panel['window_date'].max()}",
            "note": "ΔCoVaR = CoVaR_5% - CoVaR_50%; more negative = higher systemic risk"
        }

        save_dataframe(panel, "covar_rolling_panel", RESULTS_DATA_DIR, metadata)
        print(f"  ✓ Saved rolling CoVaR panel: {output_path}")

        # 7. Generate summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        # Panel dimensions
        print(f"\nPanel dimensions:")
        print(f"  Windows: {panel['window_date'].nunique()}")
        print(f"  Banks: {panel['bank_ric'].nunique()}")
        print(f"  Total observations: {len(panel)}")
        print(f"  Window date range: {panel['window_date'].min()} to {panel['window_date'].max()}")

        # ΔCoVaR statistics
        print(f"\nΔCoVaR statistics:")
        print(f"  Mean: {panel['delta_covar'].mean():.4f}")
        print(f"  Std: {panel['delta_covar'].std():.4f}")
        print(f"  Min: {panel['delta_covar'].min():.4f} (highest systemic risk)")
        print(f"  25th percentile: {panel['delta_covar'].quantile(0.25):.4f}")
        print(f"  Median: {panel['delta_covar'].median():.4f}")
        print(f"  75th percentile: {panel['delta_covar'].quantile(0.75):.4f}")
        print(f"  Max: {panel['delta_covar'].max():.4f}")

        # VaR statistics
        print(f"\nVaR statistics:")
        print(f"  Mean VaR (5%): {panel['var_5pct'].mean():.4f}")
        print(f"  Std VaR (5%): {panel['var_5pct'].std():.4f}")
        print(f"  Mean VaR (50%): {panel['var_50pct'].mean():.4f}")
        print(f"  Std VaR (50%): {panel['var_50pct'].std():.4f}")

        # CoVaR statistics
        print(f"\nCoVaR statistics:")
        print(f"  Mean CoVaR (5%): {panel['covar_5pct'].mean():.4f}")
        print(f"  Std CoVaR (5%): {panel['covar_5pct'].std():.4f}")
        print(f"  Mean CoVaR (50%): {panel['covar_50pct'].mean():.4f}")
        print(f"  Std CoVaR (50%): {panel['covar_50pct'].std():.4f}")

        # Beta statistics
        print(f"\nBeta (β_i) statistics:")
        print(f"  Mean: {panel['beta_i'].mean():.4f}")
        print(f"  Std: {panel['beta_i'].std():.4f}")
        print(f"  Range: [{panel['beta_i'].min():.4f}, {panel['beta_i'].max():.4f}]")

        # Convergence statistics
        print(f"\nConvergence:")
        print(f"  Convergence rate: {panel['converged'].mean()*100:.1f}%")
        print(f"  Successful estimations: {panel['converged'].sum()}/{len(panel)}")

        # Top banks by average ΔCoVaR (most negative = highest systemic risk)
        print(f"\nTop 10 banks by average systemic risk (most negative ΔCoVaR):")
        avg_delta_covar = panel.groupby('bank_ric')['delta_covar'].mean().sort_values()
        for i, (ric, delta_covar) in enumerate(avg_delta_covar.head(10).items(), 1):
            avg_beta = panel[panel['bank_ric'] == ric]['beta_i'].mean()
            print(f"  {i:2d}. {ric:20s}  ΔCoVaR={delta_covar:8.4f}  β_i={avg_beta:7.4f}")

        # Model fit statistics
        print(f"\nModel fit (Pseudo R²):")
        print(f"  Mean Pseudo R² (CoVaR model): {panel['pseudo_r2_covar'].mean():.4f}")
        print(f"  Std Pseudo R² (CoVaR model): {panel['pseudo_r2_covar'].std():.4f}")

        # Time series analysis
        print(f"\nTime series analysis:")
        delta_covar_by_window = panel.groupby('window_date')['delta_covar'].mean()
        print(f"  Average ΔCoVaR first window: {delta_covar_by_window.iloc[0]:.4f}")
        print(f"  Average ΔCoVaR last window: {delta_covar_by_window.iloc[-1]:.4f}")
        print(f"  Change: {(delta_covar_by_window.iloc[-1] - delta_covar_by_window.iloc[0]):.4f}")

        # Identify most systemically risky periods
        print(f"\nMost systemically risky periods (lowest average ΔCoVaR):")
        top_risk_periods = delta_covar_by_window.sort_values().head(5)
        for i, (date, delta_covar) in enumerate(top_risk_periods.items(), 1):
            print(f"  {i}. {date.date()}: {delta_covar:.4f}")

        print("\n" + "=" * 80)
        print("ROLLING CoVaR ESTIMATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_path}")
        print("\nInterpretation:")
        print("  - ΔCoVaR measures marginal contribution to systemic risk")
        print("  - More negative ΔCoVaR = bank's distress increases system tail risk")
        print("  - Positive ΔCoVaR = bank provides diversification/stability")
        print("  - Time series enables monitoring of evolving systemic importance")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
