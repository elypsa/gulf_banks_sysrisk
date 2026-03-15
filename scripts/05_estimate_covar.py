#!/usr/bin/env python3
"""Script to estimate CoVaR (Conditional Value at Risk) for systemic risk measurement.

This script implements the Adrian & Brunnermeier (2016) CoVaR methodology using
quantile regression to measure each bank's marginal contribution to systemic risk.

Methodology:
1. Load processed returns, market caps, and state variables (FRED + benchmark)
2. Calculate benchmark rolling volatility
3. Prepare lagged state variables
4. For each bank:
   - Calculate system return (excluding that bank)
   - Estimate bank VaR at 5% and 50% quantiles
   - Estimate system CoVaR conditional on bank return
   - Calculate ΔCoVaR = β_i × (VaR_5% - VaR_50%)
5. Rank banks by ΔCoVaR (most negative = highest systemic risk)

Output:
    - data/results/covar_results.parquet: CoVaR metrics for all banks

Usage:
    uv run scripts/05_estimate_covar.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.covar import (
    calculate_rolling_volatility,
    prepare_state_variables,
    estimate_covar_all_banks
)
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd


def load_processed_data():
    """Load all processed data required for CoVaR estimation."""
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

    return bank_returns, benchmark_returns, market_caps, fred_indicators


def main():
    """Main execution function for CoVaR estimation."""
    print("\n" + "=" * 80)
    print("CoVaR SYSTEMIC RISK ESTIMATION")
    print("=" * 80)

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
        print(f"\n  Using benchmark: {benchmark_name}")

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

        # 4. Prepare state variables
        print("\n" + "=" * 80)
        print("PREPARING STATE VARIABLES")
        print("=" * 80)

        state_variables = prepare_state_variables(
            fred_indicators=fred_indicators,
            benchmark_returns=benchmark_series,
            benchmark_vol=benchmark_vol
        )

        print(f"  ✓ State variables prepared: {state_variables.shape}")
        print(f"  Variables: {', '.join(state_variables.columns.tolist())}")
        print(f"  Date range: {state_variables.index.min()} to {state_variables.index.max()}")
        print(f"\n  Note: All state variables are lagged by 1 day (t-1)")

        # Check for missing values
        missing_pct = state_variables.isna().sum() / len(state_variables) * 100
        if missing_pct.sum() > 0:
            print(f"\n  Missing values:")
            for col in state_variables.columns:
                if missing_pct[col] > 0:
                    print(f"    {col}: {missing_pct[col]:.2f}%")

        # 5. Estimate CoVaR for all banks
        covar_results = estimate_covar_all_banks(
            bank_returns=bank_returns,
            market_caps=market_caps,
            state_variables=state_variables,
            verbose=True
        )

        # 6. Save results
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        metadata = {
            "description": "CoVaR systemic risk metrics for GCC banks",
            "methodology": "Adrian & Brunnermeier (2016) quantile regression",
            "var_quantile": CONFIG.COVAR_VAR_QUANTILE,
            "covar_quantile": CONFIG.COVAR_SYSTEM_QUANTILE,
            "median_quantile": CONFIG.COVAR_MEDIAN_QUANTILE,
            "state_variables": list(state_variables.columns),
            "benchmark": benchmark_name,
            "num_banks": len(covar_results),
            "date_range": f"{state_variables.index.min()} to {state_variables.index.max()}",
            "note": "ΔCoVaR = CoVaR_5% - CoVaR_50%; more negative = higher systemic risk"
        }

        save_dataframe(covar_results, "covar_results", RESULTS_DATA_DIR, metadata)

        output_path = RESULTS_DATA_DIR / "covar_results.parquet"
        print(f"  ✓ Results saved to: {output_path}")

        # 7. Summary statistics
        print("\n" + "=" * 80)
        print("CoVaR SUMMARY STATISTICS")
        print("=" * 80)

        print(f"\nΔCoVaR statistics:")
        print(f"  Mean: {covar_results['delta_covar'].mean():.4f}")
        print(f"  Std: {covar_results['delta_covar'].std():.4f}")
        print(f"  Min: {covar_results['delta_covar'].min():.4f} (highest systemic risk)")
        print(f"  25th percentile: {covar_results['delta_covar'].quantile(0.25):.4f}")
        print(f"  Median: {covar_results['delta_covar'].median():.4f}")
        print(f"  75th percentile: {covar_results['delta_covar'].quantile(0.75):.4f}")
        print(f"  Max: {covar_results['delta_covar'].max():.4f}")

        print(f"\nVaR statistics:")
        print(f"  Mean VaR (5%): {covar_results['var_5pct'].mean():.4f}%")
        print(f"  Mean VaR (50%): {covar_results['var_50pct'].mean():.4f}%")

        print(f"\nTop 10 banks by systemic risk (most negative ΔCoVaR):")
        top_10 = covar_results.head(10)
        for i, row in enumerate(top_10.itertuples(), 1):
            print(f"  {i:2d}. {row.bank_ric:20s}  ΔCoVaR={row.delta_covar:8.4f}  β_i={row.beta_i:7.4f}")

        print(f"\nConvergence:")
        print(f"  Converged: {covar_results['converged'].sum()}/{len(covar_results)} banks")
        print(f"  Convergence rate: {covar_results['converged'].mean()*100:.1f}%")

        print("\n" + "=" * 80)
        print("CoVaR ESTIMATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {output_path}")
        print("\nInterpretation:")
        print("  - ΔCoVaR measures marginal contribution to systemic risk")
        print("  - More negative ΔCoVaR = bank's distress increases system tail risk")
        print("  - Positive ΔCoVaR = bank provides diversification/stability")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
