#!/usr/bin/env python3
"""Script to calculate SRISK for GCC banks.

This script implements Phase 4 of the SRISK pipeline:
1. Load processed data (aligned debt + equity)
2. Load LRMES estimates from Phase 3
3. Calculate SRISK using configured capital ratio
4. Perform decomposition analysis (size, leverage, risk effects)
5. Calculate system-wide SRISK and country aggregates
6. Generate summary statistics
7. Save results for reporting

Capital ratio can be adjusted in src/utils/config.py (CONFIG.CAPITAL_RATIO_BASEL)

Usage:
    uv run scripts/04_calculate_srisk.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.srisk import (
    calculate_srisk_multi,
    calculate_system_srisk,
    calculate_srisk_contribution,
    srisk_summary_statistics,
    calculate_srisk_by_country,
    decompose_srisk
)
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd


def load_required_data():
    """Load all required data for SRISK calculation."""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    aligned_data = load_dataframe("aligned_daily_data", PROCESSED_DATA_DIR)
    lrmes_df = load_dataframe("lrmes_estimates", RESULTS_DATA_DIR)
    bank_universe = load_dataframe("banks_universe_clean", PROCESSED_DATA_DIR)

    print(f"\nAligned data: {aligned_data.shape}")
    print(f"LRMES estimates: {len(lrmes_df)} banks")
    print(f"Bank universe: {len(bank_universe)} banks")

    # Convert LRMES from single-row to time series (broadcast to all dates)
    # Note: In the current implementation, LRMES is constant over time
    # For time-varying LRMES, this would use the full time series
    lrmes_values = dict(zip(lrmes_df["bank_ric"], lrmes_df["lrmes"]))

    # Create time series DataFrame
    lrmes_ts = pd.DataFrame(
        {inst: [lrmes_values[inst]] * len(aligned_data) for inst in lrmes_values.keys()},
        index=aligned_data.index
    )

    return aligned_data, lrmes_ts, bank_universe


def calculate_srisk_models(aligned_data, lrmes_ts, bank_universe):
    """Calculate SRISK using configured capital ratio."""
    print("\n" + "=" * 60)
    print("CALCULATING SRISK")
    print("=" * 60)
    print(f"Capital ratio: {CONFIG.CAPITAL_RATIO_BASEL * 100:.1f}%")
    print()

    # Calculate SRISK
    srisk = calculate_srisk_multi(
        aligned_data=aligned_data,
        lrmes_df=lrmes_ts,
        bank_universe=bank_universe,
        k=CONFIG.CAPITAL_RATIO_BASEL
    )

    print(f"✓ SRISK calculated for {len(srisk.columns)} banks")
    print(f"  Shape: {srisk.shape}")

    # Save SRISK time series
    save_dataframe(srisk, "srisk_timeseries", RESULTS_DATA_DIR, {
        "description": f"SRISK time series (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "crisis_threshold": CONFIG.CRISIS_THRESHOLD,
        "horizon_weeks": CONFIG.CRISIS_HORIZON_WEEKS
    })

    return srisk


def calculate_system_aggregates(srisk):
    """Calculate system-wide SRISK and contributions."""
    print("\n" + "=" * 60)
    print("SYSTEM-WIDE AGGREGATION")
    print("=" * 60)

    # System SRISK
    system_srisk = calculate_system_srisk(srisk)

    print(f"\nSystem SRISK (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%):")
    print(f"  Latest: ${system_srisk.iloc[-1] / 1e9:.2f}B")
    print(f"  Mean: ${system_srisk.mean() / 1e9:.2f}B")
    print(f"  Max: ${system_srisk.max() / 1e9:.2f}B")

    # Contributions
    contributions = calculate_srisk_contribution(srisk)

    print(f"\nTop 5 contributors (latest):")
    latest_contrib = contributions.iloc[-1].sort_values(ascending=False).head()
    for bank, pct in latest_contrib.items():
        print(f"  - {bank}: {pct:.2f}%")

    # Save system aggregates
    save_dataframe(system_srisk, "system_srisk", RESULTS_DATA_DIR, {
        "description": "System-wide SRISK (sum of positive SRISK)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })

    save_dataframe(contributions, "srisk_contributions", RESULTS_DATA_DIR, {
        "description": "Bank contributions to system SRISK (%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })

    return system_srisk, contributions


def generate_summary_statistics(srisk, bank_universe):
    """Generate summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Summary statistics
    summary = srisk_summary_statistics(srisk, bank_universe)
    print(f"\nTop 10 banks by mean SRISK:")
    print(summary.head(10)[["bank_ric", "country", "mean_srisk", "latest_srisk"]].to_string(index=False))

    save_dataframe(summary, "srisk_summary", RESULTS_DATA_DIR, {
        "description": f"SRISK summary statistics (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })

    return summary


def calculate_country_aggregates(srisk, bank_universe):
    """Calculate country-level SRISK."""
    print("\n" + "=" * 60)
    print("COUNTRY AGGREGATION")
    print("=" * 60)

    country_srisk = calculate_srisk_by_country(srisk, bank_universe)

    print(f"\nSRISK by country (latest):")
    latest_country = country_srisk.iloc[-1].sort_values(ascending=False)
    for country, srisk_val in latest_country.items():
        print(f"  - {country}: ${srisk_val / 1e9:.2f}B")

    save_dataframe(country_srisk, "srisk_by_country", RESULTS_DATA_DIR, {
        "description": f"SRISK aggregated by country (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })

    return country_srisk


def perform_decomposition_analysis(aligned_data, lrmes_ts, bank_universe):
    """Perform decomposition analysis for top banks."""
    print("\n" + "=" * 60)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 60)
    print("Analyzing top 5 banks by latest SRISK...")

    # Load SRISK to identify top banks
    srisk = load_dataframe("srisk_timeseries", RESULTS_DATA_DIR)
    top_banks = srisk.iloc[-1].sort_values(ascending=False).head(5).index.tolist()

    decomp_results = []

    for bank_ric in top_banks:
        debt_col = f"fund_{bank_ric}_TotLiab"
        equity_col = f"mktcap_{bank_ric}"

        if debt_col in aligned_data.columns and equity_col in aligned_data.columns:
            debt = aligned_data[debt_col]
            equity = aligned_data[equity_col]
            lrmes = lrmes_ts[bank_ric]

            decomp = decompose_srisk(debt, equity, lrmes, k=CONFIG.CAPITAL_RATIO_BASEL)

            # Summarize
            decomp_results.append({
                "bank_ric": bank_ric,
                "avg_size_effect": decomp["size_effect"].mean(),
                "avg_leverage_effect": decomp["leverage_effect"].mean(),
                "avg_risk_effect": decomp["risk_effect"].mean(),
                "total_change": decomp["srisk"].iloc[-1] - decomp["srisk"].iloc[0]
            })

    decomp_df = pd.DataFrame(decomp_results)
    print(f"\nDecomposition (average daily effects):")
    print(decomp_df.to_string(index=False))

    save_dataframe(decomp_df, "srisk_decomposition", RESULTS_DATA_DIR, {
        "description": "SRISK decomposition (size, leverage, risk effects)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK: SRISK CALCULATION")
    print("=" * 60)

    try:
        # 1. Load data
        aligned_data, lrmes_ts, bank_universe = load_required_data()

        # 2. Calculate SRISK
        srisk = calculate_srisk_models(aligned_data, lrmes_ts, bank_universe)

        # 3. System aggregates
        system_srisk, contributions = calculate_system_aggregates(srisk)

        # 4. Summary statistics
        summary = generate_summary_statistics(srisk, bank_universe)

        # 5. Country aggregates
        country_srisk = calculate_country_aggregates(srisk, bank_universe)

        # 6. Decomposition
        perform_decomposition_analysis(aligned_data, lrmes_ts, bank_universe)

        # Final summary
        print("\n" + "=" * 60)
        print("✓ SRISK CALCULATION COMPLETED")
        print("=" * 60)
        print(f"\nCapital ratio used: {CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%")
        print(f"Crisis threshold: {CONFIG.CRISIS_THRESHOLD*100:.1f}% market decline")
        print(f"Horizon: {CONFIG.CRISIS_HORIZON_WEEKS} weeks")
        print(f"\nResults saved to: {RESULTS_DATA_DIR}")
        print("\nKey outputs:")
        print("  - srisk_timeseries.parquet - SRISK time series")
        print("  - system_srisk.parquet - System-wide SRISK")
        print("  - srisk_contributions.parquet - Bank contributions (%)")
        print("  - srisk_summary.parquet - Summary statistics")
        print("  - srisk_by_country.parquet - Country aggregates")
        print("  - srisk_decomposition.parquet - Decomposition analysis")

        print("\nNext step:")
        print("  - Review results in data/results/")
        print("  - Generate reports and visualizations")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
