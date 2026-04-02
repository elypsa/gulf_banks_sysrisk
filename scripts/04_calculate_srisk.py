#!/usr/bin/env python3
"""Script to calculate SRISK for GCC banks.

This script implements Phase 4 of the SRISK pipeline:
1. Load processed data (aligned debt + equity)
2. Load time-varying LRMES from rolling window estimation (Phase 3b)
3. Calculate SRISK using configured capital ratio
4. Perform decomposition analysis (size, leverage, risk effects)
5. Calculate system-wide SRISK and country aggregates
6. Generate summary statistics
7. Save results for reporting

Requirements:
    - Must run scripts/03b_estimate_garch_dcc_rolling.py first to generate LRMES panel
    - Uses time-varying LRMES from lrmes_rolling_panel.parquet
    - Capital ratio can be adjusted in src/utils/config.py (CONFIG.CAPITAL_RATIO_BASEL)

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
    """Load all required data for SRISK calculation.

    Loads time-varying LRMES from rolling window estimation (Phase 3b).
    Requires lrmes_rolling_panel.parquet to exist in data/results/.
    """
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    aligned_data = load_dataframe("aligned_daily_data", PROCESSED_DATA_DIR)
    bank_universe = load_dataframe("banks_universe_clean", PROCESSED_DATA_DIR)

    print(f"\nAligned data: {aligned_data.shape}")
    print(f"Bank universe: {len(bank_universe)} banks")

    # Load rolling LRMES panel (required)
    rolling_panel_path = RESULTS_DATA_DIR / "lrmes_rolling_panel.parquet"

    if not rolling_panel_path.exists():
        raise FileNotFoundError(
            f"\n✗ ERROR: Rolling LRMES panel not found at: {rolling_panel_path}\n"
            f"Please run scripts/03b_estimate_garch_dcc_rolling.py first to generate LRMES panel."
        )

    print("\n✓ Loading time-varying LRMES from rolling window estimation")

    # Load rolling panel data
    lrmes_panel = pd.read_parquet(rolling_panel_path)
    print(f"  Panel shape: {lrmes_panel.shape}")
    print(f"  Windows: {lrmes_panel['window_date'].nunique()}")
    print(f"  Banks: {lrmes_panel['bank_ric'].nunique()}")
    print(f"  Window date range: {lrmes_panel['window_date'].min()} to {lrmes_panel['window_date'].max()}")

    # Pivot to (date × bank_ric) format
    lrmes_ts = lrmes_panel.pivot(
        index='window_date',
        columns='bank_ric',
        values='lrmes'
    )

    # Reindex to match aligned_data dates and forward-fill
    # This propagates weekly LRMES estimates to daily frequency
    lrmes_ts = lrmes_ts.reindex(aligned_data.index).ffill()

    # Backward-fill any leading NaNs (before first window)
    lrmes_ts = lrmes_ts.bfill()

    # Check coverage
    missing_banks = set(bank_universe['bank_ric']) - set(lrmes_ts.columns)
    if missing_banks:
        print(f"\n⚠ WARNING: {len(missing_banks)} banks missing from rolling LRMES panel:")
        print(f"  {', '.join(list(missing_banks)[:5])}" +
              (f" ... and {len(missing_banks)-5} more" if len(missing_banks) > 5 else ""))

    print(f"\n✓ LRMES time series shape: {lrmes_ts.shape}")

    return aligned_data, lrmes_ts, bank_universe


def calculate_srisk_models(aligned_data, lrmes_ts, bank_universe):
    """Calculate time-varying SRISK using configured capital ratio and rolling LRMES."""
    print("\n" + "=" * 60)
    print("CALCULATING TIME-VARYING SRISK")
    print("=" * 60)
    print(f"Capital ratio: {CONFIG.CAPITAL_RATIO_BASEL * 100:.1f}%")
    print(f"LRMES: Time-varying from rolling window estimation")
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
        "description": f"Time-varying SRISK from rolling window LRMES (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "crisis_threshold": CONFIG.CRISIS_THRESHOLD,
        "horizon_weeks": CONFIG.CRISIS_HORIZON_WEEKS,
        "lrmes_type": "rolling_window"
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
        "description": "System-wide time-varying SRISK (sum of positive SRISK)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "lrmes_type": "rolling_window"
    })

    save_dataframe(contributions, "srisk_contributions", RESULTS_DATA_DIR, {
        "description": "Bank contributions to time-varying system SRISK (%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "lrmes_type": "rolling_window"
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
        "description": f"Time-varying SRISK summary statistics (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "lrmes_type": "rolling_window"
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
        "description": f"Time-varying SRISK aggregated by country (k={CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "lrmes_type": "rolling_window"
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
        "description": "Time-varying SRISK decomposition (size, leverage, risk effects)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL,
        "lrmes_type": "rolling_window"
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
        # system_srisk, contributions = calculate_system_aggregates(srisk)

        # 4. Summary statistics
        # summary = generate_summary_statistics(srisk, bank_universe)

        # 5. Country aggregates
        # country_srisk = calculate_country_aggregates(srisk, bank_universe)

        # 6. Decomposition
        # perform_decomposition_analysis(aligned_data, lrmes_ts, bank_universe)

        # Final summary
        print("\n" + "=" * 60)
        print("✓ TIME-VARYING SRISK CALCULATION COMPLETED")
        print("=" * 60)
        print(f"\nParameters:")
        print(f"  - Capital ratio: {CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%")
        print(f"  - Crisis threshold: {CONFIG.CRISIS_THRESHOLD*100:.1f}% market decline")
        print(f"  - Horizon: {CONFIG.CRISIS_HORIZON_WEEKS} weeks")
        print(f"  - LRMES: Time-varying (rolling window estimation)")
        print(f"\nResults saved to: {RESULTS_DATA_DIR}")
        print("\nKey outputs (all time-varying):")
        print("  - srisk_timeseries.parquet - SRISK time series")
        # print("  - system_srisk.parquet - System-wide SRISK")
        # print("  - srisk_contributions.parquet - Bank contributions (%)")
        # print("  - srisk_summary.parquet - Summary statistics")
        # print("  - srisk_by_country.parquet - Country aggregates")
        # print("  - srisk_decomposition.parquet - Decomposition analysis")

        print("\nNext steps:")
        print("  - Review results in data/results/")
        print("  - Generate reports and visualizations")
        print("  - Analyze time variation in SRISK estimates")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
