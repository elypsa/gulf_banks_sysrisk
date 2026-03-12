#!/usr/bin/env python3
"""Script to calculate SRISK for GCC banks.

This script implements Phase 4 of the SRISK pipeline:
1. Load processed data (aligned debt + equity)
2. Load LRMES estimates from Phase 3
3. Calculate SRISK with dual capital ratios (k=8% Basel, k=5.5% IFRS)
4. Perform decomposition analysis (size, leverage, risk effects)
5. Calculate system-wide SRISK and country aggregates
6. Generate summary statistics
7. Save results for reporting

Usage:
    uv run scripts/04_calculate_srisk.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.srisk import (
    calculate_srisk_dual_capital_ratios,
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
    lrmes_values = dict(zip(lrmes_df["instrument"], lrmes_df["lrmes"]))

    # Create time series DataFrame
    lrmes_ts = pd.DataFrame(
        {inst: [lrmes_values[inst]] * len(aligned_data) for inst in lrmes_values.keys()},
        index=aligned_data.index
    )

    return aligned_data, lrmes_ts, bank_universe


def calculate_srisk_models(aligned_data, lrmes_ts, bank_universe):
    """Calculate SRISK with dual capital ratios."""
    print("\n" + "=" * 60)
    print("CALCULATING SRISK")
    print("=" * 60)
    print(f"Capital ratio (Basel): {CONFIG.CAPITAL_RATIO_BASEL * 100:.1f}%")
    print(f"Capital ratio (IFRS): {CONFIG.CAPITAL_RATIO_IFRS * 100:.1f}%")
    print()

    # Calculate with both capital ratios
    srisk_basel, srisk_ifrs = calculate_srisk_dual_capital_ratios(
        aligned_data=aligned_data,
        lrmes_df=lrmes_ts,
        bank_universe=bank_universe,
        k_basel=CONFIG.CAPITAL_RATIO_BASEL,
        k_ifrs=CONFIG.CAPITAL_RATIO_IFRS
    )

    print(f"✓ SRISK calculated for {len(srisk_basel.columns)} banks")
    print(f"  - Basel (k=8%): {srisk_basel.shape}")
    print(f"  - IFRS (k=5.5%): {srisk_ifrs.shape}")

    # Save SRISK time series
    save_dataframe(srisk_basel, "srisk_basel", RESULTS_DATA_DIR, {
        "description": "SRISK time series (Basel k=8%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_BASEL
    })

    save_dataframe(srisk_ifrs, "srisk_ifrs", RESULTS_DATA_DIR, {
        "description": "SRISK time series (IFRS k=5.5%)",
        "capital_ratio": CONFIG.CAPITAL_RATIO_IFRS
    })

    return srisk_basel, srisk_ifrs


def calculate_system_aggregates(srisk_basel, srisk_ifrs):
    """Calculate system-wide SRISK and contributions."""
    print("\n" + "=" * 60)
    print("SYSTEM-WIDE AGGREGATION")
    print("=" * 60)

    # System SRISK
    system_basel = calculate_system_srisk(srisk_basel)
    system_ifrs = calculate_system_srisk(srisk_ifrs)

    print(f"\nLatest System SRISK:")
    print(f"  - Basel (k=8%): ${system_basel.iloc[-1] / 1e9:.2f}B")
    print(f"  - IFRS (k=5.5%): ${system_ifrs.iloc[-1] / 1e9:.2f}B")
    print(f"  - Difference: ${(system_basel.iloc[-1] - system_ifrs.iloc[-1]) / 1e9:.2f}B")

    # Contributions
    contrib_basel = calculate_srisk_contribution(srisk_basel)
    contrib_ifrs = calculate_srisk_contribution(srisk_ifrs)

    print(f"\nTop 5 contributors (Basel, latest):")
    latest_contrib = contrib_basel.iloc[-1].sort_values(ascending=False).head()
    for bank, pct in latest_contrib.items():
        print(f"  - {bank}: {pct:.2f}%")

    # Save system aggregates
    system_df = pd.DataFrame({
        "basel_k8": system_basel,
        "ifrs_k5.5": system_ifrs
    })
    save_dataframe(system_df, "system_srisk", RESULTS_DATA_DIR, {
        "description": "System-wide SRISK (sum of positive SRISK)"
    })

    save_dataframe(contrib_basel, "contributions_basel", RESULTS_DATA_DIR, {
        "description": "Bank contributions to system SRISK (%)"
    })

    return system_basel, system_ifrs, contrib_basel


def generate_summary_statistics(srisk_basel, srisk_ifrs, bank_universe):
    """Generate summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Basel summary
    summary_basel = srisk_summary_statistics(srisk_basel, bank_universe)
    print(f"\nTop 10 banks by mean SRISK (Basel):")
    print(summary_basel.head(10)[["bank_ric", "country", "mean_srisk", "latest_srisk"]].to_string(index=False))

    save_dataframe(summary_basel, "srisk_summary_basel", RESULTS_DATA_DIR, {
        "description": "SRISK summary statistics (Basel k=8%)"
    })

    # IFRS summary
    summary_ifrs = srisk_summary_statistics(srisk_ifrs, bank_universe)
    save_dataframe(summary_ifrs, "srisk_summary_ifrs", RESULTS_DATA_DIR, {
        "description": "SRISK summary statistics (IFRS k=5.5%)"
    })

    return summary_basel, summary_ifrs


def calculate_country_aggregates(srisk_basel, srisk_ifrs, bank_universe):
    """Calculate country-level SRISK."""
    print("\n" + "=" * 60)
    print("COUNTRY AGGREGATION")
    print("=" * 60)

    country_basel = calculate_srisk_by_country(srisk_basel, bank_universe)
    country_ifrs = calculate_srisk_by_country(srisk_ifrs, bank_universe)

    print(f"\nSRISK by country (Basel, latest):")
    latest_country = country_basel.iloc[-1].sort_values(ascending=False)
    for country, srisk in latest_country.items():
        print(f"  - {country}: ${srisk / 1e9:.2f}B")

    save_dataframe(country_basel, "srisk_by_country_basel", RESULTS_DATA_DIR, {
        "description": "SRISK aggregated by country (Basel)"
    })

    save_dataframe(country_ifrs, "srisk_by_country_ifrs", RESULTS_DATA_DIR, {
        "description": "SRISK aggregated by country (IFRS)"
    })

    return country_basel


def perform_decomposition_analysis(aligned_data, lrmes_ts, bank_universe):
    """Perform decomposition analysis for top banks."""
    print("\n" + "=" * 60)
    print("DECOMPOSITION ANALYSIS")
    print("=" * 60)
    print("Analyzing top 5 banks by latest SRISK...")

    # Load SRISK to identify top banks
    srisk_basel = load_dataframe("srisk_basel", RESULTS_DATA_DIR)
    top_banks = srisk_basel.iloc[-1].sort_values(ascending=False).head(5).index.tolist()

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
        "description": "SRISK decomposition (size, leverage, risk effects)"
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
        srisk_basel, srisk_ifrs = calculate_srisk_models(aligned_data, lrmes_ts, bank_universe)

        # 3. System aggregates
        system_basel, system_ifrs, contrib_basel = calculate_system_aggregates(srisk_basel, srisk_ifrs)

        # 4. Summary statistics
        summary_basel, summary_ifrs = generate_summary_statistics(srisk_basel, srisk_ifrs, bank_universe)

        # 5. Country aggregates
        country_basel = calculate_country_aggregates(srisk_basel, srisk_ifrs, bank_universe)

        # 6. Decomposition
        perform_decomposition_analysis(aligned_data, lrmes_ts, bank_universe)

        # Final summary
        print("\n" + "=" * 60)
        print("✓ SRISK CALCULATION COMPLETED")
        print("=" * 60)
        print(f"\nResults saved to: {RESULTS_DATA_DIR}")
        print("\nKey outputs:")
        print("  - srisk_basel.parquet - SRISK time series (Basel k=8%)")
        print("  - srisk_ifrs.parquet - SRISK time series (IFRS k=5.5%)")
        print("  - system_srisk.parquet - System-wide SRISK")
        print("  - srisk_summary_basel.parquet - Summary statistics")
        print("  - srisk_by_country_basel.parquet - Country aggregates")
        print("  - srisk_decomposition.parquet - Decomposition analysis")

        print("\nNext step:")
        print("  - Review results in data/results/")
        print("  - Generate reports (Phase 5 - coming soon)")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
