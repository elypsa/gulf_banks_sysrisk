#!/usr/bin/env python3
"""Script to clean and align raw data for SRISK calculation.

This script:
1. Loads raw data from data/raw/
2. Creates GCC trading calendar (handles UAE transition + Islamic holidays)
3. Calculates log returns for GARCH estimation
4. Forward-fills quarterly fundamentals to daily frequency
5. Converts all values to USD
6. Aligns all data sources to common trading days
7. Filters banks by data quality
8. Saves processed data to data/processed/

Usage:
    uv run scripts/02_clean_and_align_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_dataframe,
    save_dataframe,
    generate_missing_data_report,
    print_data_quality_summary
)
from src.data.calendar import create_gcc_holiday_calendar, align_to_trading_days
from src.data.processing import (
    align_and_merge_data,
    filter_banks_by_data_quality
)
from src.utils.config import CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def load_raw_data():
    """Load all raw data files."""
    print("\n" + "=" * 60)
    print("LOADING RAW DATA")
    print("=" * 60)

    universe = load_dataframe("gcc_banks_universe", RAW_DATA_DIR)
    fundamentals = load_dataframe("fundamentals_quarterly", RAW_DATA_DIR)
    market_data = load_dataframe("prices_daily", RAW_DATA_DIR)
    benchmarks = load_dataframe("benchmarks_daily", RAW_DATA_DIR)
    # fx_rates = load_dataframe("fx_rates_daily", RAW_DATA_DIR)  # DISABLED: Using LSEG USD conversion
    holidays_lseg = load_dataframe("holidays_calendar", RAW_DATA_DIR)

    # Load FRED indicators (optional - may not exist yet)
    try:
        fred_indicators = load_dataframe("fred_indicators", RAW_DATA_DIR)
        print("  ✓ FRED indicators loaded")
    except FileNotFoundError:
        print("  ⚠ FRED indicators not found - run scripts/01b_fetch_fred_data.py first")
        fred_indicators = None

    return universe, fundamentals, market_data, benchmarks, holidays_lseg, fred_indicators


def create_master_calendar(holidays_lseg):
    """Create comprehensive GCC holiday calendar."""
    print("\n" + "=" * 60)
    print("CREATING GCC MASTER CALENDAR")
    print("=" * 60)

    # Create calendar from 2010 to present
    holidays = create_gcc_holiday_calendar(
        start_date=CONFIG.DATA_START_DATE,
        lseg_holidays=holidays_lseg
    )

    # Save
    metadata = {
        "description": "Comprehensive GCC holiday calendar",
        "sources": ["LSEG API (2016+)", "Hijri reconstruction", "National holidays"],
        "start_date": CONFIG.HOLIDAYS_START_DATE
    }
    save_dataframe(holidays, "gcc_holidays_full", PROCESSED_DATA_DIR, metadata)

    return holidays


def create_returns_chart(bank_returns, bench_returns, bank_universe, output_dir):
    """Create multi-panel chart showing returns for benchmark and banks (5 banks per panel).

    Args:
        bank_returns: DataFrame with bank returns (columns = bank RICs).
        bench_returns: DataFrame with benchmark returns.
        bank_universe: DataFrame with bank metadata (bank_ric, country_chain).
        output_dir: Directory to save the chart.
    """
    # Use the broad benchmark
    main_benchmark_ric = CONFIG.BENCHMARK_BROAD

    # Get all bank RICs that are in the returns data
    all_bank_rics = [ric for ric in bank_returns.columns if ric in bank_universe['bank_ric'].values]

    # Split banks into groups of 5
    banks_per_panel = 5
    bank_groups = [all_bank_rics[i:i + banks_per_panel]
                   for i in range(0, len(all_bank_rics), banks_per_panel)]

    # Create figure with subplots (1 for benchmark + panels for bank groups)
    n_bank_panels = len(bank_groups)
    n_panels = 1 + n_bank_panels
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 2.5 * n_panels), sharex=True)

    if n_panels == 1:
        axes = [axes]

    # Panel 0: Main benchmark
    ax = axes[0]
    if main_benchmark_ric in bench_returns.columns:
        bench_returns[main_benchmark_ric].plot(ax=ax, color='black', linewidth=1.5, label=main_benchmark_ric)
    ax.set_ylabel('Return (%)', fontsize=10)
    ax.set_title(f'Benchmark: {main_benchmark_ric}', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Panels 1+: Banks (5 per panel)
    for i, bank_group in enumerate(bank_groups, start=1):
        ax = axes[i]

        # Plot each bank in this group
        for bank_ric in bank_group:
            bank_returns[bank_ric].plot(ax=ax, linewidth=1.0, alpha=0.8, label=bank_ric)

        ax.set_ylabel('Return (%)', fontsize=10)
        panel_num = i
        ax.set_title(f'Banks {(i-1)*banks_per_panel + 1}-{(i-1)*banks_per_panel + len(bank_group)} (Panel {panel_num})',
                     fontsize=11, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=1)

    # Format x-axis (only bottom panel)
    axes[-1].set_xlabel('Date', fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'returns_by_country.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Returns chart saved to: {output_path}")
    print(f"  - Panels: {n_panels} (1 benchmark + {n_bank_panels} bank panels)")
    print(f"  - Total banks plotted: {len(all_bank_rics)}")
    print(f"  - Banks per panel: {banks_per_panel} (last panel: {len(bank_groups[-1])} banks)")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK DATA CLEANER & ALIGNER")
    print("=" * 60)

    try:
        # 1. Load raw data
        universe, fundamentals, market_data, benchmarks, holidays_lseg, fred_indicators = load_raw_data()

        # 2. Create master calendar
        holidays = create_master_calendar(holidays_lseg)

        # 3. Align and merge all data
        print("\n" + "=" * 60)
        print("ALIGNING AND MERGING DATA (Already in USD from LSEG)")
        print("=" * 60)

        aligned = align_and_merge_data(
            fundamentals=fundamentals,
            market_data=market_data,
            benchmarks=benchmarks,
            fx_rates=None,  # Not needed - data already in USD
            bank_universe=universe,
            convert_usd=False,  # Already converted by LSEG
            interpolate_prices=True,  # Fill small gaps (≤2 days) in prices
            max_interpolation_gap=2  # Maximum gap size to interpolate
        )

        # 4. Align to GCC trading days (remove holidays)
        print("\n" + "=" * 60)
        print("FILTERING TO GCC TRADING DAYS")
        print("=" * 60)

        aligned_trading = align_to_trading_days(
            aligned,
            calendar="GCC",
            method="drop",
            holiday_df=holidays
        )

        print(f"✓ Removed {len(aligned) - len(aligned_trading)} non-trading days")
        print(f"  Final: {len(aligned_trading)} trading days")

        # 5. Run data quality checks (without filtering yet)
        print("\n" + "=" * 60)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 60)

        # Print summary to console
        print_data_quality_summary(aligned_trading, universe)

        # Run quality filter to get pass/fail report (but don't apply filter yet)
        print("\nEvaluating quality criteria...")
        clean_data, clean_universe, quality_report = filter_banks_by_data_quality(
            aligned_trading,
            universe,
            min_trading_days=CONFIG.MIN_TRADING_DAYS,
            max_missing_pct=0.20,
            return_quality_report=True
        )

        # 6. Generate visualizations WITH quality information
        print("\n" + "=" * 60)
        print("GENERATING DATA QUALITY VISUALIZATIONS")
        print("=" * 60)

        # Create output directory for visualizations
        viz_dir = PROCESSED_DATA_DIR / "data_quality_plots"

        # Generate detailed visualizations with pass/fail indicators
        generate_missing_data_report(
            aligned_trading,
            universe,
            viz_dir,
            quality_report=quality_report,
            show_plots=False
        )

        print("\n✓ Visualizations organized into:")
        print(f"  - Passed banks: {viz_dir}/passed/")
        print(f"  - Dropped banks: {viz_dir}/dropped/")

        # 7. Filter outliers in bank returns
        print("\n" + "=" * 60)
        print("FILTERING OUTLIERS IN BANK RETURNS")
        print("=" * 60)

        # Extract bank return columns
        bank_ret_cols = [col for col in clean_data.columns if col.startswith("ret_")]

        # Calculate percentiles across ALL bank returns (pooled)
        all_returns = clean_data[bank_ret_cols].values.flatten()
        all_returns_clean = all_returns[~pd.isna(all_returns)]

        p1 = np.percentile(all_returns_clean, 1)
        p99 = np.percentile(all_returns_clean, 99)

        print(f"\nOutlier thresholds (across all banks):")
        print(f"  - 1st percentile: {p1:.4f}%")
        print(f"  - 99th percentile: {p99:.4f}%")

        # Count outliers before filtering
        outliers_below = (clean_data[bank_ret_cols] < p1).sum().sum()
        outliers_above = (clean_data[bank_ret_cols] > p99).sum().sum()
        total_outliers = outliers_below + outliers_above
        total_observations = (~clean_data[bank_ret_cols].isna()).sum().sum()

        print(f"\nOutliers detected:")
        print(f"  - Below 1st percentile: {outliers_below:,} observations")
        print(f"  - Above 99th percentile: {outliers_above:,} observations")
        print(f"  - Total outliers: {total_outliers:,} / {total_observations:,} ({100*total_outliers/total_observations:.2f}%)")

        # Replace outliers with NaN
        for col in bank_ret_cols:
            clean_data.loc[clean_data[col] < p1, col] = np.nan
            clean_data.loc[clean_data[col] > p99, col] = np.nan

        print(f"\n✓ Outliers replaced with NaN")
        print(f"  Note: NaNs will be dropped in pairwise GARCH-DCC estimation")

        # 8. Save processed data
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)

        # Save aligned data
        metadata_aligned = {
            "description": "Aligned daily data (all sources merged, USD from LSEG, outliers filtered)",
            "trading_days": len(clean_data),
            "num_banks": clean_universe["bank_ric"].nunique(),
            "start_date": str(clean_data.index.min().date()),
            "end_date": str(clean_data.index.max().date()),
            "currency": "USD (converted by LSEG on-the-fly)",
            "quality_filtered": True,
            "outlier_filtered": True,
            "outlier_thresholds": f"p1={p1:.4f}%, p99={p99:.4f}%",
            "min_trading_days": CONFIG.MIN_TRADING_DAYS
        }
        save_dataframe(clean_data, "aligned_daily_data", PROCESSED_DATA_DIR, metadata_aligned)

        # Save clean universe
        metadata_universe = {
            "description": "Quality-filtered bank universe",
            "num_banks": len(clean_universe),
            "quality_criteria": f"min_{CONFIG.MIN_TRADING_DAYS}_days"
        }
        save_dataframe(clean_universe, "banks_universe_clean", PROCESSED_DATA_DIR, metadata_universe)

        # Extract bank returns separately (for GARCH estimation)
        bank_ret_cols = [col for col in clean_data.columns if col.startswith("ret_")]
        bank_returns = clean_data[bank_ret_cols]

        # Rename columns to remove 'ret_' prefix for cleaner access
        bank_returns.columns = [col.replace("ret_", "") for col in bank_returns.columns]

        metadata_bank_returns = {
            "description": "Bank log returns for GARCH-DCC estimation (outliers filtered)",
            "trading_days": len(bank_returns),
            "num_banks": len(bank_returns.columns),
            "return_type": "log",
            "outlier_filtered": True,
            "outlier_thresholds": f"p1={p1:.4f}%, p99={p99:.4f}%",
            "ready_for_garch": True
        }
        save_dataframe(bank_returns, "returns_banks_clean", PROCESSED_DATA_DIR, metadata_bank_returns)

        # Extract benchmark returns separately (for market index)
        bench_ret_cols = [col for col in clean_data.columns if col.startswith("bench_ret_")]
        bench_returns = clean_data[bench_ret_cols]

        # Rename columns to remove 'bench_ret_' prefix for cleaner access
        bench_returns.columns = [col.replace("bench_ret_", "") for col in bench_returns.columns]

        metadata_bench_returns = {
            "description": "Benchmark log returns (market indices)",
            "trading_days": len(bench_returns),
            "num_benchmarks": len(bench_returns.columns),
            "return_type": "log",
            "instruments": list(bench_returns.columns)
        }
        save_dataframe(bench_returns, "returns_benchmarks_clean", PROCESSED_DATA_DIR, metadata_bench_returns)

        # 9. Process FRED indicators (if available)
        if fred_indicators is not None:
            print("\n" + "=" * 60)
            print("PROCESSING FRED INDICATORS")
            print("=" * 60)

            # Align FRED data to GCC trading days
            print(f"\nAligning FRED data to GCC trading days...")
            print(f"  Original FRED data: {len(fred_indicators)} observations")

            # Reindex to GCC trading days (will create NaN for missing dates)
            fred_aligned = fred_indicators.reindex(clean_data.index)

            # Count NaNs before forward fill
            nans_before = fred_aligned.isna().sum()
            print(f"\n  NaN values after alignment (before forward-fill):")
            for col in fred_aligned.columns:
                print(f"    {col}: {nans_before[col]} NaNs ({100*nans_before[col]/len(fred_aligned):.2f}%)")

            # Forward-fill to propagate Friday values to Sunday trading days
            print(f"\n  Applying forward-fill (Friday → Sunday for GCC trading days)...")
            fred_aligned = fred_aligned.ffill()

            # Count remaining NaNs after forward fill
            nans_after = fred_aligned.isna().sum()
            if nans_after.sum() > 0:
                print(f"\n  Remaining NaN values after forward-fill:")
                for col in fred_aligned.columns:
                    if nans_after[col] > 0:
                        print(f"    {col}: {nans_after[col]} NaNs ({100*nans_after[col]/len(fred_aligned):.2f}%)")
            else:
                print(f"  ✓ No remaining NaN values after forward-fill")

            # Save aligned FRED indicators
            metadata_fred = {
                "description": "FRED systemic risk indicators aligned to GCC trading days",
                "series": {
                    "VIXCLS": "CBOE Volatility Index",
                    "T10Y3M": "10Y-3M Treasury spread"
                },
                "trading_days": len(fred_aligned),
                "start_date": str(fred_aligned.index.min().date()),
                "end_date": str(fred_aligned.index.max().date()),
                "alignment": "Reindexed to GCC trading days with forward-fill",
                "source": "Federal Reserve Economic Data (FRED)"
            }
            save_dataframe(fred_aligned, "fred_indicators_aligned", PROCESSED_DATA_DIR, metadata_fred)

            print(f"\n✓ FRED indicators aligned and saved")
            print(f"  - Observations: {len(fred_aligned)}")
            print(f"  - Series: {', '.join(fred_aligned.columns.tolist())}")

        # Summary statistics
        print("\n" + "=" * 60)
        print("✓ DATA CLEANING COMPLETED")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Trading days: {len(clean_data)} (from {clean_data.index.min().date()} to {clean_data.index.max().date()})")
        print(f"  - Banks (quality-filtered): {clean_universe['bank_ric'].nunique()}")
        print(f"  - Countries: {clean_universe['country_chain'].nunique()}")
        print(f"  - Total variables: {clean_data.shape[1]}")
        print("\nCountry breakdown:")
        for country in clean_universe['country_chain'].unique():
            count = len(clean_universe[clean_universe['country_chain'] == country])
            print(f"  - {country}: {count} banks")

        print("\nReturns data:")
        print(f"  - Bank returns: {len(bank_returns.columns)} instruments")
        print(f"  - Benchmark returns: {len(bench_returns.columns)} indices")
        print(f"    Benchmarks: {', '.join(bench_returns.columns.tolist())}")

        print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")
        print(f"  - aligned_daily_data.parquet (full dataset)")
        print(f"  - returns_banks_clean.parquet (bank returns only)")
        print(f"  - returns_benchmarks_clean.parquet (benchmark returns only)")
        print(f"  - banks_universe_clean.parquet (filtered universe)")
        if fred_indicators is not None:
            print(f"  - fred_indicators_aligned.parquet (FRED macro indicators)")
        print(f"\nData quality plots saved to: {viz_dir}")

        # 8. Create returns visualization
        print("\n" + "=" * 60)
        print("CREATING RETURNS VISUALIZATION")
        print("=" * 60)

        create_returns_chart(
            bank_returns=bank_returns,
            bench_returns=bench_returns,
            bank_universe=clean_universe,
            output_dir=PROCESSED_DATA_DIR
        )

        print(f"\nOutlier filtering summary:")
        print(f"  - Thresholds: {p1:.4f}% (p1) to {p99:.4f}% (p99)")
        print(f"  - Outliers removed: {total_outliers:,} ({100*total_outliers/total_observations:.2f}%)")

        print("\nNext steps:")
        print("  1. Review data quality plots in: data/processed/data_quality_plots/")
        print("  2. Check missing_data_summary.csv for detailed statistics")
        print("  3. Review returns chart: data/processed/returns_by_country.png")
        print("  4. Run GARCH-DCC estimation: scripts/03_estimate_garch_dcc.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
