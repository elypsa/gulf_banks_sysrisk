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

    return universe, fundamentals, market_data, benchmarks, holidays_lseg


def create_master_calendar(holidays_lseg):
    """Create comprehensive GCC holiday calendar."""
    print("\n" + "=" * 60)
    print("CREATING GCC MASTER CALENDAR")
    print("=" * 60)

    # Create calendar from 2010 to present
    holidays = create_gcc_holiday_calendar(
        start_date="2010-01-01",
        lseg_holidays=holidays_lseg
    )

    # Save
    metadata = {
        "description": "Comprehensive GCC holiday calendar",
        "sources": ["LSEG API (2016+)", "Hijri reconstruction", "National holidays"],
        "start_date": "2010-01-01"
    }
    save_dataframe(holidays, "gcc_holidays_full", PROCESSED_DATA_DIR, metadata)

    return holidays


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK DATA CLEANER & ALIGNER")
    print("=" * 60)

    try:
        # 1. Load raw data
        universe, fundamentals, market_data, benchmarks, holidays_lseg = load_raw_data()

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

        # 7. Save processed data
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)

        # Save aligned data
        metadata_aligned = {
            "description": "Aligned daily data (all sources merged, USD from LSEG)",
            "trading_days": len(clean_data),
            "num_banks": clean_universe["bank_ric"].nunique(),
            "start_date": str(clean_data.index.min().date()),
            "end_date": str(clean_data.index.max().date()),
            "currency": "USD (converted by LSEG on-the-fly)",
            "quality_filtered": True,
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
            "description": "Bank log returns for GARCH-DCC estimation",
            "trading_days": len(bank_returns),
            "num_banks": len(bank_returns.columns),
            "return_type": "log",
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
        print(f"\nData quality plots saved to: {viz_dir}")
        print("\nNext steps:")
        print("  1. Review data quality plots in: data/processed/data_quality_plots/")
        print("  2. Check missing_data_summary.csv for detailed statistics")
        print("  3. Run GARCH-DCC estimation: scripts/03_estimate_garch_dcc.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
