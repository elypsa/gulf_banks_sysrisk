#!/usr/bin/env python3
"""Script to fetch raw data from LSEG and persist to parquet files.

This script pulls all required data for SRISK calculation:
- GCC bank universe (by country)
- Quarterly fundamental data (balance sheets)
- Daily market data (prices, returns, market cap)
- Benchmark indices (S&P GCC Composite + country banking indices)
- Holiday calendar
- USD FX rates

All data is saved to data/raw/ in parquet format with metadata.

Usage:
    uv run scripts/01_fetch_and_persist_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    lseg_session,
    get_gcc_bank_universe,
    get_fundamentals,
    get_market_data,
    get_benchmarks,
    get_holidays,
    get_fx_rates,
    save_dataframe,
    fetch_incremental_or_full
)
from src.utils.config import CONFIG, RAW_DATA_DIR
import pandas as pd
from datetime import date

pd.set_option('future.no_silent_downcasting', True)

def fetch_universe() -> pd.DataFrame:
    """Fetch and save GCC bank universe.

    Returns:
        DataFrame with columns: bank_ric, country_chain
    """
    print("\n" + "=" * 60)
    print("FETCHING BANK UNIVERSE")
    print("=" * 60)

    with lseg_session():
        banks_by_country = get_gcc_bank_universe()

    # Convert to DataFrame for easier handling
    records = []
    for country_chain, bank_rics in banks_by_country.items():
        for bank_ric in bank_rics:
            records.append({
                "bank_ric": bank_ric,
                "country_chain": country_chain,
                "country_code": country_chain.split(".")[-1][:3]  # Extract country code
            })

    df = pd.DataFrame(records)

    # Save
    metadata = {
        "description": "GCC bank universe by country",
        "num_countries": len(banks_by_country),
        "num_banks": len(df),
        "source": "LSEG Chain 0#.TRXFLDGCPUBANK"
    }
    save_dataframe(df, "gcc_banks_universe", RAW_DATA_DIR, metadata)

    return df


def fetch_fundamentals(bank_universe: pd.DataFrame) -> pd.DataFrame:
    """Fetch and save quarterly fundamental data.

    Args:
        bank_universe: DataFrame with bank_ric column.

    Returns:
        DataFrame with quarterly fundamentals.
    """
    print("\n" + "=" * 60)
    print("FETCHING QUARTERLY FUNDAMENTALS")
    print("=" * 60)

    bank_rics = bank_universe["bank_ric"].tolist()

    with lseg_session():
        # Fetch by country to avoid timeout issues
        all_dfs = []
        for country_chain in bank_universe["country_chain"].unique():
            country_banks = bank_universe[
                bank_universe["country_chain"] == country_chain
            ]["bank_ric"].tolist()

            print(f"\nFetching {country_chain} ({len(country_banks)} banks)...")
            df = get_fundamentals(country_banks, start_date=CONFIG.DATA_START_DATE)

            # Add country identifier
            df["country_chain"] = country_chain
            all_dfs.append(df)

    # Combine all countries
    df_combined = pd.concat(all_dfs, axis=0)

    # Save
    metadata = {
        "description": "Quarterly fundamental data (balance sheets)",
        "fields": ["Total Assets", "Common Equity", "Total Liabilities", "Total Liab & Equity"],
        "frequency": "Quarterly",
        "start_date": CONFIG.DATA_START_DATE,
        "currency": "USD (converted by LSEG)",
        "num_banks": len(bank_rics)
    }
    save_dataframe(df_combined, "fundamentals_quarterly", RAW_DATA_DIR, metadata)

    return df_combined


def fetch_market_data(bank_universe: pd.DataFrame) -> pd.DataFrame:
    """Fetch and save daily market data (with incremental fetching support).

    Args:
        bank_universe: DataFrame with bank_ric column.

    Returns:
        DataFrame with daily market data (multi-index columns).
    """
    print("\n" + "=" * 60)
    print("FETCHING DAILY MARKET DATA")
    print("=" * 60)

    bank_rics = bank_universe["bank_ric"].tolist()

    def _fetch_market_data_by_country(start_date: str) -> pd.DataFrame:
        """Inner function to fetch market data for all countries from start_date."""
        with lseg_session():
            # Fetch by country to avoid timeout
            all_dfs = []
            for country_chain in bank_universe["country_chain"].unique():
                country_banks = bank_universe[
                    bank_universe["country_chain"] == country_chain
                ]["bank_ric"].tolist()

                print(f"  Fetching {country_chain} ({len(country_banks)} banks)...")
                df = get_market_data(country_banks, start_date=start_date)

                # DIAGNOSTIC: Check for object dtypes (mixed types)
                object_cols = [col for col in df.columns if df[col].dtype == 'object']
                if object_cols:
                    print(f"    ⚠️  Found {len(object_cols)} columns with mixed types:")
                    for col in object_cols[:3]:  # Show first 3
                        sample_values = df[col].dropna().head(10).tolist()
                        print(f"        {col}: {sample_values}")

                all_dfs.append(df)

        # Combine all countries
        df_combined = pd.concat(all_dfs, axis=1)

        # DIAGNOSTIC: Check combined dataframe
        object_cols_combined = [col for col in df_combined.columns if df_combined[col].dtype == 'object']
        if object_cols_combined:
            print(f"\n  ⚠️  After concat: {len(object_cols_combined)} columns with object dtype")

        # FIX: Coerce all columns to numeric (LSEG error codes → NaN)
        print("\n  🔧 Coercing mixed-type columns to numeric...")
        for col in df_combined.columns:
            if df_combined[col].dtype == 'object':
                # Convert to numeric, errors='coerce' will turn strings into NaN
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Verify fix
        remaining_object_cols = [col for col in df_combined.columns if df_combined[col].dtype == 'object']
        if remaining_object_cols:
            print(f"    ⚠️  Still have {len(remaining_object_cols)} object columns after coercion")
        else:
            if object_cols_combined:
                print(f"    ✓ All columns now numeric (converted: {len(object_cols_combined)})")

        return df_combined

    # Metadata template
    metadata = {
        "description": "Daily market data (prices, returns, market cap)",
        "fields": ["TR.PriceClose", "TR.TotalReturn", "TR.CompanyMarketCapitalization"],
        "frequency": "Daily",
        "currency": "USD (converted by LSEG)",
        "num_banks": len(bank_rics)
    }

    # Use incremental fetching
    df_combined = fetch_incremental_or_full(
        filename="prices_daily",
        data_dir=RAW_DATA_DIR,
        fetch_function=_fetch_market_data_by_country,
        full_start_date=CONFIG.DATA_START_DATE,
        metadata_template=metadata,
        verbose=True
    )

    return df_combined


def fetch_benchmarks() -> pd.DataFrame:
    """Fetch and save benchmark indices (with incremental fetching support).

    Returns:
        DataFrame with daily benchmark prices.
    """
    print("\n" + "=" * 60)
    print("FETCHING BENCHMARK INDICES")
    print("=" * 60)

    # Broad benchmark
    broad_benchmark = CONFIG.BENCHMARK_BROAD  # .GPDGC

    # Country banking indices
    country_indices = list(CONFIG.COUNTRY_INDICES.values())

    all_benchmarks = [broad_benchmark] + country_indices

    def _fetch_benchmarks_data(start_date: str) -> pd.DataFrame:
        """Inner function to fetch benchmark data from start_date."""
        with lseg_session():
            return get_benchmarks(all_benchmarks, start_date=start_date)

    # Metadata template
    metadata = {
        "description": "Benchmark indices (broad + country banking sectors)",
        "broad_benchmark": broad_benchmark,
        "country_indices": CONFIG.COUNTRY_INDICES,
        "frequency": "Daily"
    }

    # Use incremental fetching
    df = fetch_incremental_or_full(
        filename="benchmarks_daily",
        data_dir=RAW_DATA_DIR,
        fetch_function=_fetch_benchmarks_data,
        full_start_date=CONFIG.DATA_START_DATE,
        metadata_template=metadata,
        verbose=True
    )

    return df


def fetch_holidays() -> pd.DataFrame:
    """Fetch and save GCC holiday calendar.

    Returns:
        DataFrame with holiday events.
    """
    print("\n" + "=" * 60)
    print("FETCHING HOLIDAY CALENDAR")
    print("=" * 60)

    today_str = date.today().strftime("%Y-%m-%d")

    with lseg_session():
        df = get_holidays(start_date=CONFIG.HOLIDAYS_START_DATE, end_date=today_str)

    # Save
    metadata = {
        "description": "GCC holiday calendar",
        "calendars": CONFIG.HOLIDAY_CALENDARS,
        "start_date": CONFIG.HOLIDAYS_START_DATE,
        "end_date": today_str,
        "note": "LSEG API cannot retrieve holidays before 2016-01-01"
    }
    save_dataframe(df, "holidays_calendar", RAW_DATA_DIR, metadata)

    return df


def fetch_fx_rates() -> pd.DataFrame:
    """Fetch and save USD FX rates for GCC currencies (with incremental fetching support).

    Returns:
        DataFrame with daily FX rates.
    """
    print("\n" + "=" * 60)
    print("FETCHING USD FX RATES")
    print("=" * 60)

    # GCC currency pairs (vs USD)
    # Format: CCY= returns USD/CCY rate (how many CCY per 1 USD)
    currency_pairs = [
        "AED=",  # UAE Dirham
        "SAR=",  # Saudi Riyal
        "QAR=",  # Qatari Riyal
        "KWD=",  # Kuwaiti Dinar
        "OMR=",  # Omani Rial
        "BHD="   # Bahraini Dinar
    ]

    def _fetch_fx_data(start_date: str) -> pd.DataFrame:
        """Inner function to fetch FX data from start_date."""
        with lseg_session():
            return get_fx_rates(currency_pairs, start_date=start_date)

    # Metadata template
    metadata = {
        "description": "Daily USD FX rates for GCC currencies",
        "currency_pairs": currency_pairs,
        "frequency": "Daily",
        "note": "Rates are USD/CCY (how many local currency per 1 USD)"
    }

    # Use incremental fetching
    df = fetch_incremental_or_full(
        filename="fx_rates_daily",
        data_dir=RAW_DATA_DIR,
        fetch_function=_fetch_fx_data,
        full_start_date=CONFIG.DATA_START_DATE,
        metadata_template=metadata,
        verbose=True
    )

    return df


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK DATA FETCHER")
    print("=" * 60)
    print(f"\nSaving data to: {RAW_DATA_DIR}")
    print("This may take 5-10 minutes depending on LSEG API response times...\n")

    try:
        # 1. Fetch bank universe
        universe = fetch_universe()

        # 2. Fetch fundamentals
        fundamentals = fetch_fundamentals(universe)

        # 3. Fetch market data
        market_data = fetch_market_data(universe)

        # 4. Fetch benchmarks
        benchmarks = fetch_benchmarks()

        # 5. Fetch holidays
        holidays = fetch_holidays()

        # 6. Fetch FX rates (DISABLED: Using LSEG on-the-fly USD conversion)
        # fx_rates = fetch_fx_rates()

        print("\n" + "=" * 60)
        print("✓ ALL DATA FETCHED SUCCESSFULLY")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Banks: {len(universe)} across {universe['country_chain'].nunique()} countries")
        print(f"  - Fundamentals: {fundamentals.shape[0]} quarters")
        print(f"  - Market data: {market_data.shape[0]} trading days")
        print(f"  - Benchmarks: {benchmarks.shape[0]} days × {benchmarks.shape[1]} indices")
        print(f"  - Holidays: {len(holidays)} events")
        # print(f"  - FX rates: {fx_rates.shape[0]} days × {fx_rates.shape[1]} currencies")
        print(f"\nData saved to: {RAW_DATA_DIR}")
        print("\nNext step: Run scripts/02_clean_and_align_data.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
