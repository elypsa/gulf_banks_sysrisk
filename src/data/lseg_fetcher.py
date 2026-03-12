"""LSEG Data API wrapper with error handling and retry logic.

This module provides clean interfaces to LSEG Data Library for Python,
with proper session management and error handling.
"""

import lseg.data as ld
from lseg.data.discovery import Chain
from contextlib import contextmanager
import pandas as pd
from typing import List, Dict, Optional
from datetime import date
import time


@contextmanager
def lseg_session():
    """Context manager for LSEG Data Library session.

    Yields:
        None (session is managed globally by ld library)

    Example:
        >>> with lseg_session():
        ...     df = ld.get_data(universe=['ENBD.DU'], fields=['TR.PriceClose'])
    """
    try:
        ld.open_session()
        yield
    finally:
        ld.close_session()


def get_chain_constituents(
    chain_ric: str,
    retry_attempts: int = 3,
    retry_delay: float = 2.0
) -> List[str]:
    """Retrieve constituents of a chain (e.g., country bank indices).

    Args:
        chain_ric: Chain RIC (e.g., '0#.TRXFLDAEPBANK' for UAE banks).
        retry_attempts: Number of retry attempts on failure.
        retry_delay: Delay in seconds between retries.

    Returns:
        List of constituent RICs.

    Raises:
        RuntimeError: If all retry attempts fail.

    Example:
        >>> with lseg_session():
        ...     uae_banks = get_chain_constituents('0#.TRXFLDAEPBANK')
        ...     print(uae_banks)  # ['ENBD.DU', 'DIB.DU', ...]
    """
    for attempt in range(retry_attempts):
        try:
            chain = Chain(name=chain_ric)
            constituents = chain.constituents
            print(f"✓ Fetched {len(constituents)} constituents from {chain_ric}")
            return constituents
        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"⚠ Attempt {attempt + 1} failed for {chain_ric}: {e}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Failed to fetch chain {chain_ric} after {retry_attempts} attempts: {e}"
                )


def get_gcc_bank_universe() -> Dict[str, List[str]]:
    """Retrieve all GCC banks organized by country.

    Returns:
        Dictionary mapping country chain RIC to list of bank RICs.

    Example:
        >>> with lseg_session():
        ...     banks = get_gcc_bank_universe()
        ...     print(banks.keys())  # ['0#.TRXFLDAEPBANK', '0#.TRXFLDSAPBANK', ...]
    """
    banks_by_country = {}

    # Get parent chain
    parent_chain = "0#.TRXFLDGCPUBANK"
    country_chains = get_chain_constituents(parent_chain)

    # Ensure Saudi Arabia is included (manual addition as per prototype)
    if "0#.TRXFLDSAPBANK" not in country_chains:
        country_chains.append("0#.TRXFLDSAPBANK")

    # Fetch constituents for each country
    for chain_ric in country_chains[1:]:  # Skip parent chain itself
        constituents = get_chain_constituents(chain_ric)
        if constituents:
            banks_by_country[chain_ric] = constituents

    print(f"\n✓ Total: {len(banks_by_country)} countries, "
          f"{sum(len(v) for v in banks_by_country.values())} banks")

    return banks_by_country


def get_fundamentals(
    universe: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    retry_attempts: int = 3
) -> pd.DataFrame:
    """Fetch quarterly fundamental data for banks.

    Args:
        universe: List of bank RICs.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format (defaults to today).
        retry_attempts: Number of retry attempts on failure.

    Returns:
        DataFrame with quarterly fundamentals (multi-column: RIC × Field).

    Example:
        >>> with lseg_session():
        ...     df = get_fundamentals(['ENBD.DU', 'DIB.DU'])
    """
    from ..utils import LSEG_FIELDS

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    fields = list(LSEG_FIELDS["fundamentals"].values())

    for attempt in range(retry_attempts):
        try:
            df = ld.get_history(
                universe=universe,
                fields=fields,
                parameters={
                    "Curn": "USD",  # USD conversion on-the-fly
                    "Period": "FQ0",  # Most recent quarter
                    "Frq": "FQ",  # Quarterly frequency
                    "SDate": "0",
                    "EDate": start_date
                }
            )
            print(f"✓ Fetched fundamentals: {df.shape[0]} quarters × {df.shape[1]} fields")
            return df

        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"⚠ Attempt {attempt + 1} failed for fundamentals: {e}")
                time.sleep(2.0)
            else:
                raise RuntimeError(f"Failed to fetch fundamentals: {e}")


def get_market_data(
    universe: List[str],
    start_date: str = "2009-12-31",
    end_date: Optional[str] = None,
    retry_attempts: int = 3
) -> pd.DataFrame:
    """Fetch daily market data (prices, returns, market cap) for banks.

    Args:
        universe: List of bank RICs.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format (defaults to today).
        retry_attempts: Number of retry attempts on failure.

    Returns:
        DataFrame with multi-index columns (RIC, Field).

    Example:
        >>> with lseg_session():
        ...     df = get_market_data(['ENBD.DU', 'DIB.DU'])
        ...     print(df['ENBD.DU']['TR.PriceClose'].head())
    """
    from ..utils import LSEG_FIELDS

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    fields = list(LSEG_FIELDS["market_data"].values())

    # Fetch each field separately and combine with multi-index
    field_frames = []

    for field in fields:
        for attempt in range(retry_attempts):
            try:
                df_field = ld.get_history(
                    universe=universe,
                    fields=[field],
                    parameters={"Curn": "USD", "Fill": "None"},  # USD conversion on-the-fly
                    start=start_date,
                    end=end_date,
                    interval="daily"
                )

                if not df_field.empty:
                    # Clean duplicates
                    df_field = df_field.groupby(df_field.index).first()

                    # Create multi-index: (RIC, Field)
                    new_columns = pd.MultiIndex.from_product(
                        [df_field.columns, [field]],
                        names=["RIC", "Field"]
                    )
                    df_field.columns = new_columns

                    field_frames.append(df_field)
                    print(f"✓ Fetched {field}: {df_field.shape[0]} days × {df_field.shape[1]} instruments")

                break  # Success, exit retry loop

            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"⚠ Attempt {attempt + 1} failed for {field}: {e}")
                    time.sleep(2.0)
                else:
                    print(f"✗ Failed to fetch {field} after {retry_attempts} attempts: {e}")

    if not field_frames:
        raise RuntimeError("Failed to fetch any market data fields")

    # Concatenate all fields
    df_combined = pd.concat(field_frames, axis=1)
    print(f"\n✓ Combined market data: {df_combined.shape[0]} days × {df_combined.shape[1]} fields")

    return df_combined


def get_benchmarks(
    benchmark_rics: List[str],
    start_date: str = "2009-12-31",
    end_date: Optional[str] = None,
    retry_attempts: int = 3
) -> pd.DataFrame:
    """Fetch daily benchmark index prices.

    Args:
        benchmark_rics: List of benchmark RICs (e.g., ['.GPDGC', '.TRXFLDGCPU']).
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format (defaults to today).
        retry_attempts: Number of retry attempts on failure.

    Returns:
        DataFrame with daily prices (columns = benchmark RICs).

    Example:
        >>> with lseg_session():
        ...     df = get_benchmarks(['.GPDGC', '.TRXFLDGCPU'])
    """
    from ..utils import LSEG_FIELDS

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    field = LSEG_FIELDS["benchmark"]["price_close"]

    for attempt in range(retry_attempts):
        try:
            df = ld.get_history(
                universe=benchmark_rics,
                fields=[field],
                parameters={"Curn": "USD", "Fill": "None"},  # USD conversion on-the-fly
                start=start_date,
                end=end_date,
                interval="daily"
            )
            print(f"✓ Fetched benchmarks: {df.shape[0]} days × {df.shape[1]} indices")
            return df

        except Exception as e:
            if attempt < retry_attempts - 1:
                print(f"⚠ Attempt {attempt + 1} failed for benchmarks: {e}")
                time.sleep(2.0)
            else:
                raise RuntimeError(f"Failed to fetch benchmarks: {e}")


def get_holidays(
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
    calendars: Optional[List[str]] = None
) -> pd.DataFrame:
    """Fetch GCC holiday calendar.

    Note: LSEG API cannot retrieve holidays before 2016-01-01.

    Args:
        start_date: Start date (cannot be earlier than 2016-01-01).
        end_date: End date (defaults to today).
        calendars: List of calendar codes (defaults to GCC countries).

    Returns:
        DataFrame with holiday information.

    Example:
        >>> with lseg_session():
        ...     holidays = get_holidays()
    """
    from ..utils import CONFIG

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    if calendars is None:
        calendars = CONFIG.HOLIDAY_CALENDARS

    holidays_response = ld.dates_and_calendars.holidays(
        start_date=start_date,
        end_date=end_date,
        calendars=calendars
    )

    print(f"✓ Fetched holidays: {len(holidays_response.df)} events")
    return holidays_response.df


def get_fx_rates(
    currency_pairs: List[str],
    start_date: str = "2009-12-31",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Fetch daily USD foreign exchange rates.

    Args:
        currency_pairs: List of currency pair RICs (e.g., ['AED=', 'SAR=']).
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format (defaults to today).

    Returns:
        DataFrame with daily FX rates (columns = currency pairs).

    Note:
        FX RICs typically use format: 'AED=' for USD/AED rate.

    Example:
        >>> with lseg_session():
        ...     fx = get_fx_rates(['AED=', 'SAR=', 'KWD='])
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    df = ld.get_history(
        universe=currency_pairs,
        fields=["TR.MIDPRICE"],
        parameters={"Fill": "None"},
        start=start_date,
        end=end_date,
        interval="daily"
    )

    print(f"✓ Fetched FX rates: {df.shape[0]} days × {df.shape[1]} currencies")
    return df
