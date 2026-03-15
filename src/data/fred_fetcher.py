"""FRED data fetcher for macroeconomic indicators.

This module fetches macroeconomic time series from the Federal Reserve Economic Data (FRED)
API using the fredapi package. Data is used for systemic risk analysis and market stress indicators.

Key series:
- VIXCLS: CBOE Volatility Index (market stress indicator)
- T10Y3M: 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (yield curve)

API authentication is handled via .env file (FRED_API_KEY).
"""

import os
import pandas as pd
from fredapi import Fred
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv


def get_fred_api_key() -> str:
    """Load FRED API key from .env file.

    Returns:
        FRED API key string.

    Raises:
        ValueError: If FRED_API_KEY not found in .env file.

    Example:
        >>> api_key = get_fred_api_key()
    """
    # Load .env from project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at {env_path}")

    load_dotenv(env_path)

    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found in .env file. "
            "Please add: FRED_API_KEY=your_api_key_here"
        )

    return api_key


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> pd.Series:
    """Fetch a single FRED time series.

    Args:
        series_id: FRED series identifier (e.g., 'VIXCLS', 'T10Y3M').
        start_date: Start date in format 'YYYY-MM-DD'.
        end_date: End date in format 'YYYY-MM-DD' (default: today).
        verbose: Print progress messages.

    Returns:
        pandas Series with datetime index and series values.

    Raises:
        Exception: If FRED API request fails.

    Example:
        >>> vix = fetch_fred_series('VIXCLS', '2020-01-01')
        >>> print(vix.head())
    """
    # Get API key
    api_key = get_fred_api_key()

    # Initialize FRED client
    fred = Fred(api_key=api_key)

    if verbose:
        print(f"  Fetching {series_id} from FRED...")

    try:
        # Fetch series
        series = fred.get_series(
            series_id=series_id,
            observation_start=start_date,
            observation_end=end_date
        )

        if verbose:
            print(f"    ✓ Retrieved {len(series)} observations")
            print(f"      Date range: {series.index.min()} to {series.index.max()}")

        return series

    except Exception as e:
        raise Exception(f"Failed to fetch {series_id} from FRED: {e}")


def fetch_fred_multiple(
    series_ids: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Fetch multiple FRED time series and combine into DataFrame.

    Args:
        series_ids: List of FRED series identifiers.
        start_date: Start date in format 'YYYY-MM-DD'.
        end_date: End date in format 'YYYY-MM-DD' (default: today).
        verbose: Print progress messages.

    Returns:
        DataFrame with datetime index and one column per series.

    Example:
        >>> df = fetch_fred_multiple(['VIXCLS', 'T10Y3M'], '2020-01-01')
        >>> print(df.head())
    """
    if verbose:
        print(f"\nFetching {len(series_ids)} series from FRED...")

    series_dict = {}

    for series_id in series_ids:
        try:
            series = fetch_fred_series(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )
            series_dict[series_id] = series

        except Exception as e:
            print(f"    ✗ Failed to fetch {series_id}: {e}")
            continue

    if len(series_dict) == 0:
        raise Exception("No series successfully fetched from FRED")

    # Combine into DataFrame
    df = pd.DataFrame(series_dict)

    if verbose:
        print(f"\n  ✓ Combined {len(df.columns)} series into DataFrame")
        print(f"    Shape: {df.shape}")
        print(f"    Date range: {df.index.min()} to {df.index.max()}")
        print(f"    Missing values: {df.isna().sum().to_dict()}")

    return df


def fetch_systemic_risk_indicators(
    start_date: str,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Fetch key systemic risk indicators from FRED.

    Default series:
    - VIXCLS: CBOE Volatility Index (daily market stress)
    - T10Y3M: 10Y-3M Treasury spread (yield curve inversion indicator)

    Args:
        start_date: Start date in format 'YYYY-MM-DD'.
        end_date: End date in format 'YYYY-MM-DD' (default: today).
        verbose: Print progress messages.

    Returns:
        DataFrame with systemic risk indicators.

    Example:
        >>> indicators = fetch_systemic_risk_indicators('2020-01-01')
        >>> print(indicators.head())
    """
    series_ids = [
        'VIXCLS',   # VIX - Market volatility/stress
        'T10Y3M'    # 10Y-3M Treasury spread - Recession indicator
    ]

    return fetch_fred_multiple(
        series_ids=series_ids,
        start_date=start_date,
        end_date=end_date,
        verbose=verbose
    )
