"""Data processing and cleaning for SRISK calculation.

This module handles:
- Return calculation (log returns for GARCH)
- Forward-filling quarterly fundamentals to daily frequency
- Currency conversion to USD
- Missing value imputation
- Data quality checks
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import timedelta


def calculate_returns(
    prices: pd.Series,
    method: str = "log",
    handle_missing: str = "drop",
    lag: int = 1
) -> pd.Series:
    """Calculate returns from price series.

    Args:
        prices: Price series with DatetimeIndex.
        method: Return calculation method ('log' or 'simple').
        lag: Number of periods to lag for return calculation (default 1).
        handle_missing: How to handle NaN ('drop' or 'zero').

    Returns:
        Series of returns.

    Mathematical formulation:
        Log returns: r_t = log(P_t / P_{t-1})
        Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}

    Example:
        >>> prices = pd.Series([100, 102, 101], index=pd.date_range('2020-01-01', periods=3))
        >>> returns = calculate_returns(prices, method='log')
    """
    if method == "log":
        returns = np.log(prices / prices.shift(lag))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'.")

    # Handle missing values
    if handle_missing == "drop":
        returns = returns.dropna()
    elif handle_missing == "zero":
        returns = returns.fillna(0)

    return returns


def calculate_returns_multicolumn(
    df: pd.DataFrame,
    method: str = "log",
    handle_missing: str = "drop",
    suffix: str = "_return"
) -> pd.DataFrame:
    """Calculate returns for multi-column DataFrame.

    Args:
        df: DataFrame with prices (columns = instruments).
        method: Return calculation method.
        handle_missing: How to handle NaN.
        suffix: Suffix to add to column names.

    Returns:
        DataFrame with returns.

    Example:
        >>> prices_df = pd.DataFrame({'ENBD.DU': [100, 102], 'DIB.DU': [50, 51]})
        >>> returns_df = calculate_returns_multicolumn(prices_df)
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-index: Apply to each RIC
        returns_list = []
        for ric in df.columns.get_level_values(0).unique():
            if "Price" in str(df[ric].columns) or "Close" in str(df[ric].columns):
                # Find price column
                price_col = [col for col in df[ric].columns if "Price" in str(col) or "Close" in str(col)][0]
                returns = calculate_returns(df[ric][price_col], method, handle_missing, lag = 1)
                returns.name = (ric, f"{price_col}{suffix}")
                returns_list.append(returns)

        if returns_list:
            return pd.concat(returns_list, axis=1)
        else:
            raise ValueError("No price columns found in multi-index DataFrame")

    else:
        # Simple columns: Apply to all
        returns_df = df.apply(lambda col: calculate_returns(col, method, handle_missing, lag = 1))

        if suffix:
            returns_df.columns = [f"{col}{suffix}" for col in returns_df.columns]

        return returns_df


def interpolate_prices_with_max_gap(
    df: pd.DataFrame,
    max_gap_days: int = 2,
    method: str = "linear"
) -> pd.DataFrame:
    """Interpolate missing values in price data with a maximum gap constraint.

    This function fills small gaps in price data using linear interpolation,
    but only for gaps up to `max_gap_days`. Larger gaps are left as NaN.

    This is crucial for data quality: small gaps (1-2 days) are often due to
    data feed issues rather than trading halts, and linear interpolation is
    appropriate. Larger gaps likely indicate real trading suspensions or
    delisting, which should not be interpolated.

    Args:
        df: DataFrame with price data (DatetimeIndex, columns = instruments).
        max_gap_days: Maximum consecutive NaN days to interpolate (default 2).
        method: Interpolation method ('linear', 'time'). Default 'linear'.

    Returns:
        DataFrame with interpolated values (gaps <= max_gap_days filled).

    Mathematical formulation:
        For gap size g <= max_gap_days:
            P_t = P_{t-g-1} + (P_{t+1} - P_{t-g-1}) × (t - (t-g-1)) / (g + 2)

        For gap size g > max_gap_days:
            P_t = NaN (not interpolated)

    Example:
        >>> prices = pd.DataFrame({
        ...     'ENBD.DU': [100, np.nan, 102, np.nan, np.nan, np.nan, 110]
        ... }, index=pd.date_range('2020-01-01', periods=7))
        >>> interpolated = interpolate_prices_with_max_gap(prices, max_gap_days=2)
        >>> # Result: [100, 101, 102, NaN, NaN, NaN, 110]
        >>> # First gap (1 day) interpolated, second gap (3 days) not interpolated
    """
    df_interpolated = df.copy()

    for col in df.columns:
        series = df[col].copy()

        # Use pandas interpolate with limit parameter
        # This handles gaps up to max_gap_days automatically
        # limit_area='inside' ensures we don't extrapolate at edges
        series_filled = series.interpolate(
            method=method,
            limit=max_gap_days,
            limit_direction='both',
            limit_area='inside'
        )

        df_interpolated[col] = series_filled

    return df_interpolated


def forward_fill_fundamentals(
    fundamentals: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    reporting_lag_days: int = 45
) -> pd.DataFrame:
    """Forward-fill quarterly fundamentals to daily frequency.

    Accounts for reporting lag: Quarterly data for Q1 2020 (ending 2020-03-31)
    is typically not available until ~45 days later (mid-May 2020).

    Args:
        fundamentals: DataFrame with quarterly data (DatetimeIndex).
        target_dates: Daily dates to forward-fill to.
        reporting_lag_days: Days to lag fundamental data (default 45).

    Returns:
        DataFrame with daily frequency (forward-filled).

    Example:
        >>> quarterly_df = pd.DataFrame({'debt': [100, 105]},
        ...                              index=pd.to_datetime(['2020-03-31', '2020-06-30']))
        >>> daily_dates = pd.date_range('2020-01-01', '2020-12-31')
        >>> daily_df = forward_fill_fundamentals(quarterly_df, daily_dates)
    """
    # Apply reporting lag
    fundamentals_lagged = fundamentals.copy()
    fundamentals_lagged.index = fundamentals_lagged.index + timedelta(days=reporting_lag_days)

    # Handle duplicate index labels (multiple banks reporting on same date)
    # Each row has data for different banks (sparse columns with NaNs)
    # Combine by taking first non-null value for each column within each date
    if fundamentals_lagged.index.duplicated().any():
        num_dupes = fundamentals_lagged.index.duplicated().sum()
        print(f"  Warning: Found {num_dupes} duplicate dates in fundamentals")
        print(f"  Consolidating by taking first non-null value for each column per date...")

        # Group by date and take first non-null value for each column
        # .first() specifically takes the first non-null value in each group
        fundamentals_lagged = fundamentals_lagged.groupby(level=0).first()

    # Reindex to daily and forward-fill
    daily_df = fundamentals_lagged.reindex(target_dates, method="ffill")

    return daily_df


def convert_to_usd(
    df: pd.DataFrame,
    fx_rates: pd.DataFrame,
    currency_map: dict
) -> pd.DataFrame:
    """Convert local currency values to USD.

    Args:
        df: DataFrame with values in local currency (multi-column or simple).
        fx_rates: DataFrame with FX rates (columns = currency pairs like 'AED=').
        currency_map: Mapping of instrument to currency (e.g., {'ENBD.DU': 'AED='}).

    Returns:
        DataFrame with USD-converted values.

    Note:
        FX rates format: USD/CCY (how many local currency per 1 USD).
        To convert local to USD: USD = Local / FX_rate

    Example:
        >>> df = pd.DataFrame({'ENBD.DU': [1000]}, index=pd.to_datetime(['2020-01-01']))
        >>> fx = pd.DataFrame({'AED=': [3.67]}, index=pd.to_datetime(['2020-01-01']))
        >>> usd_df = convert_to_usd(df, fx, {'ENBD.DU': 'AED='})
    """
    df_usd = df.copy()

    # Align FX rates to same index
    fx_aligned = fx_rates.reindex(df.index, method="ffill")

    if isinstance(df.columns, pd.MultiIndex):
        # Multi-index: Convert each RIC
        for ric in df.columns.get_level_values(0).unique():
            if ric in currency_map:
                fx_pair = currency_map[ric]
                if fx_pair in fx_aligned.columns:
                    # Divide by FX rate to get USD
                    df_usd[ric] = df[ric].div(fx_aligned[fx_pair], axis=0)
                else:
                    print(f"Warning: FX rate {fx_pair} not found for {ric}")
            else:
                print(f"Warning: No currency mapping for {ric}")

    else:
        # Simple columns
        for col in df.columns:
            if col in currency_map:
                fx_pair = currency_map[col]
                if fx_pair in fx_aligned.columns:
                    df_usd[col] = df[col] / fx_aligned[fx_pair]
                else:
                    print(f"Warning: FX rate {fx_pair} not found for {col}")

    return df_usd


def create_currency_map(
    bank_universe: pd.DataFrame,
    fx_suffix: str = "="
) -> dict:
    """Create mapping of bank RICs to FX rate symbols.

    Args:
        bank_universe: DataFrame with bank_ric and country_chain columns.
        fx_suffix: Suffix for FX symbols (default '=').

    Returns:
        Dictionary mapping bank RIC to FX symbol.

    Example:
        >>> universe = pd.DataFrame({
        ...     'bank_ric': ['ENBD.DU', '1180.SE'],
        ...     'country_chain': ['0#.TRXFLDAEPBANK', '0#.TRXFLDSAPBANK']
        ... })
        >>> currency_map = create_currency_map(universe)
        >>> print(currency_map)  # {'ENBD.DU': 'AED=', '1180.SE': 'SAR='}
    """
    # Country to currency mapping
    country_to_currency = {
        "AEPBANK": "AED",  # UAE Dirham
        "SAPBANK": "SAR",  # Saudi Riyal
        "QAPBANK": "QAR",  # Qatari Riyal
        "KWPBANK": "KWD",  # Kuwaiti Dinar
        "OMPBANK": "OMR",  # Omani Rial
        "BHPBANK": "BHD"   # Bahraini Dinar
    }

    currency_map = {}

    for _, row in bank_universe.iterrows():
        bank_ric = row["bank_ric"]
        country_chain = row["country_chain"]

        # Extract country code from chain
        for country_code, currency_code in country_to_currency.items():
            if country_code in country_chain:
                currency_map[bank_ric] = f"{currency_code}{fx_suffix}"
                break

    return currency_map


def align_and_merge_data(
    fundamentals: pd.DataFrame,
    market_data: pd.DataFrame,
    benchmarks: pd.DataFrame,
    fx_rates: pd.DataFrame,
    bank_universe: pd.DataFrame,
    convert_usd: bool = True,
    interpolate_prices: bool = True,
    max_interpolation_gap: int = 2
) -> pd.DataFrame:
    """Align and merge all data sources into a single DataFrame.

    This creates the master dataset for SRISK calculation by:
    1. Forward-filling quarterly fundamentals to daily
    2. Interpolating small gaps in prices/market cap (optional)
    3. Calculating returns from (interpolated) prices
    4. Merging with daily market data
    5. Adding benchmark returns
    6. Converting to USD (optional)

    Args:
        fundamentals: Quarterly fundamentals DataFrame.
        market_data: Daily market data (multi-index).
        benchmarks: Daily benchmark prices.
        fx_rates: Daily FX rates.
        bank_universe: Bank universe with country mapping.
        convert_usd: Whether to convert to USD.
        interpolate_prices: Whether to interpolate small gaps in prices (default True).
        max_interpolation_gap: Maximum gap size (days) to interpolate (default 2).

    Returns:
        Aligned DataFrame with all data (DatetimeIndex, multi-column).

    Note:
        Small gaps (1-2 days) in price data are often due to data feed issues
        rather than real trading halts. Linear interpolation fills these gaps,
        improving data quality and allowing more banks to pass quality filters.

    Example:
        >>> aligned = align_and_merge_data(fund, mkt, bench, fx, universe,
        ...                                 interpolate_prices=True, max_interpolation_gap=2)
    """
    # Get date range from market data
    target_dates = market_data.index

    # 1. Forward-fill fundamentals
    print("Forward-filling quarterly fundamentals to daily...")
    fund_daily = forward_fill_fundamentals(fundamentals, target_dates)

    # 2. Calculate benchmark returns
    print("Calculating benchmark returns...")
    bench_returns = calculate_returns_multicolumn(benchmarks, method="log", suffix="")

    # 3. Extract prices and market caps from multi-index
    print("Extracting prices and market caps...")
    if isinstance(market_data.columns, pd.MultiIndex):
        prices = {}
        mkt_caps = {}
        for ric in market_data.columns.get_level_values(0).unique():
            # Find price and market cap columns
            for field in market_data[ric].columns:
                if "PriceClose" in str(field):
                    prices[ric] = market_data[ric][field]
                elif "MarketCapitalization" in str(field):
                    mkt_caps[ric] = market_data[ric][field]

        prices_df = pd.DataFrame(prices)
        mkt_caps_df = pd.DataFrame(mkt_caps)

    else:
        raise ValueError("Market data should have multi-index columns")

    # 4. Interpolate small gaps in prices and market caps (before calculating returns)
    if interpolate_prices and max_interpolation_gap > 0:
        print(f"Interpolating small gaps (≤ {max_interpolation_gap} days) in prices and market caps...")

        # Count gaps before interpolation
        gaps_before_prices = prices_df.isna().sum().sum()
        gaps_before_mktcap = mkt_caps_df.isna().sum().sum()

        # Apply interpolation
        prices_df = interpolate_prices_with_max_gap(prices_df, max_gap_days=max_interpolation_gap)
        mkt_caps_df = interpolate_prices_with_max_gap(mkt_caps_df, max_gap_days=max_interpolation_gap)

        # Count gaps after interpolation
        gaps_after_prices = prices_df.isna().sum().sum()
        gaps_after_mktcap = mkt_caps_df.isna().sum().sum()

        gaps_filled_prices = gaps_before_prices - gaps_after_prices
        gaps_filled_mktcap = gaps_before_mktcap - gaps_after_mktcap

        print(f"  ✓ Prices: filled {gaps_filled_prices} gaps ({gaps_before_prices} → {gaps_after_prices} missing)")
        print(f"  ✓ Market caps: filled {gaps_filled_mktcap} gaps ({gaps_before_mktcap} → {gaps_after_mktcap} missing)")

    # 5. Calculate returns from (potentially interpolated) prices
    print("Calculating bank returns...")
    returns_df = calculate_returns_multicolumn(prices_df, method="log", suffix="")

    # 6. Convert to USD if requested
    if convert_usd:
        print("Converting to USD...")
        currency_map = create_currency_map(bank_universe)

        fund_daily_usd = convert_to_usd(fund_daily, fx_rates, currency_map)
        mkt_caps_df_usd = convert_to_usd(mkt_caps_df, fx_rates, currency_map)
    else:
        fund_daily_usd = fund_daily
        mkt_caps_df_usd = mkt_caps_df

    # 7. Merge everything
    print("Merging all data sources...")

    # Flatten column names and add prefixes to avoid conflicts
    # fundamentals may have multi-index columns from LSEG
    if isinstance(fund_daily_usd.columns, pd.MultiIndex):
        fund_daily_usd.columns = ['_'.join(map(str, col)).strip() for col in fund_daily_usd.columns]
    fund_daily_usd = fund_daily_usd.add_prefix("fund_")

    if isinstance(mkt_caps_df_usd.columns, pd.MultiIndex):
        mkt_caps_df_usd.columns = ['_'.join(map(str, col)).strip() for col in mkt_caps_df_usd.columns]
    mkt_caps_df_usd = mkt_caps_df_usd.add_prefix("mktcap_")

    if isinstance(prices_df.columns, pd.MultiIndex):
        prices_df.columns = ['_'.join(map(str, col)).strip() for col in prices_df.columns]
    prices_df = prices_df.add_prefix("price_")

    if isinstance(returns_df.columns, pd.MultiIndex):
        returns_df.columns = ['_'.join(map(str, col)).strip() for col in returns_df.columns]
    returns_df = returns_df.add_prefix("ret_")

    if isinstance(bench_returns.columns, pd.MultiIndex):
        bench_returns.columns = ['_'.join(map(str, col)).strip() for col in bench_returns.columns]
    bench_returns = bench_returns.add_prefix("bench_ret_")

    # Concatenate
    aligned = pd.concat([
        fund_daily_usd,
        mkt_caps_df_usd,
        prices_df,
        returns_df,
        bench_returns
    ], axis=1)

    # Drop rows with all NaN
    aligned = aligned.dropna(how="all")

    print(f"✓ Aligned data: {aligned.shape[0]} days × {aligned.shape[1]} columns")

    return aligned


def filter_banks_by_data_quality(
    aligned_data: pd.DataFrame,
    bank_universe: pd.DataFrame,
    min_trading_days: int = 252,
    max_missing_pct: float = 0.20,
    return_quality_report: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out banks with insufficient data quality.

    Args:
        aligned_data: Aligned DataFrame with all data.
        bank_universe: Bank universe DataFrame.
        min_trading_days: Minimum number of trading days required.
        max_missing_pct: Maximum allowed missing data percentage.
        return_quality_report: If True, return quality report DataFrame as well.

    Returns:
        Tuple of (filtered_data, filtered_universe) or
        (filtered_data, filtered_universe, quality_report) if return_quality_report=True.

    Example:
        >>> clean_data, clean_universe = filter_banks_by_data_quality(
        ...     aligned, universe, min_trading_days=252
        ... )
    """
    # Extract return columns
    # Handle both string columns and tuple columns (multi-index)
    ret_cols = []
    for col in aligned_data.columns:
        if isinstance(col, tuple):
            # Multi-index column: check if any part starts with "ret_"
            col_str = '_'.join(map(str, col))
            if col_str.startswith("ret_"):
                ret_cols.append(col)
        elif isinstance(col, str):
            # Simple string column
            if col.startswith("ret_"):
                ret_cols.append(col)

    quality_report = []

    for col in ret_cols:
        # Extract bank RIC from column name
        if isinstance(col, tuple):
            bank_ric = '_'.join(map(str, col)).replace("ret_", "")
        else:
            bank_ric = col.replace("ret_", "")

        # Count non-missing values
        non_missing = aligned_data[col].notna().sum()
        total = len(aligned_data)
        missing_pct = 1 - (non_missing / total)

        quality_report.append({
            "bank_ric": bank_ric,
            "non_missing_days": non_missing,
            "total_days": total,
            "missing_pct": missing_pct,
            "passes_quality": (non_missing >= min_trading_days) and (missing_pct <= max_missing_pct)
        })

    quality_df = pd.DataFrame(quality_report)

    # Filter banks
    good_banks = quality_df[quality_df["passes_quality"]]["bank_ric"].tolist()

    print("\nData Quality Report:")
    print(f"  - Total banks: {len(quality_df)}")
    print(f"  - Passing quality checks: {len(good_banks)}")
    print(f"  - Removed: {len(quality_df) - len(good_banks)}")

    # Filter data - keep bank columns and benchmark columns
    cols_to_keep = []
    for col in aligned_data.columns:
        # Convert column to string for checking
        col_str = '_'.join(map(str, col)) if isinstance(col, tuple) else col

        # Keep if contains any good bank RIC or starts with "bench_"
        if any(bank in col_str for bank in good_banks) or col_str.startswith("bench_"):
            cols_to_keep.append(col)

    filtered_data = aligned_data[cols_to_keep]

    # Filter universe
    filtered_universe = bank_universe[bank_universe["bank_ric"].isin(good_banks)]

    if return_quality_report:
        return filtered_data, filtered_universe, quality_df
    else:
        return filtered_data, filtered_universe
