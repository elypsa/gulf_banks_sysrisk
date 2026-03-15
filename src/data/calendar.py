"""GCC trading calendar management.

This module handles:
- GCC holiday calendars (Islamic and national holidays)
- UAE market schedule transition (Sun-Thu → Mon-Fri in Jan 2022)
- Trading day alignment across GCC markets
- Pre-2016 Islamic holiday reconstruction

Key considerations:
- UAE shifted to Mon-Fri in January 2022
- Other GCC markets remain Sun-Thu
- Islamic holidays shift ~11 days earlier each year (lunar calendar)
- LSEG holiday API only available from 2016+
"""

import pandas as pd
from datetime import datetime, date
from typing import List, Optional
from hijri_converter import Hijri, Gregorian


def get_islamic_holidays_hijri() -> List[tuple]:
    """Get major Islamic holidays in Hijri calendar.

    Returns:
        List of (month, day, name, duration_days) tuples.

    Note:
        These are fixed dates in the Hijri calendar but shift in Gregorian.
    """
    return [
        (10, 1, "Eid al-Fitr", 3),  # Shawwal 1-3
        (12, 9, "Eid al-Adha", 4),  # Dhul Hijjah 9-12
        (1, 1, "Islamic New Year", 1),  # Muharram 1
        (3, 12, "Prophet's Birthday", 1),  # Rabi' al-Awwal 12
        (7, 27, "Isra and Mi'raj", 1),  # Rajab 27
    ]


def hijri_to_gregorian_range(hijri_year: int, month: int, day: int, duration: int) -> List[date]:
    """Convert Hijri date range to Gregorian dates.

    Args:
        hijri_year: Hijri year (e.g., 1437).
        month: Hijri month (1-12).
        day: Hijri day.
        duration: Number of days for the holiday.

    Returns:
        List of Gregorian date objects.

    Example:
        >>> # Eid al-Fitr 1437 (Shawwal 1-3)
        >>> dates = hijri_to_gregorian_range(1437, 10, 1, 3)
    """
    dates = []
    for i in range(duration):
        try:
            hijri_date = Hijri(hijri_year, month, day + i)
            greg_date = hijri_date.to_gregorian()
            dates.append(date(greg_date.year, greg_date.month, greg_date.day))
        except Exception:
            # Handle invalid dates (e.g., month only has 29 days)
            continue
    return dates


def reconstruct_islamic_holidays(start_year: int = 2005, end_year: int = 2026) -> pd.DataFrame:
    """Reconstruct Islamic holidays for years before 2016.

    LSEG API cannot provide holidays before 2016, so we reconstruct them
    using the Hijri calendar.

    Args:
        start_year: Start Gregorian year.
        end_year: End Gregorian year.

    Returns:
        DataFrame with columns: date, name, calendars.

    Example:
        >>> holidays = reconstruct_islamic_holidays(2010, 2016)
        >>> print(holidays.head())
    """
    islamic_holidays = get_islamic_holidays_hijri()

    # Convert Gregorian year range to Hijri years (approximate)
    # Hijri year is ~11 days shorter, so we need to check surrounding years
    hijri_start = Gregorian(start_year, 1, 1).to_hijri().year - 1
    hijri_end = Gregorian(end_year, 12, 31).to_hijri().year + 1

    records = []

    for hijri_year in range(hijri_start, hijri_end + 1):
        for month, day, name, duration in islamic_holidays:
            greg_dates = hijri_to_gregorian_range(hijri_year, month, day, duration)

            for greg_date in greg_dates:
                # Only include if in target Gregorian range
                if start_year <= greg_date.year <= end_year:
                    records.append({
                        "date": greg_date,
                        "name": name,
                        "calendars": ["UAE", "SAU", "QAT", "KWT", "OMN", "BAH"],  # All GCC
                        "source": "reconstructed_hijri"
                    })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["date", "name"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_gcc_national_holidays() -> List[dict]:
    """Get GCC national holidays (fixed Gregorian dates).

    Returns:
        List of holiday dictionaries with date pattern, name, and countries.

    Note:
        These are non-Islamic holidays that occur on fixed Gregorian dates.
    """
    return [
        # New Year (observed by some GCC countries)
        {"month": 1, "day": 1, "name": "New Year's Day", "calendars": ["UAE", "OMN", "BAH"]},

        # UAE National Day
        {"month": 12, "day": 2, "name": "UAE National Day", "calendars": ["UAE"]},
        {"month": 12, "day": 3, "name": "UAE National Day", "calendars": ["UAE"]},

        # Saudi National Day
        {"month": 9, "day": 23, "name": "Saudi National Day", "calendars": ["SAU"]},

        # Qatar National Day
        {"month": 12, "day": 18, "name": "Qatar National Day", "calendars": ["QAT"]},

        # Kuwait National Day
        {"month": 2, "day": 25, "name": "Kuwait National Day", "calendars": ["KWT"]},
        {"month": 2, "day": 26, "name": "Kuwait Liberation Day", "calendars": ["KWT"]},

        # Oman National Day
        {"month": 11, "day": 18, "name": "Oman National Day", "calendars": ["OMN"]},
        {"month": 11, "day": 19, "name": "Oman National Day", "calendars": ["OMN"]},

        # Bahrain National Day
        {"month": 12, "day": 16, "name": "Bahrain National Day", "calendars": ["BAH"]},
        {"month": 12, "day": 17, "name": "Bahrain National Day", "calendars": ["BAH"]},
    ]


def expand_national_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    """Expand national holidays across year range.

    Args:
        start_year: Start year.
        end_year: End year.

    Returns:
        DataFrame with date, name, calendars columns.
    """
    national_holidays = get_gcc_national_holidays()
    records = []

    for year in range(start_year, end_year + 1):
        for holiday in national_holidays:
            records.append({
                "date": date(year, holiday["month"], holiday["day"]),
                "name": holiday["name"],
                "calendars": holiday["calendars"],
                "source": "national_fixed"
            })

    return pd.DataFrame(records)


def create_gcc_holiday_calendar(
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    lseg_holidays: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Create comprehensive GCC holiday calendar.

    Combines:
    - Reconstructed Islamic holidays (2010-2015)
    - LSEG API holidays (2016+)
    - National holidays (all years)

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (defaults to today).
        lseg_holidays: Optional DataFrame from LSEG API (2016+).

    Returns:
        DataFrame with date, name, calendars columns.

    Example:
        >>> # Without LSEG data
        >>> holidays = create_gcc_holiday_calendar("2010-01-01", "2025-12-31")
        >>>
        >>> # With LSEG data
        >>> from src.data import load_dataframe, RAW_DATA_DIR
        >>> lseg_hol = load_dataframe("holidays_calendar", RAW_DATA_DIR)
        >>> holidays = create_gcc_holiday_calendar(lseg_holidays=lseg_hol)
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_holidays = []

    # 1. Reconstructed Islamic holidays (all years for consistency)
    islamic = reconstruct_islamic_holidays(start_dt.year, end_dt.year)
    all_holidays.append(islamic)

    # 2. National holidays
    national = expand_national_holidays(start_dt.year, end_dt.year)
    all_holidays.append(national)

    # 3. LSEG holidays if provided (takes precedence for 2016+)
    if lseg_holidays is not None and not lseg_holidays.empty:
        # Process LSEG format
        lseg_processed = lseg_holidays.copy()
        lseg_processed["source"] = "lseg_api"

        # Ensure date column is datetime
        if "date" in lseg_processed.columns:
            lseg_processed["date"] = pd.to_datetime(lseg_processed["date"]).dt.date

        # Keep only relevant columns
        if "calendars" in lseg_processed.columns:
            lseg_processed = lseg_processed[["date", "name", "calendars", "source"]]
            all_holidays.append(lseg_processed)

    # Combine all sources
    df = pd.concat(all_holidays, ignore_index=True)

    # Convert date to datetime for filtering
    df["date"] = pd.to_datetime(df["date"])

    # Filter to date range
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # Remove duplicates (LSEG takes precedence for 2016+)
    df = df.sort_values(["date", "source"])
    df = df.drop_duplicates(subset=["date", "name"], keep="first")

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_trading_days(
    start_date: str,
    end_date: str,
    calendar: str = "GCC",
    exclude_holidays: bool = True,
    holiday_df: Optional[pd.DataFrame] = None
) -> pd.DatetimeIndex:
    """Get trading days for GCC markets.

    Args:
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        calendar: Calendar to use ('GCC', 'UAE', 'SAU', etc.).
        exclude_holidays: Whether to exclude holidays.
        holiday_df: Optional holiday DataFrame (if None, loads from data).

    Returns:
        DatetimeIndex of trading days.

    Note:
        - Pre-2022: UAE is Sun-Thu, others are Sun-Thu
        - Post-2022: UAE is Mon-Fri, others remain Sun-Thu

    Example:
        >>> trading_days = get_trading_days("2020-01-01", "2020-12-31", "UAE")
    """
    from ..utils.config import CONFIG

    # Create daily date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # UAE calendar transition
    uae_transition = pd.Timestamp(CONFIG.UAE_TRANSITION_DATE)

    if calendar == "UAE":
        # Pre-2022: Sunday-Thursday (weekday 6, 0-3)
        # Post-2022: Monday-Friday (weekday 0-4)
        pre_transition = all_dates[all_dates < uae_transition]
        post_transition = all_dates[all_dates >= uae_transition]

        pre_trading = pre_transition[pre_transition.weekday.isin([6, 0, 1, 2, 3])]  # Sun-Thu
        post_trading = post_transition[post_transition.weekday.isin([0, 1, 2, 3, 4])]  # Mon-Fri

        trading_days = pre_trading.append(post_trading)

    elif calendar == "GCC":
        # Conservative: Use intersection of all GCC markets
        # Most GCC markets are Sun-Thu throughout
        trading_days = all_dates[all_dates.weekday.isin([6, 0, 1, 2, 3])]  # Sun-Thu

    else:
        # Other GCC countries: Sunday-Thursday
        trading_days = all_dates[all_dates.weekday.isin([6, 0, 1, 2, 3])]

    # Exclude holidays if requested
    if exclude_holidays and holiday_df is not None:
        # Filter holidays for this calendar
        if "calendars" in holiday_df.columns:
            # Check if calendar is in the list
            if calendar != "GCC":
                mask = holiday_df["calendars"].apply(
                    lambda x: calendar in x if isinstance(x, list) else calendar in str(x)
                )
                holidays = holiday_df[mask]["date"]
            else:
                # GCC: Use all holidays
                holidays = holiday_df["date"]

            holidays = pd.to_datetime(holidays)
            trading_days = trading_days[~trading_days.isin(holidays)]

    return trading_days


def align_to_trading_days(
    df: pd.DataFrame,
    calendar: str = "GCC",
    method: str = "drop",
    holiday_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Align DataFrame to trading days only.

    Args:
        df: DataFrame with DatetimeIndex.
        calendar: Trading calendar to use.
        method: How to handle non-trading days ('drop' or 'ffill').
        holiday_df: Optional holiday DataFrame.

    Returns:
        DataFrame aligned to trading days.

    Example:
        >>> prices = align_to_trading_days(prices_df, calendar="UAE", method="drop")
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Get trading days
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    trading_days = get_trading_days(start_date, end_date, calendar, holiday_df=holiday_df)

    if method == "drop":
        # Keep only trading days
        df = df[df.index.isin(trading_days)]
    elif method == "ffill":
        # Reindex to trading days and forward-fill
        df = df.reindex(trading_days, method="ffill")
    else:
        raise ValueError(f"Unknown method: {method}")

    return df
