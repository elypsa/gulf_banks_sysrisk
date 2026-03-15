"""Incremental data fetching utilities for minimizing API usage.

This module provides utilities to fetch only new data since the last update,
avoiding redundant API calls for daily frequency datasets.
"""

from pathlib import Path
from typing import Callable, Dict, Any, Optional
import pandas as pd

from .persistence import load_dataframe, save_dataframe


def get_latest_date(df: pd.DataFrame) -> pd.Timestamp:
    """Extract the most recent date from a DataFrame's DatetimeIndex.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        The maximum (most recent) timestamp in the index.

    Raises:
        ValueError: If DataFrame doesn't have a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    return df.index.max()


def fetch_incremental_or_full(
    filename: str,
    data_dir: Path,
    fetch_function: Callable[[str], pd.DataFrame],
    full_start_date: str,
    metadata_template: Dict[str, Any],
    verbose: bool = True
) -> pd.DataFrame:
    """Fetch data incrementally if exists, otherwise fetch full history.

    This function implements smart incremental fetching:
    1. If parquet file exists: load it, get latest date, fetch only new data
    2. If parquet doesn't exist: fetch full history from full_start_date
    3. Merge old + new data, remove duplicates, and save

    Args:
        filename: Base filename (without .parquet extension).
        data_dir: Directory where parquet files are stored.
        fetch_function: Function that takes start_date (str) and returns DataFrame.
        full_start_date: Start date to use for initial full fetch (e.g., "2005-12-31").
        metadata_template: Dict of metadata to save with the data.
        verbose: If True, print progress messages.

    Returns:
        Combined DataFrame (existing + new data, or just full history if first fetch).

    Example:
        >>> df = fetch_incremental_or_full(
        ...     filename="benchmarks_daily",
        ...     data_dir=RAW_DATA_DIR,
        ...     fetch_function=lambda start_date: get_benchmarks(rics, start_date),
        ...     full_start_date="2005-12-31",
        ...     metadata_template={"description": "Benchmark indices"}
        ... )
    """
    filepath = data_dir / f"{filename}.parquet"

    if filepath.exists():
        # === INCREMENTAL FETCH ===
        if verbose:
            print(f"\n  📂 Found existing {filename}.parquet")

        try:
            # Load existing data with metadata
            existing_df, existing_metadata = load_dataframe(
                filename, data_dir, load_metadata=True
            )

            # Get latest date in existing data
            latest_date = get_latest_date(existing_df)

            # Calculate new start date (day after latest)
            new_start_date = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

            if verbose:
                print(f"  📅 Existing data: {existing_df.index.min().date()} to {latest_date.date()}")
                print(f"  🔄 Fetching incremental data from {new_start_date} to {today_str}...")

            # Fetch only new data
            new_df = fetch_function(new_start_date)

            # Check if any new data was returned
            if new_df.empty or len(new_df) == 0:
                if verbose:
                    print(f"  ✓ No new data available (already up to date)")
                return existing_df

            # Merge: concatenate along rows (axis=0) and sort by date
            combined_df = pd.concat([existing_df, new_df], axis=0).sort_index()

            # Remove duplicate dates (keep most recent value)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            if verbose:
                new_rows = len(combined_df) - len(existing_df)
                print(f"  ✓ Added {new_rows} new days (total: {len(combined_df)} days)")

            # Update metadata
            updated_metadata = metadata_template.copy()
            updated_metadata['incremental_update'] = True
            updated_metadata['last_incremental_fetch'] = pd.Timestamp.now().isoformat()
            updated_metadata['original_start_date'] = existing_metadata.get('start_date', full_start_date)

            # Save merged data
            save_dataframe(combined_df, filename, data_dir, updated_metadata)

            return combined_df

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error during incremental fetch: {e}")
                print(f"  🔄 Falling back to full fetch...")
            # Fall through to full fetch

    # === FULL FETCH (first time or error during incremental) ===
    if verbose:
        print(f"\n  📥 No existing data found. Fetching full history from {full_start_date}...")

    full_df = fetch_function(full_start_date)

    # Save with metadata
    full_metadata = metadata_template.copy()
    full_metadata['start_date'] = full_start_date
    full_metadata['initial_fetch'] = True

    save_dataframe(full_df, filename, data_dir, full_metadata)

    if verbose:
        print(f"  ✓ Fetched {len(full_df)} days of data")

    return full_df
