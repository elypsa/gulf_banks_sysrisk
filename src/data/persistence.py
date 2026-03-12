"""Data persistence module for saving and loading LSEG data.

This module handles saving and loading of data in parquet format to avoid
redundant API calls. All data is timestamped for tracking updates.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    data_dir: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a DataFrame to parquet format with metadata.

    Args:
        df: DataFrame to save.
        filename: Name of the file (without extension).
        data_dir: Directory to save the file (e.g., RAW_DATA_DIR).
        metadata: Optional metadata dictionary to save alongside.

    Returns:
        None

    Example:
        >>> save_dataframe(prices_df, "prices_daily", RAW_DATA_DIR,
        ...                metadata={"source": "LSEG", "last_update": "2024-01-15"})
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / f"{filename}.parquet"

    # Add timestamp to metadata
    if metadata is None:
        metadata = {}
    metadata["saved_at"] = datetime.now().isoformat()

    # Save DataFrame
    df.to_parquet(filepath, index=True, compression="snappy")

    # Save metadata
    metadata_path = data_dir / f"{filename}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved {filename}.parquet to {data_dir}")
    print(f"  Shape: {df.shape}, Saved at: {metadata['saved_at']}")


def load_dataframe(
    filename: str,
    data_dir: Path,
    load_metadata: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a DataFrame from parquet format.

    Args:
        filename: Name of the file (without extension).
        data_dir: Directory containing the file.
        load_metadata: If True, also return metadata dictionary.

    Returns:
        DataFrame, or (DataFrame, metadata) if load_metadata=True.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> df = load_dataframe("prices_daily", RAW_DATA_DIR)
        >>> df, meta = load_dataframe("prices_daily", RAW_DATA_DIR, load_metadata=True)
    """
    filepath = data_dir / f"{filename}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"File {filename}.parquet not found in {data_dir}. "
            "Please run data fetching script first."
        )

    df = pd.read_parquet(filepath)
    print(f"✓ Loaded {filename}.parquet from {data_dir}")
    print(f"  Shape: {df.shape}")

    if load_metadata:
        metadata_path = data_dir / f"{filename}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        return df, metadata

    return df


def check_data_freshness(filename: str, data_dir: Path, max_age_days: int = 7) -> bool:
    """Check if saved data is recent enough.

    Args:
        filename: Name of the file (without extension).
        data_dir: Directory containing the file.
        max_age_days: Maximum age in days before considering stale.

    Returns:
        True if data exists and is fresh, False otherwise.

    Example:
        >>> if not check_data_freshness("prices_daily", RAW_DATA_DIR):
        ...     # Fetch fresh data
        ...     pass
    """
    metadata_path = data_dir / f"{filename}_metadata.json"

    if not metadata_path.exists():
        return False

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if "saved_at" not in metadata:
        return False

    saved_at = datetime.fromisoformat(metadata["saved_at"])
    age_days = (datetime.now() - saved_at).days

    is_fresh = age_days <= max_age_days
    print(f"Data age: {age_days} days (max: {max_age_days}), Fresh: {is_fresh}")

    return is_fresh


def list_saved_data(data_dir: Path) -> pd.DataFrame:
    """List all saved parquet files with their metadata.

    Args:
        data_dir: Directory to scan for parquet files.

    Returns:
        DataFrame with columns: filename, shape, saved_at, age_days.

    Example:
        >>> summary = list_saved_data(RAW_DATA_DIR)
        >>> print(summary)
    """
    records = []

    for filepath in data_dir.glob("*.parquet"):
        filename = filepath.stem

        # Load file to get shape
        df = pd.read_parquet(filepath)

        # Load metadata if exists
        metadata_path = data_dir / f"{filename}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            saved_at = metadata.get("saved_at", "unknown")
            if saved_at != "unknown":
                age_days = (datetime.now() - datetime.fromisoformat(saved_at)).days
            else:
                age_days = None
        else:
            saved_at = "unknown"
            age_days = None

        records.append({
            "filename": filename,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "saved_at": saved_at,
            "age_days": age_days
        })

    return pd.DataFrame(records)
