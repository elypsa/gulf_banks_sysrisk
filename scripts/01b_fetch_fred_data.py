#!/usr/bin/env python3
"""Script to fetch macroeconomic indicators from FRED.

This script fetches systemic risk indicators from the Federal Reserve Economic Data (FRED):
- VIXCLS: CBOE Volatility Index (market stress indicator)
- T10Y3M: 10-Year Treasury minus 3-Month Treasury spread (yield curve)

These indicators complement the SRISK analysis by providing macro context for systemic risk episodes.

Data is saved to data/raw/ in parquet format.

Requirements:
    - FRED API key in .env file (FRED_API_KEY=your_key_here)
    - Free API key available at: https://fred.stlouisfed.org/docs/api/api_key.html

Usage:
    uv run scripts/01b_fetch_fred_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fred_fetcher import fetch_systemic_risk_indicators
from src.data import save_dataframe
from src.utils.config import CONFIG, RAW_DATA_DIR
from datetime import date


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("FRED MACROECONOMIC INDICATORS FETCHER")
    print("=" * 60)
    print(f"\nSaving data to: {RAW_DATA_DIR}")

    try:
        # Fetch systemic risk indicators
        print("\n" + "=" * 60)
        print("FETCHING SYSTEMIC RISK INDICATORS FROM FRED")
        print("=" * 60)

        today_str = date.today().strftime("%Y-%m-%d")

        indicators = fetch_systemic_risk_indicators(
            start_date=CONFIG.DATA_START_DATE,
            end_date=today_str,
            verbose=True
        )

        # Save to parquet
        metadata = {
            "description": "Systemic risk indicators from FRED",
            "series": {
                "VIXCLS": "CBOE Volatility Index (market stress)",
                "T10Y3M": "10Y-3M Treasury spread (yield curve)"
            },
            "frequency": "Daily",
            "start_date": CONFIG.DATA_START_DATE,
            "end_date": today_str,
            "source": "Federal Reserve Economic Data (FRED)",
            "url": "https://fred.stlouisfed.org/"
        }

        save_dataframe(indicators, "fred_indicators", RAW_DATA_DIR, metadata)

        # Summary
        print("\n" + "=" * 60)
        print("✓ FRED DATA FETCHED SUCCESSFULLY")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Series fetched: {len(indicators.columns)}")
        print(f"  - Observations: {len(indicators)} days")
        print(f"  - Date range: {indicators.index.min()} to {indicators.index.max()}")
        print(f"\nSeries details:")
        for col in indicators.columns:
            non_null = indicators[col].notna().sum()
            print(f"  - {col}: {non_null} non-null observations")

        print(f"\nData saved to: {RAW_DATA_DIR / 'fred_indicators.parquet'}")
        print("\nNext step: Use FRED indicators in SRISK analysis for macro context")

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease create a .env file in the project root with:")
        print("  FRED_API_KEY=your_api_key_here")
        print("\nGet a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    except ValueError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nPlease add your FRED API key to .env file:")
        print("  FRED_API_KEY=your_api_key_here")
        print("\nGet a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
