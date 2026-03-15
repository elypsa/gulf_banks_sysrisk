"""Configuration module for SRISK calculation.

This module contains all configuration parameters for the SRISK calculation pipeline,
including GARCH-DCC settings, LRMES simulation parameters, and prudential ratios.
"""

from dataclasses import dataclass
from typing import List
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DATA_DIR = DATA_DIR / "results"


@dataclass
class SRISKConfig:
    """Configuration for SRISK calculation parameters."""

    # Prudential capital ratios
    CAPITAL_RATIO_BASEL: float = 0.08  # 8% Basel standard
    CAPITAL_RATIO_IFRS: float = 0.055  # 5.5% IFRS adjusted

    # LRMES parameters
    CRISIS_HORIZON_WEEKS: int = 22  # 6 months in trading weeks
    CRISIS_THRESHOLD: float = -0.15  # -15% market decline (percentage, converted to log return internally)
    N_SIMULATIONS: int = 40000  # Monte Carlo paths
    RANDOM_SEED: int = 42  # For reproducibility

    # Note on CRISIS_THRESHOLD:
    # - Specified as percentage decline (e.g., -0.15 = -15%)
    # - Internally converted to log return: ln(1 - 0.15) = ln(0.85) ≈ -0.1625
    # - Standard SRISK literature uses -40% over 6 months

    # GARCH-DCC parameters
    GARCH_P: int = 1  # GARCH order
    GARCH_Q: int = 1  # GARCH order
    GARCH_POWER: float = 2.0  # Standard GARCH
    VOLATILITY_MODEL: str = "GJR-GARCH"  # Asymmetric GARCH

    # DCC parameters
    DCC_ENABLED: bool = True  # Use DCC, else CCC

    # CoVaR parameters
    COVAR_VAR_QUANTILE: float = 0.05  # Quantile for bank VaR estimation (5%)
    COVAR_SYSTEM_QUANTILE: float = 0.05  # Quantile for system CoVaR estimation (5%)
    COVAR_MEDIAN_QUANTILE: float = 0.50  # Median quantile for ΔCoVaR calculation (50%)
    COVAR_VOL_WINDOW: int = 252  # Rolling window for benchmark volatility (1 year)

    # Rolling window parameters
    ROLLING_WINDOW_ENABLED: bool = True  # Enable rolling window estimation
    ROLLING_WINDOW_DAYS: int = 1260  # 5 years of trading days (252 * 5)
    ROLLING_STEP_DAYS: int = 5  # Weekly step size (5 trading days)
    ROLLING_WINDOW_START_DATE: str = "2026-01-01"  # Start date for rolling window calculations (window end dates >= this date)
    ROLLING_N_JOBS: int = -1  # Number of parallel jobs (-1 = all CPUs)
    ROLLING_MIN_CRISIS_PATHS: int = 500  # Minimum crisis scenarios required

    # Data parameters
    DATA_START_DATE: str = "2005-12-31"  # Unified start date for market data, fundamentals, benchmarks, FX
    HOLIDAYS_START_DATE: str = "2016-01-01"  # LSEG API limitation: cannot retrieve holidays before 2016-01-01
    FUNDAMENTAL_LAG_DAYS: int = 45  # Reporting delay for quarterly data
    MIN_TRADING_DAYS: int = 252  # Minimum history required (1 year)
    RECOMMENDED_TRADING_DAYS: int = 500  # Recommended history (2 years)

    # GCC countries
    GCC_COUNTRIES: List[str] = None

    # Benchmark RICs
    BENCHMARK_BROAD: str = ".GPDGC"  # S&P GCC Composite
    BENCHMARK_GCC_BANKING: str = ".TRXFLDGCPUBANK"  # GCC Banking Index

    # Country banking indices
    COUNTRY_INDICES: dict = None

    # Holiday calendars
    HOLIDAY_CALENDARS: List[str] = None

    # UAE calendar transition
    UAE_TRANSITION_DATE: str = "2022-01-01"

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.GCC_COUNTRIES is None:
            self.GCC_COUNTRIES = ["UAE", "SAU", "QAT", "KWT", "OMN", "BAH"]

        if self.COUNTRY_INDICES is None:
            self.COUNTRY_INDICES = {
                "UAE": ".TRXFLDAEPBANK",
                "SAU": ".TRXFLDSAPBANK",
                "QAT": ".TRXFLDQAPBANK",
                "KWT": ".TRXFLDKWPBANK",
                "OMN": ".TRXFLDOMPBANK",
                "BAH": ".TRXFLDBHPBANK"
            }

        if self.HOLIDAY_CALENDARS is None:
            self.HOLIDAY_CALENDARS = ["UAE", "SAU", "KWT", "OMN", "BAH"]


# Global configuration instance
CONFIG = SRISKConfig()


# LSEG field mappings
LSEG_FIELDS = {
    "fundamentals": {
        "total_assets": "TR.F.TotAssets",
        "common_equity": "TR.F.ComEqTot",
        "total_liabilities": "TR.F.TotLiab",
        "total_liab_equity": "TR.F.TotLiabEq"
    },
    "market_data": {
        "price_close": "TR.PriceClose",
        "total_return": "TR.TotalReturn",
        "market_cap": "TR.CompanyMarketCapitalization"
    },
    "benchmark": {
        "price_close": "TR.PriceClose"
    },
    "index_info": {
        "geography": "TR.IndexGeography",
        "series_name": "TR.IndexSeriesName"
    }
}


# Country chain RICs for discovering bank universe
GCC_CHAINS = {
    "UAE": "0#.TRXFLDAEPBANK",
    "SAU": "0#.TRXFLDSAPBANK",
    "QAT": "0#.TRXFLDQAPBANK",
    "KWT": "0#.TRXFLDKWPBANK",
    "OMN": "0#.TRXFLDOMPBANK",
    "BAH": "0#.TRXFLDBHPBANK"
}


# Parent chain
GCC_PARENT_CHAIN = "0#.TRXFLDGCPUBANK"
