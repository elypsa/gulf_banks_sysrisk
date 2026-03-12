"""Data fetching and persistence modules."""

from .lseg_fetcher import (
    lseg_session,
    get_gcc_bank_universe,
    get_fundamentals,
    get_market_data,
    get_benchmarks,
    get_holidays,
    get_fx_rates
)
from .persistence import (
    save_dataframe,
    load_dataframe,
    check_data_freshness,
    list_saved_data
)
from .data_quality import (
    generate_missing_data_report,
    print_data_quality_summary
)

__all__ = [
    "lseg_session",
    "get_gcc_bank_universe",
    "get_fundamentals",
    "get_market_data",
    "get_benchmarks",
    "get_holidays",
    "get_fx_rates",
    "save_dataframe",
    "load_dataframe",
    "check_data_freshness",
    "list_saved_data",
    "generate_missing_data_report",
    "print_data_quality_summary"
]
