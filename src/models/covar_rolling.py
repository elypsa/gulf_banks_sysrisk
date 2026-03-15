"""Rolling window estimation for CoVaR systemic risk measurement.

This module implements time-series CoVaR calculation by re-estimating
CoVaR models over rolling windows. This produces panel data
(window_date × bank_ric) suitable for time-varying systemic risk monitoring.

Architecture:
- Fixed window size (default 5 years = 1260 trading days)
- Weekly step size (default 5 trading days)
- For each window:
    1. Calculate system return for each bank
    2. Prepare state variables (lagged)
    3. Estimate bank VaR (5% and 50%)
    4. Estimate system CoVaR
    5. Calculate ΔCoVaR
- Compile panel data with rolling CoVaR estimates

Output: Panel DataFrame with rolling CoVaR estimates for all banks.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from tqdm import tqdm

from src.models.covar import (
    calculate_system_return,
    calculate_rolling_volatility,
    prepare_state_variables,
    calculate_delta_covar
)
from src.utils.config import CONFIG


class RollingCoVaREstimator:
    """Orchestrates rolling window CoVaR estimation.

    This class manages the rolling window estimation process, including:
    - Window date generation
    - Data extraction for each window
    - State variable preparation
    - CoVaR estimation for all banks
    - Panel data compilation

    Attributes:
        bank_returns: DataFrame of bank returns (date × bank_ric).
        market_caps: DataFrame of market capitalizations (date × bank_ric).
        fred_indicators: DataFrame of FRED indicators.
        benchmark_returns: Series of benchmark returns.
        benchmark_vol: Series of benchmark rolling volatility.
        window_days: Fixed window size in trading days.
        step_days: Step size between windows in trading days.
        start_date: Start date for rolling windows.

    Example:
        >>> estimator = RollingCoVaREstimator(
        ...     bank_returns=bank_returns,
        ...     market_caps=market_caps,
        ...     fred_indicators=fred_indicators,
        ...     benchmark_returns=benchmark_returns,
        ...     benchmark_vol=benchmark_vol
        ... )
        >>> panel = estimator.run_rolling_estimation()
    """

    def __init__(
        self,
        bank_returns: pd.DataFrame,
        market_caps: pd.DataFrame,
        fred_indicators: pd.DataFrame,
        benchmark_returns: pd.Series,
        benchmark_vol: pd.Series,
        window_days: int = None,
        step_days: int = None,
        start_date: str = None
    ):
        """Initialize rolling CoVaR estimator.

        Args:
            bank_returns: DataFrame with bank returns (date × bank_ric).
            market_caps: DataFrame with market capitalizations (date × bank_ric).
            fred_indicators: DataFrame with FRED indicators.
            benchmark_returns: Series with benchmark returns.
            benchmark_vol: Series with benchmark rolling volatility.
            window_days: Window size in trading days (default from CONFIG).
            step_days: Step size in trading days (default from CONFIG).
            start_date: Start date for rolling windows (default from CONFIG).
        """
        self.bank_returns = bank_returns
        self.market_caps = market_caps
        self.fred_indicators = fred_indicators
        self.benchmark_returns = benchmark_returns
        self.benchmark_vol = benchmark_vol

        # Configuration
        self.window_days = window_days or CONFIG.ROLLING_WINDOW_DAYS
        self.step_days = step_days or CONFIG.ROLLING_STEP_DAYS
        self.start_date = pd.Timestamp(start_date) if start_date else pd.Timestamp(CONFIG.ROLLING_WINDOW_START_DATE)

        # Align all data to common dates
        common_dates = (bank_returns.index
                       .intersection(market_caps.index)
                       .intersection(fred_indicators.index)
                       .intersection(benchmark_returns.index)
                       .intersection(benchmark_vol.index))

        self.bank_returns = bank_returns.loc[common_dates]
        self.market_caps = market_caps.loc[common_dates]
        self.fred_indicators = fred_indicators.loc[common_dates]
        self.benchmark_returns = benchmark_returns.loc[common_dates]
        self.benchmark_vol = benchmark_vol.loc[common_dates]

        print(f"Rolling CoVaR Estimator initialized:")
        print(f"  Window size: {self.window_days} days ({self.window_days/252:.1f} years)")
        print(f"  Step size: {self.step_days} days ({self.step_days/5:.0f} weeks)")
        print(f"  Window start date: {self.start_date.date()} (window end dates >= this date)")
        print(f"  Date range: {self.bank_returns.index.min()} to {self.bank_returns.index.max()}")
        print(f"  Total observations: {len(self.bank_returns)}")
        print(f"  Banks: {len(self.bank_returns.columns)}")

    def generate_window_dates(self) -> List[pd.Timestamp]:
        """Generate list of window end dates with weekly steps.

        Windows are defined by their END date. Each window contains
        [end_date - window_days : end_date] (inclusive).

        Only windows with end_date >= self.start_date are included.

        Returns:
            List of window end dates.
        """
        dates = self.bank_returns.index

        # First window end date (need window_days of history)
        first_window_end_idx = self.window_days

        # Generate window end indices with step size
        window_end_indices = range(
            first_window_end_idx,
            len(dates),
            self.step_days
        )

        # Convert to dates
        all_window_dates = [dates[idx] for idx in window_end_indices]

        # Filter to only include windows with end_date >= start_date
        window_dates = [d for d in all_window_dates if d >= self.start_date]

        if len(window_dates) == 0:
            raise ValueError(
                f"No windows found with end_date >= {self.start_date.date()}. "
                f"Available window range: {all_window_dates[0].date()} to {all_window_dates[-1].date()}"
            )

        print(f"\nGenerated {len(window_dates)} rolling windows:")
        print(f"  Total possible windows: {len(all_window_dates)}")
        print(f"  Filtered to start_date >= {self.start_date.date()}: {len(window_dates)} windows")

        # Get first filtered window details
        first_window_idx = self.bank_returns.index.get_loc(window_dates[0])
        first_window_start = dates[first_window_idx - self.window_days + 1]

        print(f"  First window: {first_window_start.date()} to {window_dates[0].date()}")
        print(f"  Last window: {dates[self.bank_returns.index.get_loc(window_dates[-1]) - self.window_days + 1].date()} to {window_dates[-1].date()}")

        return window_dates

    def get_window_data(
        self,
        window_end_date: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Extract data for a single window.

        Args:
            window_end_date: End date of the window.

        Returns:
            Tuple of (bank_returns_window, market_caps_window, fred_window,
                     benchmark_returns_window, benchmark_vol_window).
        """
        # Find position of end date
        end_idx = self.bank_returns.index.get_loc(window_end_date)
        start_idx = end_idx - self.window_days + 1

        # Extract window
        bank_ret_window = self.bank_returns.iloc[start_idx:end_idx+1]
        market_cap_window = self.market_caps.iloc[start_idx:end_idx+1]
        fred_window = self.fred_indicators.iloc[start_idx:end_idx+1]
        bench_ret_window = self.benchmark_returns.iloc[start_idx:end_idx+1]
        bench_vol_window = self.benchmark_vol.iloc[start_idx:end_idx+1]

        return bank_ret_window, market_cap_window, fred_window, bench_ret_window, bench_vol_window

    def estimate_window(
        self,
        window_date: pd.Timestamp,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Estimate CoVaR for all banks in one window.

        Args:
            window_date: End date of the window.
            verbose: Print detailed progress.

        Returns:
            DataFrame with one row per bank containing:
                - window_date: End date of estimation window
                - bank_ric: Bank identifier
                - var_5pct: Bank's 5% VaR
                - var_50pct: Bank's 50% VaR
                - covar_5pct: System's 5% CoVaR (bank at 5% VaR)
                - covar_50pct: System's 50% CoVaR (bank at 50% VaR)
                - beta_i: Bank's coefficient in system equation
                - delta_covar: ΔCoVaR (marginal systemic risk)
                - n_obs: Number of observations
                - converged: Whether estimation converged
        """
        if verbose:
            print(f"\nProcessing window ending {window_date}")

        # Extract window data
        bank_ret_win, market_cap_win, fred_win, bench_ret_win, bench_vol_win = self.get_window_data(window_date)

        # Prepare state variables for this window
        state_vars = prepare_state_variables(
            fred_indicators=fred_win,
            benchmark_returns=bench_ret_win,
            benchmark_vol=bench_vol_win
        )

        # Process each bank
        results = []

        for bank_ric in bank_ret_win.columns:
            try:
                # Calculate system return (excluding this bank)
                system_return = calculate_system_return(
                    bank_ret_win,
                    market_cap_win,
                    exclude_bank=bank_ric
                )

                # Get bank return
                bank_return = bank_ret_win[bank_ric]

                # Calculate CoVaR
                result = calculate_delta_covar(
                    bank_ric=bank_ric,
                    bank_return=bank_return,
                    state_variables=state_vars,
                    system_return=system_return
                )

                if result is not None:
                    results.append(result)

            except Exception as e:
                if verbose:
                    print(f"    ✗ {bank_ric}: {e}")
                continue

        if len(results) == 0:
            if verbose:
                print(f"  ⚠ No successful estimations for window {window_date}")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(results)
        df['window_date'] = window_date

        # Reorder columns
        cols = ['window_date', 'bank_ric', 'delta_covar', 'var_5pct', 'var_50pct',
                'covar_5pct', 'covar_50pct', 'beta_i', 'n_obs', 'pseudo_r2_covar', 'converged']
        df = df[cols]

        if verbose:
            successful = df['converged'].sum()
            print(f"  ✓ Window {window_date}: {successful}/{len(bank_ret_win.columns)} banks successful")

        return df

    def run_rolling_estimation(
        self,
        save_intermediate: bool = True,
        intermediate_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Run rolling window CoVaR estimation for all windows.

        Args:
            save_intermediate: Whether to save results after each window.
            intermediate_path: Path for intermediate saves.

        Returns:
            Panel DataFrame with columns:
                - window_date: End date of estimation window
                - bank_ric: Bank identifier
                - delta_covar: ΔCoVaR (main metric)
                - var_5pct, var_50pct: Bank VaR at 5% and 50%
                - covar_5pct, covar_50pct: System CoVaR at 5% and 50%
                - beta_i: Bank coefficient
                - n_obs: Observations
                - converged: Convergence flag
        """
        print("\n" + "=" * 80)
        print("ROLLING WINDOW CoVaR ESTIMATION")
        print("=" * 80)

        # Generate window dates
        window_dates = self.generate_window_dates()

        # Storage for results
        panel_list = []
        last_successful = None

        # Process each window with progress bar
        for window_date in tqdm(window_dates, desc="Processing windows"):
            try:
                # Estimate window
                window_results = self.estimate_window(window_date, verbose=False)

                if len(window_results) == 0:
                    # Window failed completely - forward fill
                    print(f"  ⚠ Window {window_date} failed, forward-filling from previous")

                    if last_successful is not None and len(last_successful) > 0:
                        ff_results = last_successful.copy()
                        ff_results['window_date'] = window_date
                        panel_list.append(ff_results)
                    continue

                # Check for individual bank failures and forward-fill
                successful_rics = set(window_results['bank_ric'])
                all_rics = set(self.bank_returns.columns)
                failed_rics = all_rics - successful_rics

                if failed_rics and last_successful is not None and len(last_successful) > 0:
                    # Forward-fill failed banks from previous window
                    prev_df = last_successful
                    for ric in failed_rics:
                        prev_row = prev_df[prev_df['bank_ric'] == ric]
                        if len(prev_row) > 0:
                            ff_row = prev_row.copy()
                            ff_row['window_date'] = window_date
                            window_results = pd.concat([window_results, ff_row], ignore_index=True)

                # Store results
                panel_list.append(window_results)
                last_successful = window_results.copy()

                # Save intermediate
                if save_intermediate and intermediate_path:
                    temp_panel = pd.concat(panel_list, ignore_index=True)
                    temp_panel.to_parquet(intermediate_path, index=False)

            except Exception as e:
                print(f"  ✗ Window {window_date} error: {e}")
                # Forward-fill entire window
                if last_successful is not None and len(last_successful) > 0:
                    ff_results = last_successful.copy()
                    ff_results['window_date'] = window_date
                    panel_list.append(ff_results)

        # Compile final panel
        if len(panel_list) == 0:
            raise RuntimeError("No successful window estimations")

        panel = pd.concat(panel_list, ignore_index=True)

        # Summary statistics
        print("\n" + "=" * 80)
        print("ROLLING CoVaR ESTIMATION COMPLETE")
        print("=" * 80)
        print(f"Total windows: {panel['window_date'].nunique()}")
        print(f"Total banks: {panel['bank_ric'].nunique()}")
        print(f"Total observations: {len(panel)}")
        print(f"Date range: {panel['window_date'].min()} to {panel['window_date'].max()}")
        print(f"Convergence rate: {panel['converged'].mean()*100:.1f}%")
        print(f"\nAverage ΔCoVaR: {panel['delta_covar'].mean():.4f} ± {panel['delta_covar'].std():.4f}")
        print(f"ΔCoVaR range: [{panel['delta_covar'].min():.4f}, {panel['delta_covar'].max():.4f}]")

        return panel
