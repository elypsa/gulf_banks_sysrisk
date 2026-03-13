"""Rolling window estimation for GARCH-DCC-LRMES models.

This module implements time-series SRISK calculation by re-estimating
GARCH-DCC-LRMES models over rolling windows. This produces panel data
(window_date × bank_ric) suitable for time-varying systemic risk monitoring.

Architecture:
- Fixed window size (default 5 years = 1260 trading days)
- Weekly step size (default 5 trading days)
- Parallel processing across banks within each window
- Forward-fill error handling for robustness

Mathematical approach:
    For each window t:
        1. Extract returns_{t-w:t} (w = window size)
        2. Estimate benchmark GARCH(1,1) once
        3. For each bank (in parallel):
            - Estimate bank GARCH(1,1)
            - Estimate bivariate DCC
            - Simulate 50K+ paths
            - Calculate LRMES
        4. Compile panel row: [window_date, bank_ric, lrmes, ...]

Output: Panel DataFrame with rolling LRMES estimates for all banks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from src.models.garch import (
    estimate_garch_multivariate,
    extract_standardized_residuals,
    garch_diagnostics
)
from src.models.dcc import estimate_dcc, dcc_diagnostics, estimate_ccc
from src.models.lrmes import (
    simulate_bivariate_garch_dcc,
    calculate_lrmes_from_bivariate
)
from src.utils.config import CONFIG


class RollingWindowEstimator:
    """Orchestrates rolling window GARCH-DCC-LRMES estimation.

    This class manages the rolling window estimation process, including:
    - Window date generation
    - Data extraction for each window
    - Parallel bank processing
    - Error handling and forward-fill
    - Panel data compilation

    Attributes:
        bank_returns: DataFrame of bank returns (date × bank_ric).
        benchmark_returns: DataFrame of benchmark returns (date × benchmark_ric).
        benchmark_name: Name of benchmark column to use.
        window_days: Fixed window size in trading days.
        step_days: Step size between windows in trading days.
        n_jobs: Number of parallel jobs (-1 = all CPUs).
        min_crisis_paths: Minimum crisis scenarios required.

    Example:
        >>> estimator = RollingWindowEstimator(
        ...     bank_returns=bank_returns,
        ...     benchmark_returns=benchmark_returns,
        ...     benchmark_name=".GPDGC"
        ... )
        >>> panel = estimator.run_rolling_estimation()
        >>> print(panel.head())
    """

    def __init__(
        self,
        bank_returns: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        benchmark_name: str,
        window_days: int = None,
        step_days: int = None,
        n_jobs: int = None,
        min_crisis_paths: int = None
    ):
        """Initialize rolling window estimator.

        Args:
            bank_returns: DataFrame with bank returns (date × bank_ric).
            benchmark_returns: DataFrame with benchmark returns (date × benchmark_ric).
            benchmark_name: Column name of benchmark to use.
            window_days: Window size in trading days (default from CONFIG).
            step_days: Step size in trading days (default from CONFIG).
            n_jobs: Number of parallel jobs (default from CONFIG).
            min_crisis_paths: Minimum crisis scenarios (default from CONFIG).
        """
        self.bank_returns = bank_returns
        self.benchmark_returns = benchmark_returns
        self.benchmark_name = benchmark_name

        # Configuration
        self.window_days = window_days or CONFIG.ROLLING_WINDOW_DAYS
        self.step_days = step_days or CONFIG.ROLLING_STEP_DAYS
        self.n_jobs = n_jobs or CONFIG.ROLLING_N_JOBS
        self.min_crisis_paths = min_crisis_paths or CONFIG.ROLLING_MIN_CRISIS_PATHS

        # Validate
        if benchmark_name not in benchmark_returns.columns:
            raise ValueError(f"Benchmark {benchmark_name} not found in benchmark_returns")

        # Align indices
        common_dates = bank_returns.index.intersection(benchmark_returns.index)
        self.bank_returns = bank_returns.loc[common_dates]
        self.benchmark_returns = benchmark_returns.loc[common_dates]

        print(f"Rolling Window Estimator initialized:")
        print(f"  Window size: {self.window_days} days ({self.window_days/252:.1f} years)")
        print(f"  Step size: {self.step_days} days ({self.step_days/5:.0f} weeks)")
        print(f"  Date range: {self.bank_returns.index.min()} to {self.bank_returns.index.max()}")
        print(f"  Total observations: {len(self.bank_returns)}")
        print(f"  Banks: {len(self.bank_returns.columns)}")
        print(f"  Parallel jobs: {self.n_jobs if self.n_jobs > 0 else 'all CPUs'}")

    def generate_window_dates(self) -> List[pd.Timestamp]:
        """Generate list of window end dates with weekly steps.

        Windows are defined by their END date. Each window contains
        [end_date - window_days : end_date] (inclusive).

        Returns:
            List of window end dates.

        Example:
            >>> dates = estimator.generate_window_dates()
            >>> print(f"Generated {len(dates)} windows")
            >>> print(f"First window ends: {dates[0]}")
            >>> print(f"Last window ends: {dates[-1]}")
        """
        dates = self.bank_returns.index
        min_date = dates.min()
        max_date = dates.max()

        # First window end date (need window_days of history)
        first_window_end_idx = self.window_days

        # Generate window end indices with step size
        window_end_indices = range(
            first_window_end_idx,
            len(dates),
            self.step_days
        )

        # Convert to dates
        window_dates = [dates[idx] for idx in window_end_indices]

        print(f"\nGenerated {len(window_dates)} rolling windows:")
        print(f"  First window: {dates[first_window_end_idx - self.window_days]} to {window_dates[0]}")
        print(f"  Last window: {dates[-self.window_days]} to {window_dates[-1]}")
        print(f"  Total windows: {len(window_dates)}")

        return window_dates

    def get_window_data(
        self,
        window_end_date: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract data for a single window.

        Args:
            window_end_date: End date of the window.

        Returns:
            Tuple of (bank_returns_window, benchmark_returns_window).

        Example:
            >>> bank_win, bench_win = estimator.get_window_data(pd.Timestamp('2023-12-31'))
            >>> print(f"Window size: {len(bank_win)} days")
        """
        # Find position of end date
        end_idx = self.bank_returns.index.get_loc(window_end_date)
        start_idx = end_idx - self.window_days + 1

        # Extract window
        bank_window = self.bank_returns.iloc[start_idx:end_idx+1]
        benchmark_window = self.benchmark_returns[self.benchmark_name].iloc[start_idx:end_idx+1]

        return bank_window, benchmark_window

    def estimate_window(
        self,
        window_date: pd.Timestamp,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Estimate GARCH-DCC-LRMES for all banks in one window.

        This method:
        1. Extracts window data
        2. Estimates benchmark GARCH once
        3. Processes all banks in parallel
        4. Compiles results into DataFrame

        Args:
            window_date: End date of the window.
            verbose: Print detailed progress.

        Returns:
            DataFrame with one row per bank containing:
                - window_date: End date of estimation window
                - bank_ric: Bank identifier
                - lrmes: Long-Run Marginal Expected Shortfall
                - n_crisis_scenarios: Number of crisis paths
                - avg_correlation: Average bank-market correlation
                - garch_persistence: Bank GARCH persistence (α + β + γ/2)
                - dcc_persistence: DCC persistence (a + b)
                - convergence: Whether estimation converged

        Example:
            >>> results = estimator.estimate_window(pd.Timestamp('2023-12-31'))
            >>> print(results[['bank_ric', 'lrmes', 'convergence']])
        """
        if verbose:
            print(f"\nProcessing window ending {window_date}")

        # Extract window data
        bank_window, benchmark_window = self.get_window_data(window_date)

        # Estimate benchmark GARCH once
        try:
            benchmark_garch_dict = estimate_garch_multivariate(
                pd.DataFrame({self.benchmark_name: benchmark_window}),
                p=CONFIG.GARCH_P,
                q=CONFIG.GARCH_Q,
                mean_model="Constant",
                dist="normal",
                verbose=False
            )
            benchmark_garch_result = benchmark_garch_dict[self.benchmark_name]

            # Check convergence
            bench_diag = garch_diagnostics(benchmark_garch_dict)
            if not bench_diag['convergence'].iloc[0]:
                print(f"  ⚠ Benchmark GARCH did not converge for window {window_date}")
                return pd.DataFrame()  # Return empty if benchmark fails

        except Exception as e:
            print(f"  ✗ Benchmark GARCH failed for window {window_date}: {e}")
            return pd.DataFrame()

        # Process all banks in parallel
        bank_rics = bank_window.columns.tolist()

        def process_single_bank(bank_ric: str) -> Optional[Dict]:
            """Process one bank (called in parallel)."""
            try:
                return self._process_bank(
                    bank_ric=bank_ric,
                    bank_series=bank_window[bank_ric],
                    benchmark_garch_result=benchmark_garch_result,
                    verbose=False
                )
            except Exception as e:
                if verbose:
                    print(f"    ✗ {bank_ric}: {e}")
                return None

        # Parallel execution
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(process_single_bank)(ric) for ric in bank_rics
        )

        # Filter out None results and compile
        results = [r for r in results if r is not None]

        if len(results) == 0:
            print(f"  ⚠ No successful estimations for window {window_date}")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(results)
        df['window_date'] = window_date

        # Reorder columns
        cols = ['window_date', 'bank_ric', 'lrmes', 'n_crisis_scenarios',
                'crisis_probability', 'avg_bank_loss_in_crisis', 'avg_market_loss_in_crisis',
                'avg_correlation', 'garch_persistence', 'dcc_a', 'dcc_b',
                'dcc_persistence', 'convergence']
        df = df[cols]

        if verbose:
            successful = df['convergence'].sum()
            print(f"  ✓ Window {window_date}: {successful}/{len(bank_rics)} banks successful")

        return df

    def _process_bank(
        self,
        bank_ric: str,
        bank_series: pd.Series,
        benchmark_garch_result: Dict,
        verbose: bool = False
    ) -> Optional[Dict]:
        """Process one bank using bivariate GARCH-DCC.

        This is the core estimation logic for a single bank.
        Follows Section 5.2 methodology with bivariate simulation.

        Args:
            bank_ric: Bank RIC identifier.
            bank_series: Bank return series for this window.
            benchmark_garch_result: Pre-estimated benchmark GARCH.
            verbose: Print progress.

        Returns:
            Dictionary with LRMES and diagnostics, or None if failed.
        """
        try:
            # 1. Estimate bank GARCH
            bank_garch_dict = estimate_garch_multivariate(
                pd.DataFrame({bank_ric: bank_series}),
                p=CONFIG.GARCH_P,
                q=CONFIG.GARCH_Q,
                mean_model="Constant",
                dist="normal",
                verbose=False
            )
            bank_garch_result = bank_garch_dict[bank_ric]

            # Check convergence
            bank_diag = garch_diagnostics(bank_garch_dict)
            if not bank_diag['convergence'].iloc[0]:
                return None

            # 2. Extract standardized residuals for bivariate DCC
            bivariate_garch = {
                bank_ric: bank_garch_result,
                self.benchmark_name: benchmark_garch_result
            }
            std_resids = extract_standardized_residuals(bivariate_garch)
            std_resids = std_resids[[bank_ric, self.benchmark_name]]

            # 3. Estimate bivariate DCC
            try:
                dcc_result = estimate_dcc(std_resids, initial_params=[0.01, 0.95])

                if not dcc_result["success"]:
                    # Fallback to CCC
                    dcc_result = estimate_ccc(std_resids)
            except Exception:
                # Fallback to CCC
                dcc_result = estimate_ccc(std_resids)

            # Get average correlation
            dcc_diag = dcc_diagnostics(dcc_result, [bank_ric, self.benchmark_name])
            avg_corr = dcc_diag['avg_correlation'].iloc[0]

            # 4. Simulate bivariate GARCH-DCC
            simulation = simulate_bivariate_garch_dcc(
                bank_garch_result=bank_garch_result,
                market_garch_result=benchmark_garch_result,
                dcc_result=dcc_result,
                horizon=CONFIG.CRISIS_HORIZON_WEEKS,
                n_simulations=CONFIG.N_SIMULATIONS,
                random_seed=None
            )

            # 5. Calculate LRMES
            lrmes_dict = calculate_lrmes_from_bivariate(
                simulation_result=simulation,
                crisis_threshold=CONFIG.CRISIS_THRESHOLD,
                min_crisis_scenarios=self.min_crisis_paths
            )

            # 6. Compile result
            result = {
                'bank_ric': bank_ric,
                'lrmes': lrmes_dict['lrmes'],
                'n_crisis_scenarios': lrmes_dict['n_crisis_scenarios'],
                'crisis_probability': lrmes_dict['crisis_probability'],
                'avg_bank_loss_in_crisis': lrmes_dict['avg_bank_loss_in_crisis'],
                'avg_market_loss_in_crisis': lrmes_dict['avg_market_loss_in_crisis'],
                'garch_persistence': bank_diag['persistence'].iloc[0],
                'dcc_a': dcc_result['params'][0],
                'dcc_b': dcc_result['params'][1],
                'dcc_persistence': sum(dcc_result['params']),
                'avg_correlation': lrmes_dict['avg_correlation'],
                'convergence': True
            }

            return result

        except Exception as e:
            if verbose:
                print(f"  ✗ {bank_ric}: {e}")
            return None

    def run_rolling_estimation(
        self,
        save_intermediate: bool = True,
        intermediate_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Run rolling window estimation for all windows.

        Main orchestration method that:
        1. Generates window dates
        2. Processes each window sequentially
        3. Handles errors with forward-fill
        4. Compiles panel data
        5. Optionally saves intermediate results

        Args:
            save_intermediate: Whether to save results after each window.
            intermediate_path: Path for intermediate saves (default: results dir).

        Returns:
            Panel DataFrame with columns:
                - window_date: End date of estimation window
                - bank_ric: Bank identifier
                - lrmes: Long-Run Marginal Expected Shortfall
                - ... (additional diagnostics)

        Example:
            >>> panel = estimator.run_rolling_estimation()
            >>> print(f"Panel shape: {panel.shape}")
            >>> print(f"Windows: {panel['window_date'].nunique()}")
            >>> print(f"Banks: {panel['bank_ric'].nunique()}")
        """
        print("\n" + "=" * 80)
        print("ROLLING WINDOW GARCH-DCC-LRMES ESTIMATION")
        print("=" * 80)

        # Generate window dates
        window_dates = self.generate_window_dates()

        # Storage for results
        panel_list = []
        last_successful = {}  # For forward-fill

        # Process each window with progress bar
        for window_date in tqdm(window_dates, desc="Processing windows"):
            try:
                # Estimate window
                window_results = self.estimate_window(window_date, verbose=False)

                if len(window_results) == 0:
                    # Window failed completely - forward fill
                    print(f"  ⚠ Window {window_date} failed, forward-filling from previous")

                    if last_successful:
                        # Create forward-filled results
                        ff_results = last_successful.copy()
                        ff_results['window_date'] = window_date
                        panel_list.append(ff_results)
                    continue

                # Check for individual bank failures and forward-fill
                successful_rics = set(window_results['bank_ric'])
                all_rics = set(self.bank_returns.columns)
                failed_rics = all_rics - successful_rics

                if failed_rics and last_successful is not None:
                    # Forward-fill failed banks from previous window
                    prev_df = last_successful
                    if len(prev_df) > 0:
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
        print("ROLLING ESTIMATION COMPLETE")
        print("=" * 80)
        print(f"Total windows: {panel['window_date'].nunique()}")
        print(f"Total banks: {panel['bank_ric'].nunique()}")
        print(f"Total observations: {len(panel)}")
        print(f"Date range: {panel['window_date'].min()} to {panel['window_date'].max()}")
        print(f"Convergence rate: {panel['convergence'].mean()*100:.1f}%")
        print(f"\nAverage LRMES: {panel['lrmes'].mean():.4f} ± {panel['lrmes'].std():.4f}")
        print(f"LRMES range: [{panel['lrmes'].min():.4f}, {panel['lrmes'].max():.4f}]")

        return panel
