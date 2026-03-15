"""Conditional Value at Risk (CoVaR) estimation for systemic risk measurement.

This module implements the CoVaR methodology from Adrian & Brunnermeier (2016) to measure
each bank's contribution to systemic risk using quantile regression.

Mathematical Framework:
    Step 1 - Bank VaR estimation:
        r_{i,t} = α_i + β_i' X_{t-1} + ε_{i,t}

        Estimate at quantiles τ = 5% and τ = 50% using quantile regression.

    Step 2 - System CoVaR estimation:
        r_{system,t} = α + γ' X_{t-1} + δ_i r_{i,t} + ε_t

        Estimate at quantile τ = 5% using quantile regression.

    ΔCoVaR calculation:
        ΔCoVaR_i = CoVaR_{i,5%} - CoVaR_{i,50%}

        Measures marginal contribution of bank i to system tail risk.
        CoVaR_{i,5%} = system tail risk when bank i is in distress (5% VaR)
        CoVaR_{i,50%} = system tail risk when bank i is in normal state (50% VaR)

State Variables (X):
    - VIXCLS: Market volatility/stress
    - T10Y3M: Yield curve slope (recession indicator)
    - Benchmark return (lagged)
    - Benchmark rolling volatility (lagged)

System Return:
    Market-cap weighted average of all banks EXCLUDING bank i.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from statsmodels.regression.quantile_regression import QuantReg
import warnings

from src.utils.config import CONFIG


def calculate_system_return(
    bank_returns: pd.DataFrame,
    market_caps: pd.DataFrame,
    exclude_bank: Optional[str] = None
) -> pd.Series:
    """Calculate market-cap weighted system return.

    Args:
        bank_returns: DataFrame with bank returns (date × bank_ric).
        market_caps: DataFrame with market capitalizations (date × bank_ric).
        exclude_bank: Bank RIC to exclude from system (default: None).

    Returns:
        Series with system return (market-cap weighted average).

    Mathematical formulation:
        r_system,t = Σ w_{j,t} × r_{j,t}  for j ≠ i

        where w_{j,t} = MarketCap_{j,t} / Σ MarketCap_{k,t} (k ≠ i)

    Example:
        >>> system_ret = calculate_system_return(bank_returns, market_caps, exclude_bank='ENBD.DU')
    """
    # Get common columns (banks present in both datasets)
    common_banks = bank_returns.columns.intersection(market_caps.columns)

    # Exclude specified bank if provided
    if exclude_bank is not None and exclude_bank in common_banks:
        common_banks = common_banks.drop(exclude_bank)

    # Subset data
    returns_subset = bank_returns[common_banks]
    caps_subset = market_caps[common_banks]

    # Calculate weights (normalize to sum to 1 at each date)
    weights = caps_subset.div(caps_subset.sum(axis=1), axis=0)

    # Calculate weighted average return
    system_return = (returns_subset * weights).sum(axis=1)

    return system_return


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 252
) -> pd.Series:
    """Calculate rolling standard deviation (volatility).

    Args:
        returns: Return series.
        window: Rolling window size in days (default: 252 = 1 year).

    Returns:
        Series with rolling volatility.

    Example:
        >>> bench_vol = calculate_rolling_volatility(benchmark_returns['.GPDGC'], window=252)
    """
    return returns.rolling(window=window, min_periods=window//2).std()


def prepare_state_variables(
    fred_indicators: pd.DataFrame,
    benchmark_returns: pd.Series,
    benchmark_vol: pd.Series
) -> pd.DataFrame:
    """Prepare lagged state variables for CoVaR estimation.

    Args:
        fred_indicators: DataFrame with FRED indicators (VIXCLS, T10Y3M).
        benchmark_returns: Benchmark return series.
        benchmark_vol: Benchmark rolling volatility series.

    Returns:
        DataFrame with lagged state variables (all shifted by 1 day).

    State variables (all at t-1):
        - VIXCLS: Market volatility
        - T10Y3M: Yield curve slope
        - benchmark_return: Benchmark return
        - benchmark_vol: Benchmark volatility

    Example:
        >>> state_vars = prepare_state_variables(fred_indicators, bench_returns, bench_vol)
    """
    # Combine all state variables
    state_vars = pd.DataFrame({
        'VIXCLS': fred_indicators['VIXCLS'],
        'T10Y3M': fred_indicators['T10Y3M'],
        'benchmark_return': benchmark_returns,
        'benchmark_vol': benchmark_vol
    })

    # Lag all variables by 1 day (t-1)
    state_vars_lagged = state_vars.shift(1)

    # Drop NaN rows
    state_vars_lagged = state_vars_lagged.dropna()

    return state_vars_lagged


def estimate_bank_var(
    bank_return: pd.Series,
    state_variables: pd.DataFrame,
    quantile: float = 0.05
) -> Tuple[float, Dict]:
    """Estimate bank's Value at Risk (VaR) using quantile regression.

    Args:
        bank_return: Bank return series.
        state_variables: Lagged state variables DataFrame.
        quantile: Quantile for VaR estimation (default: 0.05).

    Returns:
        Tuple of (VaR_value, diagnostics_dict).

    Model:
        r_{i,t} = α + β' X_{t-1} + ε

        Estimated at quantile τ using quantile regression.

    Example:
        >>> var_5pct, diag = estimate_bank_var(bank_returns['ENBD.DU'], state_vars, quantile=0.05)
    """
    # Align data
    common_index = bank_return.index.intersection(state_variables.index)
    y = bank_return.loc[common_index]
    X = state_variables.loc[common_index]

    # Drop NaN
    mask = y.notna() & X.notna().all(axis=1)
    y = y[mask]
    X = X[mask]

    # Ensure numeric types
    y = pd.to_numeric(y, errors='coerce').dropna()
    X = X.apply(pd.to_numeric, errors='coerce').dropna()

    # Re-align after type conversion
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]

    if len(y) < 50:
        raise ValueError(f"Insufficient observations after cleaning: {len(y)}")

    # Add constant
    X = pd.DataFrame(X)
    X.insert(0, 'const', 1.0)

    # Convert to numpy arrays to ensure clean numeric data
    y_array = np.asarray(y, dtype=np.float64)
    X_array = np.asarray(X, dtype=np.float64)

    # Estimate quantile regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QuantReg(y_array, X_array)
        result = model.fit(q=quantile, max_iter=1000)

    # Calculate fitted VaR (using quantile of fitted values)
    fitted_values = result.fittedvalues
    var_value = np.quantile(fitted_values, quantile)

    # Diagnostics
    diagnostics = {
        'n_obs': len(y),
        'quantile': quantile,
        'pseudo_r2': result.prsquared if hasattr(result, 'prsquared') else 0.0,
        'converged': True
    }

    return float(var_value), diagnostics


def estimate_covar(
    system_return: pd.Series,
    state_variables: pd.DataFrame,
    bank_return: pd.Series,
    quantile: float = 0.05
) -> Tuple[object, Dict]:
    """Estimate system CoVaR conditional on bank i using quantile regression.

    Args:
        system_return: System return series (excluding bank i).
        state_variables: Lagged state variables DataFrame.
        bank_return: Bank i return series.
        quantile: Quantile for CoVaR estimation (default: 0.05).

    Returns:
        Tuple of (fitted_model, diagnostics_dict).

    Model:
        r_{system,t} = α + γ' X_{t-1} + δ_i r_{i,t} + ε

        Estimated at quantile τ using quantile regression.

    Example:
        >>> model, diag = estimate_covar(system_ret, state_vars, bank_ret, quantile=0.05)
    """
    # Align all data
    common_index = system_return.index.intersection(
        state_variables.index
    ).intersection(bank_return.index)

    y = system_return.loc[common_index]
    X_state = state_variables.loc[common_index]
    X_bank = bank_return.loc[common_index]

    # Drop NaN
    mask = y.notna() & X_state.notna().all(axis=1) & X_bank.notna()
    y = y[mask]
    X_state = X_state[mask]
    X_bank = X_bank[mask]

    # Ensure numeric types
    y = pd.to_numeric(y, errors='coerce').dropna()
    X_state = X_state.apply(pd.to_numeric, errors='coerce').dropna()
    X_bank = pd.to_numeric(X_bank, errors='coerce').dropna()

    # Re-align after type conversion
    common_idx = y.index.intersection(X_state.index).intersection(X_bank.index)
    y = y.loc[common_idx]
    X_state = X_state.loc[common_idx]
    X_bank = X_bank.loc[common_idx]

    if len(y) < 50:
        raise ValueError(f"Insufficient observations after cleaning: {len(y)}")

    # Combine state variables and bank return
    X = pd.DataFrame(X_state)
    X['bank_return'] = X_bank.values

    # Add constant
    X.insert(0, 'const', 1.0)

    # Convert to numpy arrays to ensure clean numeric data
    y_array = np.asarray(y, dtype=np.float64)
    X_array = np.asarray(X, dtype=np.float64)

    # Store column names for later
    col_names = X.columns.tolist()

    # Estimate quantile regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QuantReg(y_array, X_array)
        result = model.fit(q=quantile, max_iter=1000)

    # Create params dict with column names
    params_dict = dict(zip(col_names, result.params))

    # Diagnostics
    diagnostics = {
        'n_obs': len(y),
        'quantile': quantile,
        'beta_i': params_dict['bank_return'],
        'pseudo_r2': result.prsquared if hasattr(result, 'prsquared') else 0.0,
        'converged': True,
        'params': params_dict,
        'col_names': col_names
    }

    return result, diagnostics


def calculate_delta_covar(
    bank_ric: str,
    bank_return: pd.Series,
    state_variables: pd.DataFrame,
    system_return: pd.Series
) -> Dict:
    """Calculate ΔCoVaR for a single bank.

    This is the main function that orchestrates the two-step estimation.

    Args:
        bank_ric: Bank RIC identifier.
        bank_return: Bank return series.
        state_variables: Lagged state variables DataFrame.
        system_return: System return series (excluding this bank).

    Returns:
        Dictionary with CoVaR metrics:
            - bank_ric: Bank identifier
            - var_5pct: Bank's 5% VaR
            - var_50pct: Bank's 50% VaR (median)
            - covar_5pct: System's 5% CoVaR (when bank at 5% VaR)
            - covar_50pct: System's 50% CoVaR (when bank at 50% VaR)
            - beta_i: Coefficient of bank return in system equation
            - delta_covar: ΔCoVaR (marginal systemic risk)
            - n_obs: Number of observations
            - converged: Whether estimation converged

    Mathematical formulation:
        ΔCoVaR_i = CoVaR_{i,5%} - CoVaR_{i,50%}

        where:
        - CoVaR_{i,5%} = E[r_system | r_i = VaR_{i,5%}, X] at 5% quantile
        - CoVaR_{i,50%} = E[r_system | r_i = VaR_{i,50%}, X] at 5% quantile

    Example:
        >>> covar_metrics = calculate_delta_covar('ENBD.DU', bank_ret, state_vars, system_ret)
    """
    try:
        # Step 1: Estimate bank VaR at 5% and 50%
        var_5pct, diag_var_5 = estimate_bank_var(
            bank_return,
            state_variables,
            quantile=CONFIG.COVAR_VAR_QUANTILE
        )

        var_50pct, diag_var_50 = estimate_bank_var(
            bank_return,
            state_variables,
            quantile=CONFIG.COVAR_MEDIAN_QUANTILE
        )

        # Step 2: Estimate system CoVaR model at 5% quantile
        covar_model_5pct, diag_covar_5 = estimate_covar(
            system_return,
            state_variables,
            bank_return,
            quantile=CONFIG.COVAR_SYSTEM_QUANTILE
        )

        # Step 3: Estimate system CoVaR model at 50% quantile (for comparison)
        covar_model_50pct, diag_covar_50 = estimate_covar(
            system_return,
            state_variables,
            bank_return,
            quantile=CONFIG.COVAR_MEDIAN_QUANTILE
        )

        # Extract beta_i (coefficient of bank return) from 5% model
        beta_i = diag_covar_5['beta_i']

        # Align data for prediction
        common_index = system_return.index.intersection(
            state_variables.index
        ).intersection(bank_return.index)

        X_state = state_variables.loc[common_index]
        X_bank = bank_return.loc[common_index]

        # Ensure numeric and drop NaN
        X_state = X_state.apply(pd.to_numeric, errors='coerce').dropna()
        common_idx = X_state.index
        X_state = X_state.loc[common_idx]

        # Create prediction dataset
        X = pd.DataFrame(X_state)
        X.insert(0, 'const', 1.0)

        # Calculate CoVaR at 5%: system return when bank is at 5% VaR
        X_covar_5 = X.copy()
        X_covar_5['bank_return'] = var_5pct

        # Convert to numpy array matching model's column order
        X_covar_5_array = np.asarray(X_covar_5[diag_covar_5['col_names']], dtype=np.float64)
        pred_5 = covar_model_5pct.predict(X_covar_5_array)
        covar_5pct = float(np.quantile(pred_5, CONFIG.COVAR_SYSTEM_QUANTILE))

        # Calculate CoVaR at 50%: system return when bank is at 50% VaR
        X_covar_50 = X.copy()
        X_covar_50['bank_return'] = var_50pct

        X_covar_50_array = np.asarray(X_covar_50[diag_covar_5['col_names']], dtype=np.float64)
        pred_50 = covar_model_5pct.predict(X_covar_50_array)
        covar_50pct = float(np.quantile(pred_50, CONFIG.COVAR_SYSTEM_QUANTILE))

        # Calculate ΔCoVaR = CoVaR(5%) - CoVaR(50%)
        delta_covar = covar_5pct - covar_50pct

        # Compile results
        result = {
            'bank_ric': bank_ric,
            'var_5pct': var_5pct,
            'var_50pct': var_50pct,
            'covar_5pct': covar_5pct,
            'covar_50pct': covar_50pct,
            'beta_i': beta_i,
            'delta_covar': delta_covar,
            'n_obs': diag_covar_5['n_obs'],
            'pseudo_r2_covar': diag_covar_5['pseudo_r2'],
            'converged': (diag_var_5['converged'] and diag_var_50['converged'] and
                         diag_covar_5['converged'] and diag_covar_50['converged'])
        }

        return result

    except Exception as e:
        print(f"  ✗ {bank_ric}: {e}")
        return None


def estimate_covar_all_banks(
    bank_returns: pd.DataFrame,
    market_caps: pd.DataFrame,
    state_variables: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """Estimate CoVaR for all banks in the sample.

    Args:
        bank_returns: DataFrame with bank returns (date × bank_ric).
        market_caps: DataFrame with market capitalizations (date × bank_ric).
        state_variables: Lagged state variables DataFrame.
        verbose: Print progress messages.

    Returns:
        DataFrame with CoVaR metrics for all banks (one row per bank).

    Example:
        >>> covar_results = estimate_covar_all_banks(bank_returns, market_caps, state_vars)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("ESTIMATING CoVaR FOR ALL BANKS")
        print("=" * 80)

    results = []

    for i, bank_ric in enumerate(bank_returns.columns, 1):
        if verbose:
            print(f"\n[{i}/{len(bank_returns.columns)}] Processing {bank_ric}...")

        # Calculate system return (excluding this bank)
        system_return = calculate_system_return(
            bank_returns,
            market_caps,
            exclude_bank=bank_ric
        )

        # Get bank return
        bank_return = bank_returns[bank_ric]

        # Calculate ΔCoVaR
        result = calculate_delta_covar(
            bank_ric=bank_ric,
            bank_return=bank_return,
            state_variables=state_variables,
            system_return=system_return
        )

        if result is not None:
            results.append(result)
            if verbose:
                print(f"  ✓ ΔCoVaR = {result['delta_covar']:.4f}")

    # Create DataFrame
    if len(results) == 0:
        raise ValueError("No successful CoVaR estimations. Check data quality and state variables.")

    df = pd.DataFrame(results)

    # Verify delta_covar column exists
    if 'delta_covar' not in df.columns:
        raise ValueError(f"delta_covar column missing. Available columns: {df.columns.tolist()}")

    # Sort by ΔCoVaR (most negative = highest systemic risk)
    df = df.sort_values('delta_covar', ascending=True)

    if verbose:
        print("\n" + "=" * 80)
        print("CoVaR ESTIMATION COMPLETE")
        print("=" * 80)
        print(f"Total banks: {len(df)}")
        print(f"Convergence rate: {df['converged'].mean()*100:.1f}%")
        print(f"\nΔCoVaR statistics:")
        print(f"  Mean: {df['delta_covar'].mean():.4f}")
        print(f"  Std: {df['delta_covar'].std():.4f}")
        print(f"  Min: {df['delta_covar'].min():.4f} (highest systemic risk)")
        print(f"  Max: {df['delta_covar'].max():.4f}")

    return df
