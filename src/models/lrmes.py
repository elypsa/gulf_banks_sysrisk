"""Long-Run Marginal Expected Shortfall (LRMES) estimation via Monte Carlo.

LRMES measures the expected equity loss of a bank conditional on a severe
market decline over a specified horizon. It is a key input to SRISK calculation.

Mathematical formulation:
    LRMES_i,t = E_t[1 - W_i,t+h/W_i,t | R_m,t:t+h < C]

Where:
    - W_i,t = Market equity value at time t
    - R_m,t:t+h = Cumulative market return over horizon h
    - C = Crisis threshold (typically -40% over 6 months)
    - h = Forecast horizon (typically 22 weeks ≈ 6 months)

Estimation procedure:
    1. Estimate GARCH-DCC → get volatility and correlation dynamics
    2. Simulate N paths of returns over horizon h using GARCH-DCC
    3. Identify crisis scenarios: paths where R_m,t:t+h < C
    4. Compute average equity loss in crisis scenarios

References:
    Brownlees, C. T., & Engle, R. F. (2017). "SRISK: A Conditional Capital
    Shortfall Measure of Systemic Risk." Review of Financial Studies, 30(1), 48-79.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


def simulate_garch_dcc_returns(
    garch_params: Dict,
    dcc_result: Dict,
    instrument_names: list,
    horizon: int = 22,
    n_simulations: int = 50000,
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Simulate returns using GARCH-DCC model.

    Args:
        garch_params: Dictionary of GARCH parameters per instrument.
        dcc_result: DCC estimation result dictionary.
        instrument_names: List of instrument names (first = market benchmark).
        horizon: Forecast horizon in periods (default 22 weeks).
        n_simulations: Number of Monte Carlo paths.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - simulated_returns: (n_simulations × horizon × n_instruments) array
            - cumulative_returns: (n_simulations × n_instruments) array
            - market_idx: Index of market benchmark (0)

    Example:
        >>> simulations = simulate_garch_dcc_returns(
        ...     garch_params, dcc_result, names, horizon=22, n_simulations=50000
        ... )
        >>> cum_returns = simulations['cumulative_returns']
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_instruments = len(instrument_names)

    # Extract DCC parameters
    dcc_a, dcc_b = dcc_result["params"]
    Q_bar = dcc_result["Q_bar"]

    # Initialize Q_0 = Q̄
    Q_current = Q_bar.copy()

    # Storage for simulations
    simulated_returns = np.zeros((n_simulations, horizon, n_instruments))

    # Last conditional volatilities from GARCH (use last observed)
    last_vols = []
    for name in instrument_names:
        if name in garch_params:
            last_vol = garch_params[name]["conditional_volatility"].iloc[-1]
            last_vols.append(last_vol)
        else:
            raise ValueError(f"No GARCH parameters for {name}")

    volatilities = np.array(last_vols)

    # Simulate each period
    for h in range(horizon):
        # Generate standardized shocks from multivariate normal with correlation R_t
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q_current)))
        R_current = D_inv @ Q_current @ D_inv

        # Ensure valid correlation matrix
        R_current = make_correlation_matrix(R_current)

        # Draw standardized innovations
        try:
            z_t = np.random.multivariate_normal(
                mean=np.zeros(n_instruments),
                cov=R_current,
                size=n_simulations
            )
        except np.linalg.LinAlgError:
            warnings.warn("Correlation matrix not positive definite, using identity")
            z_t = np.random.randn(n_simulations, n_instruments)

        # Convert to returns: r_t = σ_t × z_t
        returns_t = z_t * volatilities

        simulated_returns[:, h, :] = returns_t

        # Update volatilities using GARCH persistence
        # Simplified: σ²_{t+1} ≈ ω + (α+γ/2+β) σ²_t (assume stationarity)
        # For simplicity, hold volatilities constant (conservative)
        # In production, could simulate full GARCH dynamics

        # Update DCC Q_t
        # Use average shock from simulations
        z_t_mean = z_t.mean(axis=0).reshape(-1, 1)
        Q_current = (1 - dcc_a - dcc_b) * Q_bar + dcc_a * (z_t_mean @ z_t_mean.T) + dcc_b * Q_current

    # Compute cumulative returns: sum of log returns over horizon
    cumulative_returns = simulated_returns.sum(axis=1)

    return {
        "simulated_returns": simulated_returns,
        "cumulative_returns": cumulative_returns,
        "market_idx": 0,  # Assume first instrument is market
        "instrument_names": instrument_names,
        "horizon": horizon,
        "n_simulations": n_simulations
    }


def calculate_lrmes(
    simulations: Dict,
    crisis_threshold: float = -0.40,
    market_idx: int = 0,
    min_crisis_scenarios: int = 500
) -> Dict[str, float]:
    """Calculate LRMES for each instrument given simulations.

    Args:
        simulations: Output from simulate_garch_dcc_returns.
        crisis_threshold: Market decline threshold (default -40%).
        market_idx: Index of market benchmark in simulations.
        min_crisis_scenarios: Minimum crisis scenarios required.

    Returns:
        Dictionary mapping instrument name to LRMES value.

    Note:
        LRMES = Expected equity loss in crisis = 1 - E[exp(r_i) | R_m < C]

    Example:
        >>> lrmes_values = calculate_lrmes(simulations, crisis_threshold=-0.40)
        >>> print(f"ENBD.DU LRMES: {lrmes_values['ENBD.DU']:.4f}")
    """
    cum_returns = simulations["cumulative_returns"]
    instrument_names = simulations["instrument_names"]
    n_simulations = simulations["n_simulations"]

    # Identify crisis scenarios: market return < threshold
    market_returns = cum_returns[:, market_idx]
    crisis_mask = market_returns < crisis_threshold

    n_crisis = crisis_mask.sum()

    if n_crisis < min_crisis_scenarios:
        warnings.warn(
            f"Only {n_crisis} crisis scenarios out of {n_simulations} "
            f"({n_crisis/n_simulations*100:.2f}%). Consider increasing simulations "
            f"or relaxing threshold."
        )

    if n_crisis == 0:
        raise ValueError(
            f"No crisis scenarios found with threshold {crisis_threshold}. "
            "Check if threshold is too extreme or volatility too low."
        )

    # Calculate LRMES for each instrument
    lrmes = {}

    for i, name in enumerate(instrument_names):
        if i == market_idx:
            # Market itself
            crisis_returns = cum_returns[crisis_mask, i]
            expected_value_loss = 1 - np.exp(crisis_returns).mean()
            lrmes[name] = expected_value_loss
        else:
            # Bank: equity loss conditional on market crisis
            crisis_returns = cum_returns[crisis_mask, i]

            # LRMES = 1 - E[exp(r_i) | crisis]
            # exp(r_i) = W_i,t+h / W_i,t (equity value ratio)
            equity_value_ratio = np.exp(crisis_returns)
            expected_equity_loss = 1 - equity_value_ratio.mean()

            lrmes[name] = max(0, expected_equity_loss)  # LRMES >= 0

    # Add diagnostics
    lrmes["_n_crisis_scenarios"] = n_crisis
    lrmes["_crisis_probability"] = n_crisis / n_simulations

    return lrmes


def estimate_lrmes_rolling(
    returns_df: pd.DataFrame,
    garch_results: Dict[str, Dict],
    dcc_result: Dict,
    horizon: int = 22,
    crisis_threshold: float = -0.40,
    n_simulations: int = 50000,
    window_size: int = 1000,
    step_size: int = 22,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """Estimate LRMES using rolling window.

    Args:
        returns_df: DataFrame with returns (columns = instruments).
        garch_results: GARCH estimation results.
        dcc_result: DCC estimation result.
        horizon: Forecast horizon (weeks).
        crisis_threshold: Market decline threshold.
        n_simulations: Number of Monte Carlo paths.
        window_size: Rolling window size for re-estimation.
        step_size: Step size for rolling window.
        random_seed: Random seed.

    Returns:
        DataFrame with LRMES time series (index = dates, columns = instruments).

    Note:
        For production SRISK, typically estimate LRMES daily or weekly
        using expanding or rolling window. This function provides a template.

    Example:
        >>> lrmes_series = estimate_lrmes_rolling(
        ...     returns_df, garch_results, dcc_result, window_size=1000
        ... )
    """
    # For simplicity, compute single LRMES using full sample
    # In production, implement proper rolling estimation

    instrument_names = list(returns_df.columns)

    # Extract GARCH parameters
    garch_params = {}
    for name in instrument_names:
        if name in garch_results:
            garch_params[name] = garch_results[name]

    # Simulate
    simulations = simulate_garch_dcc_returns(
        garch_params=garch_params,
        dcc_result=dcc_result,
        instrument_names=instrument_names,
        horizon=horizon,
        n_simulations=n_simulations,
        random_seed=random_seed
    )

    # Calculate LRMES
    lrmes_values = calculate_lrmes(
        simulations=simulations,
        crisis_threshold=crisis_threshold,
        market_idx=0
    )

    # Create DataFrame with constant LRMES (simplification)
    lrmes_df = pd.DataFrame(
        {name: [lrmes_values[name]] * len(returns_df)
         for name in instrument_names if name in lrmes_values},
        index=returns_df.index
    )

    return lrmes_df


def lrmes_diagnostics(
    lrmes_values: Dict[str, float],
    simulations: Dict
) -> pd.DataFrame:
    """Generate diagnostic summary for LRMES estimation.

    Args:
        lrmes_values: LRMES estimates from calculate_lrmes.
        simulations: Simulation dictionary.

    Returns:
        DataFrame with diagnostics.
    """
    diagnostics = {
        "n_simulations": simulations["n_simulations"],
        "horizon_weeks": simulations["horizon"],
        "n_crisis_scenarios": lrmes_values.get("_n_crisis_scenarios", 0),
        "crisis_probability": lrmes_values.get("_crisis_probability", 0),
        "avg_lrmes": np.mean([v for k, v in lrmes_values.items() if not k.startswith("_")]),
        "max_lrmes": max([v for k, v in lrmes_values.items() if not k.startswith("_")], default=0),
        "min_lrmes": min([v for k, v in lrmes_values.items() if not k.startswith("_")], default=0)
    }

    return pd.DataFrame([diagnostics])


def make_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensure matrix is a valid correlation matrix."""
    # Ensure symmetric
    matrix = (matrix + matrix.T) / 2

    # Ensure diagonal = 1
    np.fill_diagonal(matrix, 1.0)

    # Clip off-diagonal
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            matrix[i, j] = np.clip(matrix[i, j], -0.999, 0.999)
            matrix[j, i] = matrix[i, j]

    # Check positive definite
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # Use nearest correlation matrix
        matrix = nearest_correlation_matrix(matrix)

    return matrix


def nearest_correlation_matrix(A: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Find nearest correlation matrix using Higham's algorithm.

    Args:
        A: Candidate correlation matrix.
        max_iter: Maximum iterations.

    Returns:
        Nearest valid correlation matrix.
    """
    n = A.shape[0]
    Y = A.copy()
    X = A.copy()

    for _ in range(max_iter):
        # Project to S (positive semidefinite)
        eigval, eigvec = np.linalg.eigh(Y)
        eigval = np.maximum(eigval, 0)
        X = eigvec @ np.diag(eigval) @ eigvec.T

        # Project to U (unit diagonal)
        np.fill_diagonal(X, 1.0)

        # Check convergence
        if np.allclose(X, Y):
            break

        Y = X

    return X


def sensitivity_analysis_lrmes(
    garch_params: Dict,
    dcc_result: Dict,
    instrument_names: list,
    thresholds: list = [-0.30, -0.35, -0.40, -0.45, -0.50],
    horizons: list = [22, 13, 4],
    n_simulations: int = 50000
) -> pd.DataFrame:
    """Perform sensitivity analysis on LRMES for different thresholds and horizons.

    Args:
        garch_params: GARCH parameters.
        dcc_result: DCC result.
        instrument_names: List of instruments.
        thresholds: List of crisis thresholds to test.
        horizons: List of horizons (in weeks) to test.
        n_simulations: Number of simulations.

    Returns:
        DataFrame with LRMES under different scenarios.

    Example:
        >>> sensitivity = sensitivity_analysis_lrmes(
        ...     garch_params, dcc_result, names,
        ...     thresholds=[-0.30, -0.40, -0.50]
        ... )
    """
    results = []

    for horizon in horizons:
        # Simulate once per horizon
        simulations = simulate_garch_dcc_returns(
            garch_params=garch_params,
            dcc_result=dcc_result,
            instrument_names=instrument_names,
            horizon=horizon,
            n_simulations=n_simulations
        )

        for threshold in thresholds:
            try:
                lrmes_values = calculate_lrmes(
                    simulations=simulations,
                    crisis_threshold=threshold,
                    min_crisis_scenarios=100
                )

                for instrument in instrument_names:
                    if instrument in lrmes_values and not instrument.startswith("_"):
                        results.append({
                            "instrument": instrument,
                            "horizon_weeks": horizon,
                            "crisis_threshold": threshold,
                            "lrmes": lrmes_values[instrument],
                            "n_crisis": lrmes_values["_n_crisis_scenarios"]
                        })

            except Exception as e:
                warnings.warn(f"Failed for threshold={threshold}, horizon={horizon}: {e}")

    df = pd.DataFrame(results)
    return df
