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


def simulate_market_paths(
    market_garch_result,
    horizon: int = 22,
    n_simulations: int = 50000,
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Simulate market benchmark paths using GARCH dynamics (ONCE for all banks).

    This generates common market crisis scenarios that all banks will be evaluated against.
    This is critical for systemic risk measurement where all banks face the SAME market shock.

    Args:
        market_garch_result: GARCH estimation result for market benchmark.
        horizon: Forecast horizon in periods (default 22 weeks).
        n_simulations: Number of Monte Carlo paths.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - standardized_shocks: (n_simulations × horizon) array of z_m,t
            - cumulative_returns: (n_simulations,) array of cumulative market returns
            - conditional_volatilities: (n_simulations × horizon) array of σ_m,t
            - innovations: (n_simulations × horizon) array of ε_m,t

    Mathematical formulation:
        For each path n and step t:
            z_m,t ~ N(0, 1)
            ε_m,t = σ_m,t × z_m,t
            r_m,t = μ_m + ε_m,t
            σ²_m,t+1 = ω + α ε²_m,t + γ ε²_m,t I[ε_m,t<0] + β σ²_m,t

    Example:
        >>> market_paths = simulate_market_paths(benchmark_garch, horizon=22, n_sim=50000)
        >>> # Use same market_paths for all banks
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract GARCH parameters
    # NOTE: Parameters are estimated on returns*100 (percentage points)
    # But conditional_volatility is already scaled back to original scale
    params = market_garch_result["params"]
    mu = params.get("mu", 0.0) / 100  # Scale back to log return scale
    omega = params.get("omega") / (100**2)  # Scale back variance
    alpha = params.get("alpha[1]")  # Unitless, no scaling needed
    gamma = params.get("gamma[1]", 0.0)  # Unitless, no scaling needed
    beta = params.get("beta[1]")  # Unitless, no scaling needed

    # Initial volatility (already in log return scale)
    sigma_0 = market_garch_result["conditional_volatility"].iloc[-1]

    # Storage
    z_m = np.zeros((n_simulations, horizon))
    sigma_m = np.zeros((n_simulations, horizon))
    eps_m = np.zeros((n_simulations, horizon))
    returns_m = np.zeros((n_simulations, horizon))

    # Initialize volatility for all paths
    sigma_current = np.full(n_simulations, sigma_0)

    for t in range(horizon):
        # Generate standardized shocks (independent across simulations)
        z_m[:, t] = np.random.randn(n_simulations)

        # Current volatility
        sigma_m[:, t] = sigma_current

        # Innovation
        eps_m[:, t] = sigma_current * z_m[:, t]

        # Return
        returns_m[:, t] = mu + eps_m[:, t]

        # Update volatility for next period (GJR-GARCH)
        # σ²_{t+1} = ω + α ε²_t + γ ε²_t I[ε_t<0] + β σ²_t
        eps_sq = eps_m[:, t] ** 2
        leverage_term = gamma * eps_sq * (eps_m[:, t] < 0)

        sigma_sq_next = omega + alpha * eps_sq + leverage_term + beta * (sigma_current ** 2)
        sigma_current = np.sqrt(np.maximum(sigma_sq_next, 1e-8))  # Avoid negative variance

    # Cumulative returns
    cumulative_returns = returns_m.sum(axis=1)

    return {
        "standardized_shocks": z_m,
        "cumulative_returns": cumulative_returns,
        "conditional_volatilities": sigma_m,
        "innovations": eps_m,
        "returns": returns_m,
        "horizon": horizon,
        "n_simulations": n_simulations
    }


def simulate_bank_conditional_on_market(
    bank_garch_result,
    market_paths: Dict,
    dcc_result: Dict,
    random_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Simulate bank returns conditional on FIXED market paths.

    This implements the key insight: given the same market crisis scenarios,
    generate bank-specific returns using the bivariate correlation structure.

    Mathematical formulation (conditional distribution):
        Given z_m,t (from market_paths), generate z_i,t:
            z_i,t | z_m,t ~ N(ρ_t × z_m,t, 1 - ρ²_t)

        Where ρ_t is time-varying correlation from DCC.

    Args:
        bank_garch_result: GARCH estimation result for bank.
        market_paths: Pre-simulated market paths from simulate_market_paths().
        dcc_result: DCC estimation result (bivariate: bank-market).
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - cumulative_returns: (n_simulations,) array of bank cumulative returns
            - standardized_shocks: (n_simulations × horizon) array of z_i,t
            - conditional_volatilities: (n_simulations × horizon) array of σ_i,t

    Example:
        >>> # First, simulate market once
        >>> market_paths = simulate_market_paths(benchmark_garch, n_sim=50000)
        >>> # Then simulate each bank conditional on those paths
        >>> bank_A_paths = simulate_bank_conditional_on_market(bank_A_garch, market_paths, dcc_A)
        >>> bank_B_paths = simulate_bank_conditional_on_market(bank_B_garch, market_paths, dcc_B)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_simulations = market_paths["n_simulations"]
    horizon = market_paths["horizon"]

    # Extract bank GARCH parameters
    # NOTE: Parameters are estimated on returns*100 (percentage points)
    # But conditional_volatility is already scaled back to original scale
    params = bank_garch_result["params"]
    mu_i = params.get("mu", 0.0) / 100  # Scale back to log return scale
    omega_i = params.get("omega") / (100**2)  # Scale back variance
    alpha_i = params.get("alpha[1]")  # Unitless, no scaling needed
    gamma_i = params.get("gamma[1]", 0.0)  # Unitless, no scaling needed
    beta_i = params.get("beta[1]")  # Unitless, no scaling needed

    # Initial volatility (already in log return scale)
    sigma_i_0 = bank_garch_result["conditional_volatility"].iloc[-1]

    # Extract DCC parameters
    dcc_a, dcc_b = dcc_result["params"]
    Q_bar = dcc_result["Q_bar"]

    # Initialize Q for DCC
    Q_current = Q_bar.copy()

    # Extract market shocks (these are FIXED)
    z_m = market_paths["standardized_shocks"]

    # Storage for bank
    z_i = np.zeros((n_simulations, horizon))
    sigma_i = np.zeros((n_simulations, horizon))
    returns_i = np.zeros((n_simulations, horizon))

    # Initialize bank volatility
    sigma_i_current = np.full(n_simulations, sigma_i_0)

    for t in range(horizon):
        # Compute current correlation from DCC
        # Q_t has structure: [[q11, q12], [q21, q22]]
        q11 = Q_current[0, 0]
        q12 = Q_current[0, 1]
        q22 = Q_current[1, 1]

        # ρ_t = q12 / sqrt(q11 × q22)
        rho_t = q12 / np.sqrt(q11 * q22)
        rho_t = np.clip(rho_t, -0.999, 0.999)  # Ensure valid correlation

        # Generate bank shocks CONDITIONAL on market shocks
        # z_i,t | z_m,t ~ N(ρ_t × z_m,t, 1 - ρ²_t)
        conditional_mean = rho_t * z_m[:, t]
        conditional_std = np.sqrt(1 - rho_t ** 2)

        # Draw from conditional distribution
        z_i[:, t] = conditional_mean + conditional_std * np.random.randn(n_simulations)

        # Current bank volatility
        sigma_i[:, t] = sigma_i_current

        # Bank innovation
        eps_i_t = sigma_i_current * z_i[:, t]

        # Bank return
        returns_i[:, t] = mu_i + eps_i_t

        # Update bank volatility for next period
        eps_i_sq = eps_i_t ** 2
        leverage_term_i = gamma_i * eps_i_sq * (eps_i_t < 0)

        sigma_i_sq_next = omega_i + alpha_i * eps_i_sq + leverage_term_i + beta_i * (sigma_i_current ** 2)
        sigma_i_current = np.sqrt(np.maximum(sigma_i_sq_next, 1e-8))

        # Update DCC Q_t
        # Use average z vector for updating (simplified)
        # In practice, could update per-path, but this is computationally expensive
        z_avg_i = z_i[:, t].mean()
        z_avg_m = z_m[:, t].mean()
        z_vec = np.array([[z_avg_i], [z_avg_m]])

        Q_current = (1 - dcc_a - dcc_b) * Q_bar + dcc_a * (z_vec @ z_vec.T) + dcc_b * Q_current

    # Cumulative returns
    cumulative_returns = returns_i.sum(axis=1)

    return {
        "cumulative_returns": cumulative_returns,
        "standardized_shocks": z_i,
        "conditional_volatilities": sigma_i,
        "returns": returns_i
    }


def calculate_lrmes_with_common_market(
    bank_cumulative_returns: np.ndarray,
    market_cumulative_returns: np.ndarray,
    crisis_threshold: float = -0.40,
    min_crisis_scenarios: int = 500
) -> Dict[str, float]:
    """Calculate LRMES given bank returns and market returns (common paths).

    Args:
        bank_cumulative_returns: (n_simulations,) array of bank cumulative returns.
        market_cumulative_returns: (n_simulations,) array of market cumulative returns.
        crisis_threshold: Market decline threshold as percentage (e.g., -0.40 for -40%).
        min_crisis_scenarios: Minimum crisis scenarios required.

    Returns:
        Dictionary with LRMES and diagnostics.

    Note:
        Crisis threshold is interpreted as percentage decline. For example:
        - crisis_threshold = -0.40 means -40% price decline
        - Converted to log return: ln(1 - 0.40) = ln(0.60) ≈ -0.5108

    Example:
        >>> lrmes_dict = calculate_lrmes_with_common_market(
        ...     bank_paths["cumulative_returns"],
        ...     market_paths["cumulative_returns"],
        ...     crisis_threshold=-0.40
        ... )
        >>> print(f"LRMES: {lrmes_dict['lrmes']:.4f}")
    """
    n_simulations = len(market_cumulative_returns)

    # Convert percentage threshold to log return threshold
    # crisis_threshold = -0.40 (40% decline) → ln(1 - 0.40) = ln(0.60) ≈ -0.5108
    crisis_threshold_log = np.log(1 + crisis_threshold)

    # Identify crisis scenarios where market declined more than threshold
    crisis_mask = market_cumulative_returns < crisis_threshold_log

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

    # Bank returns in crisis scenarios
    crisis_returns = bank_cumulative_returns[crisis_mask]

    # LRMES = 1 - E[exp(r_i) | crisis]
    # exp(r_i) = W_i,t+h / W_i,t (equity value ratio)
    equity_value_ratio = np.exp(crisis_returns)
    expected_equity_loss = 1 - equity_value_ratio.mean()

    lrmes = max(0, expected_equity_loss)  # LRMES >= 0

    # Market LRMES (for reference)
    market_crisis_returns = market_cumulative_returns[crisis_mask]
    market_equity_ratio = np.exp(market_crisis_returns)
    market_lrmes = 1 - market_equity_ratio.mean()

    return {
        "lrmes": lrmes,
        "market_lrmes": market_lrmes,
        "n_crisis_scenarios": int(n_crisis),
        "crisis_probability": float(n_crisis / n_simulations),
        "avg_bank_loss_in_crisis": -crisis_returns.mean(),
        "avg_market_loss_in_crisis": -market_crisis_returns.mean()
    }


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
