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
import logging
from scipy.stats import norm

# Configure logger for LRMES module
logger = logging.getLogger(__name__)


def compute_importance_weight(z_market: np.ndarray, mu_tilt: float) -> np.ndarray:
    """Compute importance sampling weight for tilted distribution.

    Under importance sampling, we sample market shocks from N(mu_tilt, 1) instead
    of N(0, 1) to increase the probability of crisis scenarios. This function
    computes the importance weight to correct for this bias.

    Mathematical formula:
        w = φ(z; 0, 1) / φ(z; μ_tilt, 1)
          = exp(-z²/2) / exp(-(z - μ_tilt)²/2)
          = exp(μ_tilt·z - μ_tilt²/2)

    Args:
        z_market: Standardized market shock(s). Can be scalar or array.
        mu_tilt: Mean shift for importance sampling (< 0 to tilt toward crises).

    Returns:
        Importance weight(s) with same shape as z_market.

    Note:
        - For standard sampling (mu_tilt=0), weight = 1.0 (no correction)
        - For mu_tilt < 0, negative shocks get higher weight (crisis-tilted)
        - Total weight across all samples should average to ~1.0

    Example:
        >>> z = np.array([-2.0, -1.0, 0.0, 1.0])
        >>> w = compute_importance_weight(z, mu_tilt=-0.5)
        >>> # z=-2.0 gets higher weight (more likely under standard sampling)
    """
    # Closed-form expression: exp(mu_tilt * z - mu_tilt^2 / 2)
    # This avoids numerical issues with density ratio computation
    log_weight = mu_tilt * z_market - (mu_tilt ** 2) / 2
    return np.exp(log_weight)


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
            logger.warning("Correlation matrix not positive definite, using identity matrix")
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
        logger.warning(
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


def simulate_bivariate_garch_dcc(
    bank_garch_result,
    market_garch_result,
    dcc_result: Dict,
    horizon: int = 22,
    n_simulations: int = 50000,
    random_seed: Optional[int] = None,
    use_importance_sampling: bool = False,
    mu_tilt: float = 0.0
) -> Dict[str, np.ndarray]:
    """Simulate bivariate GARCH-DCC paths for bank and market jointly.

    This is the CORRECT implementation following Section 5.2 of srisks.md.
    Both bank and market returns are simulated together with time-varying
    volatilities and correlations updated along each path.

    Algorithm (for each simulation path n):
        Initialize:
            σ_i,t^(n) = σ_i,t, σ_m,t^(n) = σ_m,t (last observed)
            Q_t^(n) = Q_t (last observed DCC state)
            r_i,cum^(n) = 0, r_m,cum^(n) = 0

        For each step s = 1, ..., h:
            Step 1: Generate correlated shocks
                z_t = [z_i,t, z_m,t]^T ~ N(0, R_t)  where R_t from Q_t

            Step 2: Compute returns
                r_i,t = μ_i + σ_i,t × z_i,t
                r_m,t = μ_m + σ_m,t × z_m,t

            Step 3: Accumulate
                r_i,cum += r_i,t
                r_m,cum += r_m,t

            Step 4: Update volatilities (GJR-GARCH)
                ε_i,t = σ_i,t × z_i,t
                σ²_i,t+1 = ω_i + α_i ε²_i,t + γ_i ε²_i,t I[ε_i,t<0] + β_i σ²_i,t
                (same for market)

            Step 5: Update correlation (DCC)
                Q_t+1 = (1-a-b)Q̄ + a(z_t z_t^T) + b Q_t
                R_t+1 = diag(Q_t+1)^{-1/2} Q_t+1 diag(Q_t+1)^{-1/2}

    Args:
        bank_garch_result: GARCH estimation result for bank.
        market_garch_result: GARCH estimation result for market benchmark.
        dcc_result: DCC estimation result (bivariate: bank-market).
        horizon: Forecast horizon in periods (default 22 weeks).
        n_simulations: Number of Monte Carlo paths.
        random_seed: Random seed for reproducibility.
        use_importance_sampling: If True, tilt market shocks toward crises (default False).
        mu_tilt: Mean shift for market shocks under importance sampling (< 0 for crisis-tilting).

    Returns:
        Dictionary containing:
            - bank_cumulative_returns: (n_simulations,) array
            - market_cumulative_returns: (n_simulations,) array
            - bank_returns: (n_simulations × horizon) array
            - market_returns: (n_simulations × horizon) array
            - bank_volatilities: (n_simulations × horizon) array
            - market_volatilities: (n_simulations × horizon) array
            - correlations: (n_simulations × horizon) array of ρ_t
            - importance_weights: (n_simulations,) array of importance weights (1.0 if not using IS)

    Example:
        >>> result = simulate_bivariate_garch_dcc(
        ...     bank_garch, market_garch, dcc_result, horizon=22, n_sim=50000
        ... )
        >>> bank_lrmes = calculate_lrmes_from_bivariate(result)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract bank GARCH parameters
    params_i = bank_garch_result["params"]
    mu_i = params_i.get("mu", 0.0) 
    omega_i = params_i.get("omega") 
    alpha_i = params_i.get("alpha[1]")
    gamma_i = params_i.get("gamma[1]", 0.0)
    beta_i = params_i.get("beta[1]")
    sigma_i_0 = bank_garch_result["conditional_volatility"].iloc[-1]

    # Extract market GARCH parameters
    params_m = market_garch_result["params"]
    mu_m = params_m.get("mu", 0.0) 
    omega_m = params_m.get("omega") 
    alpha_m = params_m.get("alpha[1]")
    gamma_m = params_m.get("gamma[1]", 0.0)
    beta_m = params_m.get("beta[1]")
    sigma_m_0 = market_garch_result["conditional_volatility"].iloc[-1]

    # Extract DCC parameters
    dcc_a, dcc_b = dcc_result["params"]
    Q_bar = dcc_result["Q_bar"]

    # Storage arrays
    bank_returns = np.zeros((n_simulations, horizon))
    market_returns = np.zeros((n_simulations, horizon))
    bank_vols = np.zeros((n_simulations, horizon))
    market_vols = np.zeros((n_simulations, horizon))
    correlations = np.zeros((n_simulations, horizon))
    market_shocks = np.zeros((n_simulations, horizon))  # For importance weights

    # Simulate each path independently
    for n in range(n_simulations):
        # Initialize state for this path (Step: Initialize)
        sigma_i = sigma_i_0
        sigma_m = sigma_m_0
        Q_current = Q_bar.copy()

        # Simulate each time step
        for t in range(horizon):
            # Step 1: Generate correlated standardized shocks
            # Compute R_t from Q_t
            q11 = Q_current[0, 0]
            q12 = Q_current[0, 1]
            q22 = Q_current[1, 1]

            # ρ_t = q12 / sqrt(q11 × q22)
            rho_t = q12 / np.sqrt(q11 * q22)
            rho_t = np.clip(rho_t, -0.999, 0.999)

            # Store correlation
            correlations[n, t] = rho_t

            # Generate correlated normal shocks using Cholesky
            # z = [z_i, z_m]^T ~ N(0, R_t)
            # R_t = [[1, ρ_t], [ρ_t, 1]]
            # Cholesky: L = [[1, 0], [ρ_t, sqrt(1-ρ²_t)]]
            u1 = np.random.randn()
            u2 = np.random.randn()
            z_i = u1
            z_m = rho_t * u1 + np.sqrt(1 - rho_t**2) * u2

            # Apply importance sampling tilt to market shock if enabled
            # Sample from N(mu_tilt, 1) instead of N(0, 1) for crisis scenarios
            if use_importance_sampling:
                z_m = z_m + mu_tilt

            # Store raw market shock for importance weight calculation
            market_shocks[n, t] = z_m

            # Step 2: Compute returns
            r_i = mu_i + sigma_i * z_i
            r_m = mu_m + sigma_m * z_m

            # Store returns and volatilities
            bank_returns[n, t] = r_i
            market_returns[n, t] = r_m
            bank_vols[n, t] = sigma_i
            market_vols[n, t] = sigma_m

            # Step 4: Update volatilities (GJR-GARCH)
            eps_i = sigma_i * z_i
            eps_m = sigma_m * z_m

            # Bank volatility update
            sigma_i_sq = (omega_i +
                         alpha_i * eps_i**2 +
                         gamma_i * eps_i**2 * (eps_i < 0) +
                         beta_i * sigma_i**2)
            sigma_i = np.sqrt(max(sigma_i_sq, 1e-8))

            # Market volatility update
            sigma_m_sq = (omega_m +
                         alpha_m * eps_m**2 +
                         gamma_m * eps_m**2 * (eps_m < 0) +
                         beta_m * sigma_m**2)
            sigma_m = np.sqrt(max(sigma_m_sq, 1e-8))

            # Step 5: Update correlation (DCC)
            z_vec = np.array([[z_i], [z_m]])
            Q_current = ((1 - dcc_a - dcc_b) * Q_bar +
                        dcc_a * (z_vec @ z_vec.T) +
                        dcc_b * Q_current)

    # Step 3: Compute cumulative returns (sum of log returns)
    bank_cumulative = bank_returns.sum(axis=1)
    market_cumulative = market_returns.sum(axis=1)

    # Compute importance weights if using importance sampling
    if use_importance_sampling:
        # Importance weight for each simulation is product of weights across time
        # w^(n) = ∏_{t=1}^h φ(z_m,t; 0, 1) / φ(z_m,t; μ_tilt, 1)
        # In log space: log w^(n) = ∑_{t=1}^h [μ_tilt·z_m,t - μ_tilt²/2]
        log_weights = np.sum(
            mu_tilt * market_shocks - (mu_tilt ** 2) / 2,
            axis=1
        )
        importance_weights = np.exp(log_weights)
    else:
        # No importance sampling: uniform weights
        importance_weights = np.ones(n_simulations)

    return {
        "bank_cumulative_returns": bank_cumulative,
        "market_cumulative_returns": market_cumulative,
        "bank_returns": bank_returns,
        "market_returns": market_returns,
        "bank_volatilities": bank_vols,
        "market_volatilities": market_vols,
        "correlations": correlations,
        "importance_weights": importance_weights,
        "horizon": horizon,
        "n_simulations": n_simulations,
        "use_importance_sampling": use_importance_sampling,
        "mu_tilt": mu_tilt if use_importance_sampling else 0.0
    }


def calculate_lrmes_from_bivariate(
    simulation_result: Dict,
    crisis_threshold: float = -0.40,
    min_crisis_scenarios: int = 500
) -> Dict[str, float]:
    """Calculate LRMES from bivariate GARCH-DCC simulation results.

    Supports both standard Monte Carlo and importance sampling estimators.
    If importance sampling was used in simulation, applies proper weighting
    to correct for the tilted distribution.

    Args:
        simulation_result: Output from simulate_bivariate_garch_dcc().
        crisis_threshold: Market decline threshold (e.g., -0.40 for -40%).
        min_crisis_scenarios: Minimum crisis scenarios required.

    Returns:
        Dictionary with LRMES and diagnostics:
            - lrmes: Long-run marginal expected shortfall
            - n_crisis_scenarios: Number of crisis paths
            - crisis_probability: Fraction of paths in crisis
            - avg_bank_loss_in_crisis: Average bank loss | crisis
            - avg_market_loss_in_crisis: Average market loss | crisis
            - avg_correlation: Average correlation over horizon
            - use_importance_sampling: Whether importance sampling was used
            - effective_sample_size: ESS (only if IS used)
            - efficiency_ratio: ESS/N_crisis, closer to 1.0 is better (only if IS used)
            - mu_tilt: Tilt parameter (only if IS used)
            - avg/max/min_importance_weight: Weight statistics (only if IS used)

    Mathematical formulas:
        Standard MC:
            LRMES = E[1 - exp(r_i,t:t+h) | r_m,t:t+h < ln(1 + C)]

        Importance Sampling:
            LRMES^IS = ∑_{n∈C} w^(n) · (1 - exp(r_i^(n))) / ∑_{n∈C} w^(n)
            where w^(n) are importance weights and C is crisis scenarios

    Example:
        >>> # Standard sampling
        >>> sim = simulate_bivariate_garch_dcc(bank, market, dcc, n_sim=50000)
        >>> lrmes_dict = calculate_lrmes_from_bivariate(sim, crisis_threshold=-0.40)
        >>> print(f"LRMES: {lrmes_dict['lrmes']:.4f}")
        >>>
        >>> # With importance sampling for variance reduction
        >>> sim_is = simulate_bivariate_garch_dcc(
        ...     bank, market, dcc, n_sim=50000,
        ...     use_importance_sampling=True, mu_tilt=-0.5
        ... )
        >>> lrmes_dict_is = calculate_lrmes_from_bivariate(sim_is, crisis_threshold=-0.40)
        >>> print(f"LRMES: {lrmes_dict_is['lrmes']:.4f}")
        >>> print(f"ESS: {lrmes_dict_is['effective_sample_size']:.0f}")
    """
    bank_cumulative = simulation_result["bank_cumulative_returns"]
    market_cumulative = simulation_result["market_cumulative_returns"]
    correlations = simulation_result["correlations"]
    n_simulations = simulation_result["n_simulations"]

    # Convert percentage threshold to log return in percentage points
    # crisis_threshold is in decimal (e.g., -0.40 for -40%)
    # Returns are in percentage points, so multiply by 100
    crisis_threshold_log = np.log(1 + crisis_threshold) * 100

    # Identify crisis scenarios
    crisis_mask = market_cumulative < crisis_threshold_log
    n_crisis = crisis_mask.sum()

    if n_crisis < min_crisis_scenarios:
        logger.warning(
            f"Only {n_crisis} crisis scenarios out of {n_simulations} "
            f"({n_crisis/n_simulations*100:.2f}%). Consider increasing simulations."
        )

    if n_crisis == 0:
        raise ValueError(
            f"No crisis scenarios with threshold {crisis_threshold}. "
            "Threshold may be too extreme."
        )

    # Calculate LRMES with importance sampling correction if applicable
    bank_crisis_returns = bank_cumulative[crisis_mask]
    market_crisis_returns = market_cumulative[crisis_mask]

    # Check if importance sampling was used
    use_importance_sampling = simulation_result.get("use_importance_sampling", False)

    if use_importance_sampling:
        # Importance sampling estimator:
        # LRMES^IS = ∑_{n∈C} w^(n) · (1 - exp(r_i^(n))) / ∑_{n∈C} w^(n)
        importance_weights = simulation_result["importance_weights"]
        crisis_weights = importance_weights[crisis_mask]

        # Returns are in percentage points, so divide by 100 before exp
        equity_value_ratio = np.exp(bank_crisis_returns / 100)
        losses = 1 - equity_value_ratio

        # Weighted average
        total_weight = crisis_weights.sum()
        if total_weight > 0:
            lrmes = (crisis_weights * losses).sum() / total_weight
        else:
            raise ValueError("Total importance weight is zero in crisis scenarios")

        # Effective sample size (ESS) diagnostic
        ess = (crisis_weights.sum() ** 2) / (crisis_weights ** 2).sum()
    else:
        # Standard Monte Carlo estimator (uniform weights)
        # LRMES = 1 - E[exp(r_i) | crisis]
        equity_value_ratio = np.exp(bank_crisis_returns / 100)
        lrmes = 1 - equity_value_ratio.mean()
        ess = n_crisis  # ESS = N for uniform weights

    lrmes = max(0, lrmes)  # LRMES >= 0

    # Diagnostics
    result = {
        "lrmes": lrmes,
        "n_crisis_scenarios": int(n_crisis),
        "crisis_probability": float(n_crisis / n_simulations),
        "avg_bank_loss_in_crisis": float(-bank_crisis_returns.mean()),
        "avg_market_loss_in_crisis": float(-market_crisis_returns.mean()),
        "avg_correlation": float(correlations.mean()),
        "use_importance_sampling": use_importance_sampling
    }

    # Add importance sampling diagnostics
    if use_importance_sampling:
        result["effective_sample_size"] = float(ess)
        result["efficiency_ratio"] = float(ess / n_crisis)  # ESS/N_crisis (1.0 is ideal)
        result["mu_tilt"] = simulation_result.get("mu_tilt", 0.0)

        # Weight diagnostics
        crisis_weights = simulation_result["importance_weights"][crisis_mask]
        result["avg_importance_weight"] = float(crisis_weights.mean())
        result["max_importance_weight"] = float(crisis_weights.max())
        result["min_importance_weight"] = float(crisis_weights.min())

    return result


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
                logger.warning(f"Failed for threshold={threshold}, horizon={horizon}: {e}")

    df = pd.DataFrame(results)
    return df
