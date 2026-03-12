"""Dynamic Conditional Correlation (DCC) estimation.

This module implements the DCC model of Engle (2002) for time-varying correlations.
DCC is a two-step procedure:
1. Estimate univariate GARCH models → standardized residuals z_t
2. Estimate DCC dynamics on z_t → time-varying correlation matrix R_t

Mathematical formulation:
    Q_t = (1 - a - b) Q̄ + a (z_{t-1} z'_{t-1}) + b Q_{t-1}
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

Where:
    - Q̄ = unconditional correlation of standardized residuals
    - a, b = DCC parameters (a + b < 1 for stationarity)
    - R_t = time-varying correlation matrix

References:
    Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of
    Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models."
    Journal of Business & Economic Statistics, 20(3), 339-350.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import warnings


def estimate_dcc(
    standardized_residuals: pd.DataFrame,
    initial_params: Optional[np.ndarray] = None,
    method: str = "SLSQP"
) -> Dict:
    """Estimate DCC model parameters.

    Args:
        standardized_residuals: DataFrame of standardized residuals from GARCH
                                (columns = instruments, index = dates).
        initial_params: Initial values for [a, b] (default: [0.01, 0.95]).
        method: Optimization method ('SLSQP', 'L-BFGS-B').

    Returns:
        Dictionary containing:
            - params: Estimated [a, b] parameters
            - Q_bar: Unconditional correlation matrix
            - Q_series: Time series of Q_t matrices
            - R_series: Time series of correlation matrices R_t
            - loglikelihood: Log-likelihood value
            - success: Whether optimization converged

    Example:
        >>> std_resids = pd.DataFrame({'ENBD.DU': [...], 'DIB.DU': [...]})
        >>> dcc_result = estimate_dcc(std_resids)
        >>> print(f"DCC params: a={dcc_result['params'][0]:.4f}, b={dcc_result['params'][1]:.4f}")
    """
    # Remove missing values
    resids = standardized_residuals.dropna()

    if len(resids) < 100:
        raise ValueError(f"Insufficient data: {len(resids)} observations")

    # Unconditional correlation matrix (Q̄)
    Q_bar = resids.corr().values

    # Ensure positive definite
    Q_bar = make_positive_definite(Q_bar)

    # Initial parameters
    if initial_params is None:
        initial_params = np.array([0.01, 0.95])

    # Bounds: a, b > 0 and a + b < 1
    bounds = [(1e-6, 0.99), (1e-6, 0.99)]

    # Constraint: a + b < 1
    constraints = {"type": "ineq", "fun": lambda x: 0.999 - (x[0] + x[1])}

    # Optimization
    def objective(params):
        """Negative log-likelihood for DCC."""
        return -dcc_loglikelihood(params, resids.values, Q_bar)

    try:
        result = minimize(
            objective,
            initial_params,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000}
        )

        if not result.success:
            warnings.warn(f"DCC optimization did not converge: {result.message}")

        # Extract results
        a, b = result.x

        # Generate Q_t and R_t series
        Q_series, R_series = generate_dcc_correlations(resids.values, Q_bar, a, b)

        return {
            "params": result.x,
            "Q_bar": Q_bar,
            "Q_series": Q_series,
            "R_series": R_series,
            "loglikelihood": -result.fun,
            "success": result.success,
            "message": result.message
        }

    except Exception as e:
        raise RuntimeError(f"DCC estimation failed: {e}")


def dcc_loglikelihood(params: np.ndarray, resids: np.ndarray, Q_bar: np.ndarray) -> float:
    """Compute log-likelihood for DCC model.

    Args:
        params: DCC parameters [a, b].
        resids: Standardized residuals (T × N).
        Q_bar: Unconditional correlation matrix (N × N).

    Returns:
        Log-likelihood value.
    """
    a, b = params
    T, N = resids.shape

    # Check stationarity
    if a + b >= 1:
        return -1e10  # Invalid parameters

    # Initialize Q_0 = Q̄
    Q_t = Q_bar.copy()

    loglik = 0.0

    for t in range(T):
        z_t = resids[t, :].reshape(-1, 1)

        # Update Q_t
        if t > 0:
            Q_t = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q_t

        # Ensure positive definite
        Q_t = make_positive_definite(Q_t)

        # Compute R_t
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q_t)))
        R_t = D_inv @ Q_t @ D_inv

        # Ensure valid correlation matrix
        R_t = make_correlation_matrix(R_t)

        # Log-likelihood contribution
        try:
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0:
                return -1e10  # Not positive definite

            inv_R_t = np.linalg.inv(R_t)
            quad_form = z_t.T @ inv_R_t @ z_t

            loglik += -0.5 * (logdet + quad_form[0, 0])

        except np.linalg.LinAlgError:
            return -1e10  # Singular matrix

    return loglik


def generate_dcc_correlations(
    resids: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time series of Q_t and R_t matrices.

    Args:
        resids: Standardized residuals (T × N).
        Q_bar: Unconditional correlation matrix.
        a: DCC parameter a.
        b: DCC parameter b.

    Returns:
        Tuple of (Q_series, R_series) where each is (T × N × N) array.
    """
    T, N = resids.shape

    Q_series = np.zeros((T, N, N))
    R_series = np.zeros((T, N, N))

    Q_t = Q_bar.copy()

    for t in range(T):
        z_t = resids[t, :].reshape(-1, 1)

        # Update Q_t
        if t > 0:
            Q_t = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q_t

        # Ensure positive definite
        Q_t = make_positive_definite(Q_t)

        # Compute R_t
        D_inv = np.diag(1.0 / np.sqrt(np.diag(Q_t)))
        R_t = D_inv @ Q_t @ D_inv

        # Ensure valid correlation
        R_t = make_correlation_matrix(R_t)

        Q_series[t, :, :] = Q_t
        R_series[t, :, :] = R_t

    return Q_series, R_series


def extract_pairwise_correlation(
    R_series: np.ndarray,
    instrument_names: list,
    i: int,
    j: int
) -> pd.Series:
    """Extract time series of correlation between two instruments.

    Args:
        R_series: Time series of correlation matrices (T × N × N).
        instrument_names: List of instrument names.
        i: Index of first instrument.
        j: Index of second instrument.

    Returns:
        Time series of correlation ρ_ij,t.

    Example:
        >>> corr_ENBD_DIB = extract_pairwise_correlation(R_series, names, 0, 1)
        >>> print(corr_ENBD_DIB.mean())
    """
    correlations = R_series[:, i, j]
    return pd.Series(correlations, name=f"{instrument_names[i]}_vs_{instrument_names[j]}")


def extract_market_correlations(
    R_series: np.ndarray,
    instrument_names: list,
    market_idx: int = 0
) -> pd.DataFrame:
    """Extract correlations of all instruments with market benchmark.

    Args:
        R_series: Time series of correlation matrices (T × N × N).
        instrument_names: List of instrument names.
        market_idx: Index of market benchmark (typically first column).

    Returns:
        DataFrame with correlations (columns = instruments).

    Example:
        >>> # Assuming first column is market benchmark
        >>> market_corrs = extract_market_correlations(R_series, names, market_idx=0)
        >>> print(market_corrs.mean())  # Average correlation with market
    """
    T, N, _ = R_series.shape

    corrs = {}
    for i, name in enumerate(instrument_names):
        if i != market_idx:
            corrs[name] = R_series[:, market_idx, i]

    df = pd.DataFrame(corrs)
    return df


def make_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Ensure matrix is positive definite using eigenvalue adjustment.

    Args:
        matrix: Square matrix (N × N).
        epsilon: Minimum eigenvalue threshold.

    Returns:
        Positive definite matrix.
    """
    # Symmetrize
    matrix = (matrix + matrix.T) / 2

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)

    # Reconstruct matrix
    matrix_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return matrix_pd


def make_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensure matrix is a valid correlation matrix (diagonal = 1, symmetric).

    Args:
        matrix: Candidate correlation matrix.

    Returns:
        Valid correlation matrix.
    """
    # Ensure symmetric
    matrix = (matrix + matrix.T) / 2

    # Ensure diagonal = 1
    np.fill_diagonal(matrix, 1.0)

    # Clip off-diagonal to [-1, 1]
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            matrix[i, j] = np.clip(matrix[i, j], -0.999, 0.999)
            matrix[j, i] = matrix[i, j]

    return matrix


def estimate_ccc(standardized_residuals: pd.DataFrame) -> Dict:
    """Estimate Constant Conditional Correlation (CCC) model.

    Simpler alternative to DCC where correlation is constant over time.

    Args:
        standardized_residuals: DataFrame of standardized residuals.

    Returns:
        Dictionary with constant correlation matrix.

    Note:
        Use CCC as fallback if DCC fails to converge.

    Example:
        >>> ccc_result = estimate_ccc(std_resids)
        >>> print(ccc_result['R'])  # Constant correlation matrix
    """
    resids = standardized_residuals.dropna()

    # Unconditional correlation
    R = resids.corr().values

    # Ensure positive definite
    R = make_positive_definite(R)
    R = make_correlation_matrix(R)

    return {
        "R": R,
        "model": "CCC",
        "success": True
    }


def dcc_diagnostics(dcc_result: Dict, instrument_names: list) -> pd.DataFrame:
    """Generate diagnostic summary for DCC estimation.

    Args:
        dcc_result: DCC result dictionary.
        instrument_names: List of instrument names.

    Returns:
        DataFrame with diagnostic statistics.
    """
    a, b = dcc_result["params"]
    R_series = dcc_result["R_series"]

    # Compute average correlations over time
    avg_corrs = {}
    std_corrs = {}

    for i in range(len(instrument_names)):
        for j in range(i + 1, len(instrument_names)):
            pair = f"{instrument_names[i]}_vs_{instrument_names[j]}"
            corr_series = R_series[:, i, j]
            avg_corrs[pair] = corr_series.mean()
            std_corrs[pair] = corr_series.std()

    diagnostics = {
        "dcc_a": a,
        "dcc_b": b,
        "persistence": a + b,
        "avg_correlation": np.mean(list(avg_corrs.values())),
        "std_correlation": np.mean(list(std_corrs.values())),
        "loglikelihood": dcc_result["loglikelihood"],
        "success": dcc_result["success"]
    }

    return pd.DataFrame([diagnostics])
