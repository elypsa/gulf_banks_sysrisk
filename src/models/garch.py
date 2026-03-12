"""GJR-GARCH(1,1) estimation for volatility modeling.

This module implements asymmetric GARCH models for capturing volatility dynamics
in financial returns. The GJR-GARCH model allows for asymmetric responses to
positive and negative shocks (leverage effect).

Mathematical formulation:
    r_t = μ + ε_t
    ε_t = σ_t × z_t,  z_t ~ N(0, 1)
    σ²_t = ω + α ε²_{t-1} + γ I[ε_{t-1}<0] ε²_{t-1} + β σ²_{t-1}

Where:
    - γ captures the leverage effect (negative shocks → higher volatility)
    - I[·] is indicator function
    - α + β + γ/2 < 1 for stationarity
"""

import pandas as pd
import numpy as np
from arch import arch_model
from typing import Dict
from scipy import stats


def estimate_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean_model: str = "Constant",
    dist: str = "normal"
) -> Dict:
    """Estimate GJR-GARCH(p,q) model for a return series.

    Args:
        returns: Return series (log returns).
        p: GARCH order (default 1).
        q: ARCH order (default 1).
        mean_model: Mean model specification ('Constant', 'Zero', 'AR').
        dist: Error distribution ('normal', 't', 'skewt').

    Returns:
        Dictionary containing:
            - model: Fitted arch model object
            - params: Parameter estimates
            - conditional_volatility: σ_t series
            - standardized_residuals: z_t series
            - aic: Akaike Information Criterion
            - bic: Bayesian Information Criterion
            - ljung_box_p: Ljung-Box test p-value (H0: no autocorrelation)

    Example:
        >>> returns = pd.Series(np.random.randn(1000) * 0.01)
        >>> result = estimate_gjr_garch(returns)
        >>> print(result['params'])
    """
    # Remove missing values
    returns_clean = returns.dropna()

    if len(returns_clean) < 100:
        raise ValueError(f"Insufficient data: {len(returns_clean)} observations (need >= 100)")


    returns_scaled = returns_clean 

    # Estimate GJR-GARCH
    try:
        model = arch_model(
            returns_scaled,
            mean=mean_model,
            vol="GARCH",
            p=p,
            o=1,  # GJR asymmetric term
            q=q,
            dist=dist
        )

        fitted = model.fit(disp="off", show_warning=False)

        # Extract results
        params = fitted.params
        conditional_vol = fitted.conditional_volatility / 100  # Scale back
        std_resid = fitted.std_resid

        # Diagnostic: Ljung-Box test on standardized residuals
        lb_test = ljung_box_test(std_resid, lags=10)

        result = {
            "model": fitted,
            "params": params,
            "conditional_volatility": conditional_vol,
            "standardized_residuals": std_resid,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "ljung_box_p": lb_test["p_value"],
            "loglikelihood": fitted.loglikelihood,
            "convergence": fitted.convergence_flag == 0
        }

        return result

    except Exception as e:
        raise RuntimeError(f"GARCH estimation failed: {e}")


def estimate_garch_multivariate(
    returns_df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    mean_model: str = "Constant",
    dist: str = "normal",
    verbose: bool = True
) -> Dict[str, Dict]:
    """Estimate GJR-GARCH models for multiple return series.

    Args:
        returns_df: DataFrame with returns (columns = instruments).
        p: GARCH order.
        q: ARCH order.
        mean_model: Mean model specification.
        dist: Error distribution.
        verbose: Print progress.

    Returns:
        Dictionary mapping instrument name to GARCH results.

    Example:
        >>> returns = pd.DataFrame({'ENBD.DU': [...], 'DIB.DU': [...]})
        >>> results = estimate_garch_multivariate(returns)
        >>> print(results['ENBD.DU']['params'])
    """
    results = {}
    failed = []

    for col in returns_df.columns:
        if verbose:
            print(f"Estimating GARCH for {col}...", end=" ")

        try:
            result = estimate_gjr_garch(
                returns_df[col],
                p=p,
                q=q,
                mean_model=mean_model,
                dist=dist
            )
            results[col] = result

            if verbose:
                convergence_str = "✓" if result["convergence"] else "✗"
                print(f"{convergence_str} AIC: {result['aic']:.2f}")

        except Exception as e:
            failed.append(col)
            if verbose:
                print(f"✗ FAILED: {e}")

    if verbose:
        print(f"\nSummary: {len(results)}/{len(returns_df.columns)} successful, {len(failed)} failed")
        if failed:
            print(f"Failed instruments: {', '.join(failed)}")

    return results


def extract_volatilities(garch_results: Dict[str, Dict]) -> pd.DataFrame:
    """Extract conditional volatilities from GARCH results.

    Args:
        garch_results: Dictionary of GARCH results from estimate_garch_multivariate.

    Returns:
        DataFrame with conditional volatilities (columns = instruments).

    Example:
        >>> vols = extract_volatilities(garch_results)
        >>> print(vols.head())
    """
    vols = {}

    for instrument, result in garch_results.items():
        vols[instrument] = result["conditional_volatility"]

    df = pd.DataFrame(vols)
    return df


def extract_standardized_residuals(garch_results: Dict[str, Dict]) -> pd.DataFrame:
    """Extract standardized residuals from GARCH results.

    Args:
        garch_results: Dictionary of GARCH results.

    Returns:
        DataFrame with standardized residuals (columns = instruments).

    Note:
        Standardized residuals should have mean ≈ 0 and variance ≈ 1.

    Example:
        >>> resids = extract_standardized_residuals(garch_results)
        >>> print(resids.mean())  # Should be close to 0
        >>> print(resids.std())   # Should be close to 1
    """
    resids = {}

    for instrument, result in garch_results.items():
        resids[instrument] = result["standardized_residuals"]

    df = pd.DataFrame(resids)
    return df


def ljung_box_test(residuals: pd.Series, lags: int = 10) -> Dict:
    """Perform Ljung-Box test for autocorrelation in residuals.

    Tests H0: No autocorrelation in residuals up to lag k.

    Args:
        residuals: Residual series.
        lags: Number of lags to test.

    Returns:
        Dictionary with test statistic and p-value.

    Note:
        - p-value > 0.05: No evidence of autocorrelation (good)
        - p-value < 0.05: Evidence of autocorrelation (model misspecification)

    Example:
        >>> resids = pd.Series(np.random.randn(1000))
        >>> test = ljung_box_test(resids)
        >>> print(f"p-value: {test['p_value']:.4f}")
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    residuals = residuals.reset_index(drop=True)  # Ensure it's a clean Series
    result = acorr_ljungbox(residuals, lags=lags)

    # Return result for specified lag
    return {
        "test_statistic": result.iloc[:,0],
        "p_value": result.iloc[:,1],
        "lags": lags
    }


def garch_diagnostics(garch_results: Dict[str, Dict]) -> pd.DataFrame:
    """Generate diagnostic summary for GARCH results.

    Args:
        garch_results: Dictionary of GARCH results.

    Returns:
        DataFrame with diagnostic statistics per instrument.

    Example:
        >>> diagnostics = garch_diagnostics(garch_results)
        >>> print(diagnostics)
    """
    diagnostics = []

    for instrument, result in garch_results.items():
        # Extract parameters
        params = result["params"]

        # Check stationarity condition (α + β + γ/2 < 1)
        if "alpha[1]" in params and "beta[1]" in params:
            alpha = params.get("alpha[1]", 0)
            beta = params.get("beta[1]", 0)
            gamma = params.get("gamma[1]", 0)
            persistence = alpha + beta + gamma / 2
        else:
            persistence = np.nan

        # Standardized residuals checks
        std_resid = result["standardized_residuals"]
        resid_mean = std_resid.mean()
        resid_std = std_resid.std()

        # Normality test (Jarque-Bera)
        jb_stat, jb_p = stats.jarque_bera(std_resid.dropna())

        diagnostics.append({
            "instrument": instrument,
            "aic": result["aic"],
            "bic": result["bic"],
            "loglik": result["loglikelihood"],
            "persistence": persistence,
            "ljung_box_p": result["ljung_box_p"],
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "jarque_bera_p": jb_p,
            "convergence": result["convergence"]
        })

    df = pd.DataFrame(diagnostics)
    return df


def forecast_volatility(
    garch_result: Dict,
    horizon: int = 1
) -> pd.Series:
    """Forecast conditional volatility h steps ahead.

    Args:
        garch_result: GARCH result dictionary from estimate_gjr_garch.
        horizon: Forecast horizon in periods.

    Returns:
        Series of forecasted volatilities.

    Example:
        >>> forecast = forecast_volatility(garch_result, horizon=22)  # 22 weeks
        >>> print(forecast)
    """
    fitted_model = garch_result["model"]
    forecast = fitted_model.forecast(horizon=horizon)

    # Extract variance forecast and take square root
    variance_forecast = forecast.variance.iloc[-1, :]
    volatility_forecast = np.sqrt(variance_forecast) / 100  # Scale back

    return volatility_forecast


def get_garch_parameters(garch_results: Dict[str, Dict]) -> pd.DataFrame:
    """Extract GARCH parameter estimates for all instruments.

    Args:
        garch_results: Dictionary of GARCH results.

    Returns:
        DataFrame with parameters (rows = instruments, columns = parameters).

    Example:
        >>> params = get_garch_parameters(garch_results)
        >>> print(params[['omega', 'alpha[1]', 'beta[1]', 'gamma[1]']])
    """
    params_list = []

    for instrument, result in garch_results.items():
        params = result["params"].to_dict()
        params["instrument"] = instrument
        params_list.append(params)

    df = pd.DataFrame(params_list)
    df = df.set_index("instrument")

    return df
