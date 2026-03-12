"""SRISK (Systemic Risk) calculation module.

SRISK measures the expected capital shortfall of a financial institution
conditional on a systemic crisis. It quantifies how much capital a bank would
need to maintain minimum prudential requirements during a market-wide crisis.

Core Formula:
    SRISK_i,t = max(0, k × Debt_i,t - (1 - k) × (1 - LRMES_i,t) × Equity_i,t)

Where:
    - k = Prudential capital ratio (e.g., 0.08 for Basel, 0.055 for IFRS)
    - Debt_i,t = Book value of liabilities (USD)
    - Equity_i,t = Market capitalization (USD)
    - LRMES_i,t = Long-Run Marginal Expected Shortfall

Interpretation:
    - SRISK > 0: Bank undercapitalized in crisis (needs external capital)
    - SRISK = 0: Bank adequately capitalized
    - Higher SRISK → Greater systemic importance and vulnerability

System-wide aggregation:
    SRISK_system = Σ max(0, SRISK_i) for all banks i

Only positive SRISK values are summed (capital cannot be easily mobilized
across institutions during a crisis).

References:
    Brownlees, C. T., & Engle, R. F. (2017). "SRISK: A Conditional Capital
    Shortfall Measure of Systemic Risk." Review of Financial Studies, 30(1), 48-79.
"""

import pandas as pd
from typing import Dict, Tuple


def calculate_srisk(
    debt: pd.Series,
    equity: pd.Series,
    lrmes: pd.Series,
    k: float = 0.08
) -> pd.Series:
    """Calculate SRISK for a single bank.

    Args:
        debt: Book value of liabilities (USD).
        equity: Market capitalization (USD).
        lrmes: Long-Run Marginal Expected Shortfall.
        k: Prudential capital ratio (default 0.08 = 8%).

    Returns:
        Time series of SRISK values.

    Mathematical formulation:
        SRISK_t = max(0, k × D_t - (1-k) × (1-LRMES_t) × W_t)

    Example:
        >>> srisk = calculate_srisk(debt, equity, lrmes, k=0.08)
        >>> print(f"Latest SRISK: ${srisk.iloc[-1]/1e9:.2f}B")
    """
    # Capital shortfall formula
    capital_shortfall = k * debt - (1 - k) * (1 - lrmes) * equity

    # SRISK = max(0, shortfall)
    srisk = capital_shortfall.clip(lower=0)

    return srisk


def calculate_srisk_multi(
    aligned_data: pd.DataFrame,
    lrmes_df: pd.DataFrame,
    bank_universe: pd.DataFrame,
    k: float = 0.08
) -> pd.DataFrame:
    """Calculate SRISK for all banks in the universe.

    Args:
        aligned_data: DataFrame with debt and equity data.
                      Columns: fund_{RIC}_TotLiab, mktcap_{RIC}
        lrmes_df: DataFrame with LRMES values (columns = instruments).
        bank_universe: Bank universe DataFrame with bank_ric column.
        k: Prudential capital ratio.

    Returns:
        DataFrame with SRISK time series (columns = bank RICs).

    Example:
        >>> srisk_all = calculate_srisk_multi(aligned, lrmes, universe, k=0.08)
        >>> print(srisk_all.iloc[-1].sort_values(ascending=False).head())
    """
    srisk_results = {}

    for _, row in bank_universe.iterrows():
        bank_ric = row["bank_ric"]

        # Find columns in aligned_data
        debt_col = f"fund_{bank_ric}_TotLiab"
        equity_col = f"mktcap_{bank_ric}"

        if debt_col not in aligned_data.columns or equity_col not in aligned_data.columns:
            print(f"⚠ Skipping {bank_ric}: Missing debt or equity data")
            continue

        if bank_ric not in lrmes_df.columns:
            print(f"⚠ Skipping {bank_ric}: Missing LRMES")
            continue

        # Extract data
        debt = aligned_data[debt_col]
        equity = aligned_data[equity_col]
        lrmes = lrmes_df[bank_ric]

        # Calculate SRISK
        srisk = calculate_srisk(debt, equity, lrmes, k)
        srisk_results[bank_ric] = srisk

    df = pd.DataFrame(srisk_results)
    return df


def calculate_system_srisk(srisk_df: pd.DataFrame) -> pd.Series:
    """Calculate system-wide SRISK (sum of positive SRISK).

    Args:
        srisk_df: DataFrame with SRISK per bank (columns = banks).

    Returns:
        Time series of system-wide SRISK.

    Note:
        Only positive SRISK values are summed. Banks with negative SRISK
        (capital surplus) are excluded from aggregation.

    Example:
        >>> system_srisk = calculate_system_srisk(srisk_df)
        >>> print(f"System SRISK: ${system_srisk.iloc[-1]/1e9:.2f}B")
    """
    # Sum positive SRISK only (max with 0 already applied in calculate_srisk)
    system_srisk = srisk_df.sum(axis=1)

    return system_srisk


def calculate_srisk_contribution(srisk_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each bank's contribution to system-wide SRISK (%).

    Args:
        srisk_df: DataFrame with SRISK per bank.

    Returns:
        DataFrame with SRISK contributions (%) per bank.

    Example:
        >>> contributions = calculate_srisk_contribution(srisk_df)
        >>> print(contributions.iloc[-1].sort_values(ascending=False).head())
    """
    system_srisk = calculate_system_srisk(srisk_df)

    # Avoid division by zero
    contributions = srisk_df.div(system_srisk, axis=0) * 100

    # Handle cases where system SRISK = 0
    contributions = contributions.fillna(0)

    return contributions


def decompose_srisk(
    debt: pd.Series,
    equity: pd.Series,
    lrmes: pd.Series,
    k: float = 0.08
) -> Dict[str, pd.Series]:
    """Decompose SRISK into size, leverage, and risk effects.

    This decomposition helps identify what drives changes in SRISK:
    - Size effect: Changes in market cap (bank size)
    - Leverage effect: Changes in debt-to-equity ratio
    - Risk effect: Changes in LRMES (correlation with market)

    Args:
        debt: Book value of liabilities.
        equity: Market capitalization.
        lrmes: Long-Run Marginal Expected Shortfall.
        k: Prudential capital ratio.

    Returns:
        Dictionary with:
            - srisk: Total SRISK
            - size_effect: Impact of equity changes
            - leverage_effect: Impact of debt changes
            - risk_effect: Impact of LRMES changes

    Mathematical decomposition:
        ΔSRISK ≈ ∂SRISK/∂W × ΔW + ∂SRISK/∂D × ΔD + ∂SRISK/∂LRMES × ΔLRMES

    Example:
        >>> decomp = decompose_srisk(debt, equity, lrmes)
        >>> print(decomp['risk_effect'].tail())
    """
    srisk = calculate_srisk(debt, equity, lrmes, k)

    # Partial derivatives
    # ∂SRISK/∂W = -(1-k)(1-LRMES)
    d_srisk_d_equity = -(1 - k) * (1 - lrmes)

    # ∂SRISK/∂D = k
    d_srisk_d_debt = k

    # ∂SRISK/∂LRMES = (1-k) × W
    d_srisk_d_lrmes = (1 - k) * equity

    # Changes
    delta_equity = equity.diff()
    delta_debt = debt.diff()
    delta_lrmes = lrmes.diff()

    # Effects
    size_effect = d_srisk_d_equity * delta_equity
    leverage_effect = d_srisk_d_debt * delta_debt
    risk_effect = d_srisk_d_lrmes * delta_lrmes

    return {
        "srisk": srisk,
        "size_effect": size_effect,
        "leverage_effect": leverage_effect,
        "risk_effect": risk_effect,
        "total_change": srisk.diff()
    }


def calculate_leverage_ratio(debt: pd.Series, equity: pd.Series) -> pd.Series:
    """Calculate quasi-leverage ratio.

    Quasi-leverage = (Debt + Equity) / Equity

    Args:
        debt: Book value of liabilities.
        equity: Market capitalization.

    Returns:
        Time series of leverage ratios.

    Example:
        >>> leverage = calculate_leverage_ratio(debt, equity)
        >>> print(f"Average leverage: {leverage.mean():.2f}x")
    """
    return (debt + equity) / equity


def srisk_summary_statistics(srisk_df: pd.DataFrame, bank_universe: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for SRISK.

    Args:
        srisk_df: DataFrame with SRISK per bank.
        bank_universe: Bank universe with metadata.

    Returns:
        DataFrame with summary statistics per bank.

    Example:
        >>> summary = srisk_summary_statistics(srisk_df, universe)
        >>> print(summary.sort_values('mean_srisk', ascending=False))
    """
    summary = []

    for bank_ric in srisk_df.columns:
        srisk_series = srisk_df[bank_ric]

        # Find country
        country_info = bank_universe[bank_universe["bank_ric"] == bank_ric]
        country = country_info["country_chain"].iloc[0] if len(country_info) > 0 else "Unknown"

        summary.append({
            "bank_ric": bank_ric,
            "country": country,
            "mean_srisk": srisk_series.mean(),
            "median_srisk": srisk_series.median(),
            "max_srisk": srisk_series.max(),
            "latest_srisk": srisk_series.iloc[-1],
            "pct_positive": (srisk_series > 0).sum() / len(srisk_series) * 100
        })

    df = pd.DataFrame(summary)
    df = df.sort_values("mean_srisk", ascending=False).reset_index(drop=True)

    return df


def calculate_srisk_by_country(
    srisk_df: pd.DataFrame,
    bank_universe: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate SRISK by country.

    Args:
        srisk_df: DataFrame with SRISK per bank.
        bank_universe: Bank universe with country mapping.

    Returns:
        DataFrame with SRISK time series per country.

    Example:
        >>> country_srisk = calculate_srisk_by_country(srisk_df, universe)
        >>> print(country_srisk.iloc[-1].sort_values(ascending=False))
    """
    # Map banks to countries
    bank_to_country = dict(zip(bank_universe["bank_ric"], bank_universe["country_chain"]))

    # Group by country
    country_srisk = {}

    for country in bank_universe["country_chain"].unique():
        banks_in_country = [b for b, c in bank_to_country.items() if c == country and b in srisk_df.columns]

        if banks_in_country:
            country_srisk[country] = srisk_df[banks_in_country].sum(axis=1)

    df = pd.DataFrame(country_srisk)
    return df


def sensitivity_analysis_capital_ratio(
    debt: pd.Series,
    equity: pd.Series,
    lrmes: pd.Series,
    k_values: list = [0.055, 0.08, 0.10]
) -> pd.DataFrame:
    """Perform sensitivity analysis on capital ratio k.

    Args:
        debt: Book value of liabilities.
        equity: Market capitalization.
        lrmes: LRMES values.
        k_values: List of capital ratios to test.

    Returns:
        DataFrame with SRISK under different k values.

    Example:
        >>> sensitivity = sensitivity_analysis_capital_ratio(
        ...     debt, equity, lrmes, k_values=[0.055, 0.08, 0.10]
        ... )
        >>> print(sensitivity.iloc[-1])
    """
    results = {}

    for k in k_values:
        srisk = calculate_srisk(debt, equity, lrmes, k)
        results[f"k={k:.3f}"] = srisk

    df = pd.DataFrame(results)
    return df


def calculate_srisk_dual_capital_ratios(
    aligned_data: pd.DataFrame,
    lrmes_df: pd.DataFrame,
    bank_universe: pd.DataFrame,
    k_basel: float = 0.08,
    k_ifrs: float = 0.055
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate SRISK with both Basel and IFRS capital ratios.

    Args:
        aligned_data: Aligned dataset with debt and equity.
        lrmes_df: LRMES estimates.
        bank_universe: Bank universe.
        k_basel: Basel capital ratio (default 0.08).
        k_ifrs: IFRS capital ratio (default 0.055).

    Returns:
        Tuple of (srisk_basel, srisk_ifrs) DataFrames.

    Example:
        >>> srisk_basel, srisk_ifrs = calculate_srisk_dual_capital_ratios(
        ...     aligned, lrmes, universe
        ... )
    """
    srisk_basel = calculate_srisk_multi(aligned_data, lrmes_df, bank_universe, k=k_basel)
    srisk_ifrs = calculate_srisk_multi(aligned_data, lrmes_df, bank_universe, k=k_ifrs)

    return srisk_basel, srisk_ifrs
