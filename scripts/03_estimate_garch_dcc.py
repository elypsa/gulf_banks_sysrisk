#!/usr/bin/env python3
"""Script to estimate GARCH-DCC models and calculate LRMES.

This script implements Phase 3 of the SRISK pipeline:
1. Load processed returns data
2. Estimate GJR-GARCH(1,1) for each bank + market benchmark
3. Estimate Dynamic Conditional Correlation (DCC)
4. Perform Monte Carlo simulation (50,000+ paths)
5. Calculate LRMES (Long-Run Marginal Expected Shortfall)
6. Save results for SRISK calculation

Usage:
    uv run scripts/03_estimate_garch_dcc.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataframe, save_dataframe
from src.models.garch import (
    estimate_garch_multivariate,
    extract_volatilities,
    extract_standardized_residuals,
    garch_diagnostics,
    get_garch_parameters
)
from src.models.dcc import estimate_dcc, dcc_diagnostics, estimate_ccc
from src.models.lrmes import (
    simulate_garch_dcc_returns,
    calculate_lrmes,
    lrmes_diagnostics
)
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd


def load_processed_data():
    """Load processed returns and bank universe."""
    print("\n" + "=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)

    returns = load_dataframe("returns_clean", PROCESSED_DATA_DIR)
    universe = load_dataframe("banks_universe_clean", PROCESSED_DATA_DIR)

    print(f"\nReturns shape: {returns.shape}")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    print(f"Banks: {len(universe)}")

    return returns, universe


def estimate_garch_models(returns_df):
    """Estimate GJR-GARCH(1,1) for all instruments."""
    print("\n" + "=" * 60)
    print("ESTIMATING GJR-GARCH(1,1) MODELS")
    print("=" * 60)
    print(f"Estimating for {len(returns_df.columns)} instruments...")
    print()

    garch_results = estimate_garch_multivariate(
        returns_df,
        p=CONFIG.GARCH_P,
        q=CONFIG.GARCH_Q,
        mean_model="Constant",
        dist="normal",
        verbose=True
    )

    # Extract diagnostics
    diagnostics = garch_diagnostics(garch_results)

    print(f"\n{diagnostics['convergence'].sum()}/{len(diagnostics)} models converged")
    print(f"Average persistence: {diagnostics['persistence'].mean():.4f}")
    avg_p = diagnostics['ljung_box_p'].apply(pd.Series).mean()
    print(f"Average Ljung-Box p-values:\n{avg_p.round(4)}")

    # Save diagnostics
    metadata = {
        "description": "GARCH estimation diagnostics",
        "model": "GJR-GARCH(1,1)",
        "n_instruments": len(garch_results),
        "convergence_rate": float(diagnostics["convergence"].sum() / len(diagnostics))
    }
    diagnostics.drop('ljung_box_p', axis=1, inplace=True) # remove p-values from diagnostics as they are vectors
    save_dataframe(diagnostics, "garch_diagnostics", RESULTS_DATA_DIR, metadata)

    # Save parameters
    params = get_garch_parameters(garch_results)
    save_dataframe(params, "garch_parameters", RESULTS_DATA_DIR, {"description": "GARCH parameters"})

    # Save conditional volatilities
    vols = extract_volatilities(garch_results)
    save_dataframe(vols, "conditional_volatilities", RESULTS_DATA_DIR, {"description": "GARCH conditional volatilities"})

    return garch_results


def estimate_dcc_model(garch_results):
    """Estimate Dynamic Conditional Correlation."""
    print("\n" + "=" * 60)
    print("ESTIMATING DYNAMIC CONDITIONAL CORRELATION (DCC)")
    print("=" * 60)

    # Extract standardized residuals
    std_resids = extract_standardized_residuals(garch_results)

    print(f"Standardized residuals shape: {std_resids.shape}")
    print(f"Mean: {std_resids.mean().mean():.4f} (should be ~0)")
    print(f"Std: {std_resids.std().mean():.4f} (should be ~1)")

    # Estimate DCC
    try:
        print("\nEstimating DCC...")
        dcc_result = estimate_dcc(std_resids, initial_params=[0.01, 0.95])

        if dcc_result["success"]:
            a, b = dcc_result["params"]
            print(f"✓ DCC converged: a={a:.4f}, b={b:.4f}, persistence={a+b:.4f}")
        else:
            print(f"⚠ DCC did not fully converge: {dcc_result['message']}")

        # Diagnostics
        dcc_diag = dcc_diagnostics(dcc_result, list(std_resids.columns))
        print(f"Average correlation: {dcc_diag['avg_correlation'].iloc[0]:.4f}")
        print(f"Correlation volatility: {dcc_diag['std_correlation'].iloc[0]:.4f}")

        save_dataframe(dcc_diag, "dcc_diagnostics", RESULTS_DATA_DIR, {"description": "DCC diagnostics"})

        return dcc_result

    except Exception as e:
        print(f"✗ DCC estimation failed: {e}")
        print("Falling back to Constant Conditional Correlation (CCC)...")

        ccc_result = estimate_ccc(std_resids)
        print(f"✓ Using CCC (constant correlation)")

        return ccc_result


def calculate_lrmes_estimates(garch_results, dcc_result, instrument_names):
    """Calculate LRMES via Monte Carlo simulation."""
    print("\n" + "=" * 60)
    print("CALCULATING LRMES VIA MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"Horizon: {CONFIG.CRISIS_HORIZON_WEEKS} weeks (6 months)")
    print(f"Crisis threshold: {CONFIG.CRISIS_THRESHOLD * 100:.0f}% market decline")
    print(f"Number of simulations: {CONFIG.N_SIMULATIONS:,}")
    print()

    # Prepare GARCH parameters
    garch_params = {name: garch_results[name] for name in instrument_names if name in garch_results}

    # Simulate
    print("Running Monte Carlo simulation...")
    simulations = simulate_garch_dcc_returns(
        garch_params=garch_params,
        dcc_result=dcc_result,
        instrument_names=instrument_names,
        horizon=CONFIG.CRISIS_HORIZON_WEEKS,
        n_simulations=CONFIG.N_SIMULATIONS,
        random_seed=CONFIG.RANDOM_SEED
    )

    print("✓ Simulation complete")
    print(f"Simulated returns shape: {simulations['simulated_returns'].shape}")

    # Calculate LRMES
    print("\nCalculating LRMES...")
    lrmes_values = calculate_lrmes(
        simulations=simulations,
        crisis_threshold=CONFIG.CRISIS_THRESHOLD,
        market_idx=0,  # First column is market benchmark
        min_crisis_scenarios=500
    )

    # Diagnostics
    lrmes_diag = lrmes_diagnostics(lrmes_values, simulations)
    print(f"\n✓ LRMES calculated")
    print(f"Crisis scenarios: {lrmes_values['_n_crisis_scenarios']} ({lrmes_values['_crisis_probability']*100:.2f}%)")
    print(f"Average LRMES: {lrmes_diag['avg_lrmes'].iloc[0]:.4f}")
    print(f"Range: [{lrmes_diag['min_lrmes'].iloc[0]:.4f}, {lrmes_diag['max_lrmes'].iloc[0]:.4f}]")

    # Save diagnostics
    save_dataframe(lrmes_diag, "lrmes_diagnostics", RESULTS_DATA_DIR, {"description": "LRMES diagnostics"})

    # Convert to DataFrame
    lrmes_df = pd.DataFrame({
        "instrument": [k for k in lrmes_values.keys() if not k.startswith("_")],
        "lrmes": [lrmes_values[k] for k in lrmes_values.keys() if not k.startswith("_")]
    })

    save_dataframe(lrmes_df, "lrmes_estimates", RESULTS_DATA_DIR, {
        "description": "LRMES estimates",
        "horizon_weeks": CONFIG.CRISIS_HORIZON_WEEKS,
        "crisis_threshold": CONFIG.CRISIS_THRESHOLD,
        "n_simulations": CONFIG.N_SIMULATIONS,
        "n_crisis_scenarios": lrmes_values["_n_crisis_scenarios"]
    })

    return lrmes_df


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK: GARCH-DCC & LRMES ESTIMATION")
    print("=" * 60)

    try:
        # 1. Load data
        returns, universe = load_processed_data()

        # 2. Estimate GARCH
        garch_results = estimate_garch_models(returns)

        # 3. Estimate DCC
        dcc_result = estimate_dcc_model(garch_results)

        # 4. Calculate LRMES
        instrument_names = list(returns.columns)
        lrmes_df = calculate_lrmes_estimates(garch_results, dcc_result, instrument_names)

        # Summary
        print("\n" + "=" * 60)
        print("✓ GARCH-DCC & LRMES ESTIMATION COMPLETED")
        print("=" * 60)
        print("\nTop 10 banks by LRMES:")
        print(lrmes_df.sort_values("lrmes", ascending=False).head(10).to_string(index=False))

        print(f"\nResults saved to: {RESULTS_DATA_DIR}")
        print("\nNext step: Run scripts/04_calculate_srisk.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
