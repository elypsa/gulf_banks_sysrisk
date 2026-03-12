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
    simulate_market_paths,
    simulate_bank_conditional_on_market,
    calculate_lrmes_with_common_market
)
from src.utils.config import CONFIG, PROCESSED_DATA_DIR, RESULTS_DATA_DIR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


def load_processed_data():
    """Load processed returns and bank universe."""
    print("\n" + "=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)

    # Load bank returns and benchmarks separately
    bank_returns = load_dataframe("returns_banks_clean", PROCESSED_DATA_DIR)
    benchmark_returns = load_dataframe("returns_benchmarks_clean", PROCESSED_DATA_DIR)
    universe = load_dataframe("banks_universe_clean", PROCESSED_DATA_DIR)

    print(f"\nBank returns shape: {bank_returns.shape}")
    print(f"Benchmark returns shape: {benchmark_returns.shape}")
    print(f"Date range: {bank_returns.index.min()} to {bank_returns.index.max()}")
    print(f"Banks: {len(universe)}")
    print(f"Benchmarks available: {', '.join(benchmark_returns.columns.tolist())}")

    return bank_returns, benchmark_returns, universe


def estimate_benchmark_garch(benchmark_returns, benchmark_name):
    """Estimate GJR-GARCH(1,1) for the benchmark once."""
    print("\n" + "=" * 60)
    print(f"ESTIMATING BENCHMARK GARCH: {benchmark_name}")
    print("=" * 60)

    benchmark_series = benchmark_returns[benchmark_name]

    garch_results = estimate_garch_multivariate(
        pd.DataFrame({benchmark_name: benchmark_series}),
        p=CONFIG.GARCH_P,
        q=CONFIG.GARCH_Q,
        mean_model="Constant",
        dist="normal",
        verbose=True
    )

    # Diagnostics
    diagnostics = garch_diagnostics(garch_results)
    print(f"✓ Benchmark GARCH converged: {diagnostics['convergence'].iloc[0]}")
    print(f"  Persistence: {diagnostics['persistence'].iloc[0]:.4f}")

    return garch_results[benchmark_name]


def process_bank_with_common_market(
    bank_ric: str,
    bank_series: pd.Series,
    benchmark_garch_result,
    market_paths: dict,
    benchmark_name: str,
    verbose: bool = False
) -> dict:
    """Process one bank using COMMON market paths (systemic risk measurement).

    This implements the corrected bivariate specification:
    - All banks evaluated under THE SAME market crisis scenarios
    - Bank returns simulated CONDITIONAL on fixed market paths
    - Uses conditional distribution: z_i,t | z_m,t ~ N(ρ_t × z_m,t, 1 - ρ²_t)

    Args:
        bank_ric: Bank RIC identifier.
        bank_series: Bank return series.
        benchmark_garch_result: Pre-estimated GARCH result for benchmark.
        market_paths: Pre-simulated market paths (shared across all banks).
        benchmark_name: Name of the benchmark.
        verbose: Whether to print progress.

    Returns:
        Dictionary with LRMES, GARCH params, DCC params, diagnostics.
    """
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Processing: {bank_ric}")
        print(f"{'─' * 60}")

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
            print(f"  ⚠ GARCH did not converge for {bank_ric}")
            return None

        if verbose:
            print(f"  ✓ Bank GARCH: persistence={bank_diag['persistence'].iloc[0]:.4f}")

        # 2. Extract standardized residuals for bivariate DCC
        bivariate_garch = {
            bank_ric: bank_garch_result,
            benchmark_name: benchmark_garch_result
        }
        std_resids = extract_standardized_residuals(bivariate_garch)

        # Reorder: [bank, benchmark]
        std_resids = std_resids[[bank_ric, benchmark_name]]

        if verbose:
            print(f"  Standardized residuals: mean={std_resids.mean().mean():.4f}, std={std_resids.std().mean():.4f}")

        # 3. Estimate bivariate DCC (2x2)
        try:
            dcc_result = estimate_dcc(std_resids, initial_params=[0.01, 0.95])

            if dcc_result["success"]:
                a, b = dcc_result["params"]
                if verbose:
                    print(f"  ✓ DCC: a={a:.4f}, b={b:.4f}, persistence={a+b:.4f}")
            else:
                if verbose:
                    print(f"  ⚠ DCC: {dcc_result['message']}, using result anyway")

        except Exception as e:
            if verbose:
                print(f"  ⚠ DCC failed ({e}), falling back to CCC")
            dcc_result = estimate_ccc(std_resids)

        # Get average correlation
        dcc_diag = dcc_diagnostics(dcc_result, [bank_ric, benchmark_name])
        avg_corr = dcc_diag['avg_correlation'].iloc[0]
        if verbose:
            print(f"  Average correlation: {avg_corr:.4f}")

        # 4. Simulate bank returns CONDITIONAL on common market paths
        # This is the KEY improvement: all banks face the SAME market scenarios
        bank_paths = simulate_bank_conditional_on_market(
            bank_garch_result=bank_garch_result,
            market_paths=market_paths,
            dcc_result=dcc_result,
            random_seed=None  # Different random seed for bank-specific shocks
        )

        # 5. Calculate LRMES using common market
        lrmes_dict = calculate_lrmes_with_common_market(
            bank_cumulative_returns=bank_paths["cumulative_returns"],
            market_cumulative_returns=market_paths["cumulative_returns"],
            crisis_threshold=CONFIG.CRISIS_THRESHOLD,
            min_crisis_scenarios=500
        )

        bank_lrmes = lrmes_dict['lrmes']
        n_crisis = lrmes_dict['n_crisis_scenarios']

        if verbose:
            print(f"  ✓ LRMES: {bank_lrmes:.4f} ({n_crisis} crisis scenarios)")
            print(f"  Bank avg loss in crisis: {lrmes_dict['avg_bank_loss_in_crisis']:.4f}")

        # 6. Compile results
        result = {
            'bank_ric': bank_ric,
            'lrmes': bank_lrmes,
            'n_crisis_scenarios': n_crisis,
            'crisis_probability': lrmes_dict['crisis_probability'],
            'avg_bank_loss_in_crisis': lrmes_dict['avg_bank_loss_in_crisis'],
            'avg_market_loss_in_crisis': lrmes_dict['avg_market_loss_in_crisis'],
            'garch_persistence': bank_diag['persistence'].iloc[0],
            'dcc_a': dcc_result['params'][0],
            'dcc_b': dcc_result['params'][1],
            'dcc_persistence': sum(dcc_result['params']),
            'avg_correlation': avg_corr,
            'convergence': True
        }

        return result

    except Exception as e:
        print(f"  ✗ ERROR processing {bank_ric}: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_market_paths(
    market_paths: dict,
    crisis_threshold: float,
    benchmark_name: str,
    save_path: str,
    n_sample_paths: int = 500
):
    """Visualize simulated market paths and save chart.

    Creates a comprehensive visualization showing:
    - Sample of simulated market paths
    - Crisis threshold line
    - Distribution of final returns
    - Crisis vs non-crisis scenarios

    Args:
        market_paths: Dictionary from simulate_market_paths().
        crisis_threshold: Crisis threshold in log returns (e.g., ln(0.60) ≈ -0.51).
        benchmark_name: Name of market benchmark.
        save_path: Path to save the figure.
        n_sample_paths: Number of paths to plot (default 500).
    """
    # Convert log threshold back to percentage for display
    crisis_threshold_pct = (np.exp(crisis_threshold) - 1) * 100
    cumulative_returns = market_paths["cumulative_returns"]
    returns = market_paths["returns"]
    n_simulations = market_paths["n_simulations"]
    horizon = market_paths["horizon"]

    # Identify crisis scenarios
    crisis_mask = cumulative_returns < crisis_threshold
    n_crisis = crisis_mask.sum()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Sample paths (crisis vs non-crisis)
    ax1 = fig.add_subplot(gs[0:2, 0])

    # Sample paths to plot
    n_plot = min(n_sample_paths, n_simulations)
    sample_indices = np.random.choice(n_simulations, n_plot, replace=False)

    # Compute cumulative returns over time for sample paths
    cum_returns_over_time = np.cumsum(returns[sample_indices, :], axis=1)

    # Plot non-crisis paths in gray
    for idx in sample_indices:
        if not crisis_mask[idx]:
            ax1.plot(range(horizon), np.cumsum(returns[idx, :]),
                    color='gray', alpha=0.1, linewidth=0.5)

    # Plot crisis paths in red
    crisis_sample = [idx for idx in sample_indices if crisis_mask[idx]]
    for idx in crisis_sample:
        ax1.plot(range(horizon), np.cumsum(returns[idx, :]),
                color='red', alpha=0.3, linewidth=0.8)

    # Add crisis threshold line
    ax1.axhline(y=crisis_threshold, color='darkred', linestyle='--',
                linewidth=2, label=f'Crisis Threshold ({crisis_threshold_pct:.1f}% decline)')

    # Add zero line
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    ax1.set_xlabel('Time Horizon (weeks)', fontsize=11)
    ax1.set_ylabel('Cumulative Log Return', fontsize=11)
    ax1.set_title(f'Simulated Market Paths: {benchmark_name}\n'
                  f'{n_plot} sample paths (red = crisis, gray = non-crisis)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Distribution of final returns
    ax2 = fig.add_subplot(gs[0, 1])

    # Histogram
    ax2.hist(cumulative_returns[~crisis_mask], bins=50,
            color='gray', alpha=0.6, label='Non-crisis', density=True)
    ax2.hist(cumulative_returns[crisis_mask], bins=50,
            color='red', alpha=0.6, label='Crisis', density=True)

    # Add threshold line
    ax2.axvline(x=crisis_threshold, color='darkred', linestyle='--',
                linewidth=2, label='Threshold')

    ax2.set_xlabel('Cumulative Return', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Distribution of Final Returns', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Summary statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    # Calculate statistics
    mean_return = cumulative_returns.mean()
    std_return = cumulative_returns.std()
    median_return = np.median(cumulative_returns)
    percentile_5 = np.percentile(cumulative_returns, 5)
    percentile_95 = np.percentile(cumulative_returns, 95)

    crisis_mean = cumulative_returns[crisis_mask].mean()
    crisis_std = cumulative_returns[crisis_mask].std()

    stats_text = f"""
    SIMULATION STATISTICS
    {'─' * 40}
    Total simulations: {n_simulations:,}
    Horizon: {horizon} weeks

    ALL SCENARIOS
    Mean return: {mean_return:.4f} ({(np.exp(mean_return)-1)*100:.2f}%)
    Std deviation: {std_return:.4f}
    Median return: {median_return:.4f}
    5th percentile: {percentile_5:.4f}
    95th percentile: {percentile_95:.4f}

    CRISIS SCENARIOS (< {crisis_threshold_pct:.1f}%)
    Number: {n_crisis:,} ({n_crisis/n_simulations*100:.2f}%)
    Mean return: {crisis_mean:.4f} ({(np.exp(crisis_mean)-1)*100:.2f}%)
    Std deviation: {crisis_std:.4f}
    """

    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Subplot 4: Volatility path (average)
    ax4 = fig.add_subplot(gs[2, :])

    avg_volatility = market_paths["conditional_volatilities"].mean(axis=0)
    percentile_5_vol = np.percentile(market_paths["conditional_volatilities"], 5, axis=0)
    percentile_95_vol = np.percentile(market_paths["conditional_volatilities"], 95, axis=0)

    ax4.plot(range(horizon), avg_volatility, color='blue', linewidth=2, label='Mean volatility')
    ax4.fill_between(range(horizon), percentile_5_vol, percentile_95_vol,
                     color='blue', alpha=0.2, label='5th-95th percentile')

    ax4.set_xlabel('Time Horizon (weeks)', fontsize=11)
    ax4.set_ylabel('Conditional Volatility', fontsize=11)
    ax4.set_title('Evolution of Market Volatility (GARCH dynamics)',
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Market Crisis Scenario Simulation: {benchmark_name}\n'
                f'Common Paths for Systemic Risk Measurement',
                fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Chart saved: {save_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("GCC SRISK: BIVARIATE GARCH-DCC & LRMES ESTIMATION")
    print("=" * 60)
    print("\nMethodology: Each bank is modeled in a bivariate system with")
    print("the market benchmark (2x2 correlation matrix per srisks.md)")
    print("=" * 60)

    try:
        # 1. Load data
        bank_returns, benchmark_returns, universe = load_processed_data()

        # 2. Select primary benchmark
        # Use the first benchmark (typically broad market index)
        benchmark_name = benchmark_returns.columns[0]
        print(f"\nPrimary benchmark: {benchmark_name}")

        # 3. Calibrate crisis threshold from historical data
        print("\n" + "=" * 60)
        print("CALIBRATING CRISIS THRESHOLD FROM HISTORICAL DATA")
        print("=" * 60)

        # Calculate rolling 6-month returns for benchmark
        # Using ~6 months = 22 weeks × 5 days/week ≈ 110-130 trading days
        rolling_window = min(130, CONFIG.CRISIS_HORIZON_WEEKS * 5)  # Approximately 6 months in trading days

        benchmark_series = benchmark_returns[benchmark_name].dropna()

        print(f"\nBenchmark series: {len(benchmark_series)} observations")
        print(f"Rolling window: {rolling_window} days")

        # Check if we have enough data
        if len(benchmark_series) < rolling_window:
            print(f"\n⚠ WARNING: Insufficient data for crisis calibration")
            print(f"  Need at least {rolling_window} observations, have {len(benchmark_series)}")
            print(f"  Skipping historical calibration - using configured threshold: {CONFIG.CRISIS_THRESHOLD*100:.1f}%")
        else:
            # Calculate cumulative returns over rolling window
            rolling_returns = benchmark_series.rolling(window=rolling_window).apply(
                lambda x: x.sum(), raw=True
            ).dropna()

            if len(rolling_returns) == 0:
                print(f"\n⚠ WARNING: Rolling returns calculation produced no valid observations")
                print(f"  Skipping historical calibration - using configured threshold: {CONFIG.CRISIS_THRESHOLD*100:.1f}%")
            else:
                # Calculate percentiles
                percentile_5 = np.percentile(rolling_returns, 5)
                percentile_1 = np.percentile(rolling_returns, 1)
                percentile_10 = np.percentile(rolling_returns, 10)
                min_return = rolling_returns.min()
                mean_return = rolling_returns.mean()

                print(f"\nHistorical {rolling_window}-day rolling returns for {benchmark_name}:")
                print(f"  Observations: {len(rolling_returns)}")
                print(f"  Mean: {mean_return:.4f} ({(np.exp(mean_return)-1)*100:.2f}%)")
                print(f"  Minimum: {min_return:.4f} ({(np.exp(min_return)-1)*100:.2f}%)")
                print(f"  1st percentile: {percentile_1:.4f} ({(np.exp(percentile_1)-1)*100:.2f}%)")
                print(f"  5th percentile: {percentile_5:.4f} ({(np.exp(percentile_5)-1)*100:.2f}%)")
                print(f"  10th percentile: {percentile_10:.4f} ({(np.exp(percentile_10)-1)*100:.2f}%)")

                # Compare with configured threshold
                configured_threshold_log = np.log(1 + CONFIG.CRISIS_THRESHOLD)
                print(f"\nConfigured crisis threshold: {CONFIG.CRISIS_THRESHOLD*100:.1f}% = {configured_threshold_log:.4f} (log)")

                # Recommendation
                if configured_threshold_log < percentile_5:
                    print(f"\n⚠ WARNING: Configured threshold ({CONFIG.CRISIS_THRESHOLD*100:.1f}%) is MORE extreme than 5th percentile")
                    print(f"  This may result in too few crisis scenarios in simulation.")
                    print(f"  Consider using 5th percentile: {(np.exp(percentile_5)-1)*100:.1f}%")
                elif configured_threshold_log > percentile_10:
                    print(f"\n⚠ WARNING: Configured threshold ({CONFIG.CRISIS_THRESHOLD*100:.1f}%) is LESS extreme than 10th percentile")
                    print(f"  This may result in too many crisis scenarios.")
                else:
                    print(f"\n✓ Configured threshold is within reasonable range (between 5th and 10th percentile)")

        # 4. Estimate benchmark GARCH once (reused for all banks)
        benchmark_garch = estimate_benchmark_garch(benchmark_returns, benchmark_name)

        # 4. Simulate common market paths ONCE (all banks evaluated on same scenarios)
        print("\n" + "=" * 60)
        print("SIMULATING COMMON MARKET CRISIS SCENARIOS")
        print("=" * 60)
        print(f"Horizon: {CONFIG.CRISIS_HORIZON_WEEKS} weeks ({CONFIG.CRISIS_HORIZON_WEEKS * 5} trading days)")
        print(f"Crisis threshold: {CONFIG.CRISIS_THRESHOLD*100:.0f}% market decline")
        print(f"Number of simulations: {CONFIG.N_SIMULATIONS:,}")
        print("\nThis ensures all banks are evaluated under THE SAME market shocks")
        print("(critical for systemic risk measurement)")

        market_paths = simulate_market_paths(
            market_garch_result=benchmark_garch,
            horizon=CONFIG.CRISIS_HORIZON_WEEKS,
            n_simulations=CONFIG.N_SIMULATIONS,
            random_seed=CONFIG.RANDOM_SEED
        )

        # Identify crisis scenarios
        # Convert percentage threshold to log return threshold
        crisis_threshold_log = np.log(1 + CONFIG.CRISIS_THRESHOLD)
        crisis_mask = market_paths["cumulative_returns"] < crisis_threshold_log
        n_market_crisis = crisis_mask.sum()

        print(f"\n✓ Market paths simulated")
        print(f"  Crisis threshold: {CONFIG.CRISIS_THRESHOLD*100:.0f}% decline = {crisis_threshold_log:.4f} log return")
        print(f"  Crisis scenarios: {n_market_crisis:,} ({n_market_crisis/CONFIG.N_SIMULATIONS*100:.2f}%)")
        print(f"  Market avg return: {market_paths['cumulative_returns'].mean():.4f}")
        print(f"  Market avg return in crisis: {market_paths['cumulative_returns'][crisis_mask].mean():.4f}")

        # Visualize and save market paths
        print("\nGenerating market paths visualization...")
        chart_path = RESULTS_DATA_DIR / "market_crisis_scenarios.png"
        visualize_market_paths(
            market_paths=market_paths,
            crisis_threshold=crisis_threshold_log,  # Use log return threshold
            benchmark_name=benchmark_name,
            save_path=str(chart_path),
            n_sample_paths=500
        )

        # 5. Process each bank using COMMON market paths
        print("\n" + "=" * 60)
        print(f"PROCESSING {len(bank_returns.columns)} BANKS (CONDITIONAL ON COMMON MARKET)")
        print("=" * 60)
        print()

        results_list = []
        for i, bank_ric in enumerate(bank_returns.columns, 1):
            print(f"[{i}/{len(bank_returns.columns)}] {bank_ric}", end=" ")

            bank_series = bank_returns[bank_ric]
            result = process_bank_with_common_market(
                bank_ric=bank_ric,
                bank_series=bank_series,
                benchmark_garch_result=benchmark_garch,
                market_paths=market_paths,  # SAME market paths for all banks
                benchmark_name=benchmark_name,
                verbose=False  # Set to True for detailed output
            )

            if result is not None:
                results_list.append(result)
                print(f"✓ LRMES={result['lrmes']:.4f}")
            else:
                print("✗ FAILED")

        # 5. Compile results
        if not results_list:
            raise ValueError("No banks successfully processed!")

        results_df = pd.DataFrame(results_list)

        # Save main results
        metadata = {
            "description": "LRMES estimates from bivariate GARCH-DCC with COMMON market paths",
            "methodology": "All banks evaluated conditional on same market crisis scenarios",
            "benchmark": benchmark_name,
            "horizon_weeks": CONFIG.CRISIS_HORIZON_WEEKS,
            "crisis_threshold": CONFIG.CRISIS_THRESHOLD,
            "n_simulations": CONFIG.N_SIMULATIONS,
            "n_banks": len(results_df),
            "n_market_crisis_scenarios": int(n_market_crisis),
            "avg_crisis_scenarios": int(results_df['n_crisis_scenarios'].mean())
        }
        save_dataframe(results_df, "lrmes_estimates", RESULTS_DATA_DIR, metadata)

        # Save summary statistics
        summary = {
            "avg_lrmes": results_df['lrmes'].mean(),
            "median_lrmes": results_df['lrmes'].median(),
            "min_lrmes": results_df['lrmes'].min(),
            "max_lrmes": results_df['lrmes'].max(),
            "avg_correlation": results_df['avg_correlation'].mean(),
            "avg_garch_persistence": results_df['garch_persistence'].mean(),
            "avg_dcc_persistence": results_df['dcc_persistence'].mean(),
            "convergence_rate": results_df['convergence'].sum() / len(results_df)
        }
        summary_df = pd.DataFrame([summary])
        save_dataframe(summary_df, "lrmes_summary_stats", RESULTS_DATA_DIR, {"description": "LRMES summary statistics"})

        # Summary
        print("\n" + "=" * 60)
        print("✓ BIVARIATE GARCH-DCC & LRMES ESTIMATION COMPLETED")
        print("=" * 60)
        print(f"\nSuccessfully processed: {len(results_df)}/{len(bank_returns.columns)} banks")
        print(f"Average LRMES: {summary['avg_lrmes']:.4f}")
        print(f"Average bank-market correlation: {summary['avg_correlation']:.4f}")
        print(f"Common market crisis scenarios: {n_market_crisis:,} ({n_market_crisis/CONFIG.N_SIMULATIONS*100:.2f}%)")

        print("\n" + "=" * 60)
        print("METHODOLOGY HIGHLIGHT")
        print("=" * 60)
        print(f"✓ All banks evaluated under THE SAME {n_market_crisis:,} market crisis scenarios")
        print("✓ Bank returns simulated conditional on common market paths")
        print("✓ Uses z_i,t | z_m,t ~ N(ρ_t × z_m,t, 1 - ρ²_t)")
        print("✓ Ensures proper systemic risk measurement (not independent crises)")

        print("\n" + "=" * 60)
        print("Top 10 banks by LRMES (highest systemic risk):")
        print("=" * 60)
        top10 = results_df.sort_values("lrmes", ascending=False).head(10)
        print(top10[['bank_ric', 'lrmes', 'avg_correlation', 'avg_bank_loss_in_crisis']].to_string(index=False))

        print(f"\nResults saved to: {RESULTS_DATA_DIR}")
        print("  - lrmes_estimates.parquet (full results)")
        print("  - lrmes_summary_stats.parquet (aggregate statistics)")
        print("  - market_crisis_scenarios.png (visualization of common market paths)")
        print("\nNext step: Run scripts/04_calculate_srisk.py")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
