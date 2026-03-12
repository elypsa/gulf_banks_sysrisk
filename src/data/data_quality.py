"""Data quality visualization and reporting.

This module provides functions for visualizing and reporting data quality issues,
including missing value patterns, coverage statistics, and data completeness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List


def generate_missing_data_report(
    aligned_data: pd.DataFrame,
    bank_universe: pd.DataFrame,
    output_dir: Path,
    quality_report: pd.DataFrame = None,
    show_plots: bool = False
) -> pd.DataFrame:
    """Generate comprehensive missing data report with visualizations.

    Creates missing data matrix plots similar to R's naniar::vis_miss(),
    showing patterns of missingness for each bank across time.
    Organizes plots into subdirectories based on quality test results.

    Args:
        aligned_data: Aligned DataFrame with all data.
        bank_universe: Bank universe DataFrame with bank_ric column.
        output_dir: Directory to save visualizations.
        quality_report: Optional DataFrame with quality test results (from filter_banks_by_data_quality).
        show_plots: Whether to display plots (default False for scripts).

    Returns:
        DataFrame with missing data statistics per bank.

    Example:
        >>> stats = generate_missing_data_report(aligned, universe, Path("output/"), quality_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for passed/dropped banks
    passed_dir = output_dir / "passed"
    dropped_dir = output_dir / "dropped"
    passed_dir.mkdir(parents=True, exist_ok=True)
    dropped_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING DATA QUALITY VISUALIZATIONS")
    print("=" * 60)

    # Create quality lookup dict for fast access
    quality_lookup = {}
    if quality_report is not None:
        quality_lookup = dict(zip(quality_report["bank_ric"], quality_report["passes_quality"]))

    # Extract bank RICs from universe
    bank_rics = bank_universe["bank_ric"].tolist()

    # Collect statistics for each bank
    stats_records = []

    for bank_ric in bank_rics:
        # Find all columns for this bank
        bank_cols = [col for col in aligned_data.columns if bank_ric in str(col)]

        if not bank_cols:
            continue

        # Extract data for this bank
        bank_data = aligned_data[bank_cols].copy()

        # Calculate statistics
        total_cells = bank_data.size
        missing_cells = bank_data.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        # Per-column missing stats
        col_missing = bank_data.isna().sum()
        col_missing_pct = (col_missing / len(bank_data)) * 100

        stats_records.append({
            "bank_ric": bank_ric,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_pct": missing_pct,
            "num_columns": len(bank_cols),
            "num_rows": len(bank_data),
            "complete_rows": bank_data.notna().all(axis=1).sum(),
            "complete_rows_pct": (bank_data.notna().all(axis=1).sum() / len(bank_data)) * 100
        })

        # Determine if bank passed quality check
        passed = quality_lookup.get(bank_ric, None)

        # Choose output directory
        if passed is True:
            plot_dir = passed_dir
        elif passed is False:
            plot_dir = dropped_dir
        else:
            plot_dir = output_dir  # No quality info, save to root

        # Generate missing data matrix plot for this bank
        _plot_missing_matrix(
            bank_data,
            bank_ric,
            plot_dir,
            passed,
            show_plots
        )

    stats_df = pd.DataFrame(stats_records)
    stats_df = stats_df.sort_values("missing_pct", ascending=False)

    # Save summary statistics
    stats_file = output_dir / "missing_data_summary.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✓ Saved missing data summary to: {stats_file}")

    # Generate overview heatmap
    _plot_overview_heatmap(aligned_data, bank_universe, output_dir, show_plots)

    # Print summary
    print("\nMissing Data Summary:")
    print(f"  - Total banks analyzed: {len(stats_df)}")
    print(f"  - Mean missing %: {stats_df['missing_pct'].mean():.2f}%")
    print(f"  - Median missing %: {stats_df['missing_pct'].median():.2f}%")
    print(f"\nTop 5 banks by missing data:")
    for _, row in stats_df.head(5).iterrows():
        print(f"  - {row['bank_ric']}: {row['missing_pct']:.2f}% missing")

    print(f"\n✓ All visualizations saved to: {output_dir}")

    return stats_df


def _plot_missing_matrix(
    bank_data: pd.DataFrame,
    bank_ric: str,
    output_dir: Path,
    passed_quality: bool = None,
    show_plots: bool = False
):
    """Generate missing data matrix plot for a single bank.

    Creates a heatmap showing missing (white) vs present (black) data
    across time and variables, similar to naniar::vis_miss().

    Args:
        bank_data: DataFrame with data for one bank.
        bank_ric: Bank RIC identifier.
        output_dir: Directory to save plot.
        passed_quality: Whether bank passed quality test (None if not tested).
        show_plots: Whether to display plot.
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[4, 1])

    # Main missing data matrix
    ax1 = axes[0]

    # Create binary matrix (1 = present, 0 = missing)
    missing_matrix = (~bank_data.isna()).astype(int)

    # Transpose so columns are on y-axis, time on x-axis
    missing_matrix_T = missing_matrix.T

    # Plot heatmap
    sns.heatmap(
        missing_matrix_T,
        cmap=["white", "black"],
        cbar=False,
        ax=ax1,
        yticklabels=True,
        xticklabels=False
    )

    # Simplify y-axis labels (remove prefixes)
    yticks = [label.get_text() for label in ax1.get_yticklabels()]
    simplified_labels = [_simplify_column_name(label) for label in yticks]
    ax1.set_yticklabels(simplified_labels, rotation=0, fontsize=8)

    ax1.set_xlabel("Time (chronological order)", fontsize=10)
    ax1.set_ylabel("Variables", fontsize=10)

    # Add quality status to title
    title = f"Missing Data Pattern: {bank_ric}\n(Black = Present, White = Missing)"
    if passed_quality is True:
        title += " ✓ PASSED"
        title_color = 'darkgreen'
    elif passed_quality is False:
        title += " ✗ DROPPED"
        title_color = 'darkred'
    else:
        title_color = 'black'

    ax1.set_title(title, fontsize=12, fontweight="bold", color=title_color)

    # Bottom panel: Overall completeness by time
    ax2 = axes[1]

    # Calculate % complete for each time point
    completeness_by_time = (~bank_data.isna()).mean(axis=1) * 100

    ax2.plot(completeness_by_time.values, color='steelblue', linewidth=1.5)
    ax2.fill_between(range(len(completeness_by_time)), completeness_by_time.values,
                     alpha=0.3, color='steelblue')
    ax2.set_ylabel("% Complete", fontsize=10)
    ax2.set_xlabel("Time (chronological order)", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # Add statistics text box with quality status
    missing_pct = bank_data.isna().sum().sum() / bank_data.size * 100
    text_str = f"Overall Missing: {missing_pct:.2f}%\nRows: {len(bank_data)}\nCols: {len(bank_data.columns)}"

    if passed_quality is True:
        text_str += "\n\nStatus: PASSED ✓"
        box_color = 'lightgreen'
    elif passed_quality is False:
        text_str += "\n\nStatus: DROPPED ✗"
        box_color = 'lightcoral'
    else:
        box_color = 'wheat'

    ax2.text(0.02, 0.95, text_str, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.7))

    plt.tight_layout()

    # Save plot
    safe_ric = bank_ric.replace(".", "_").replace("/", "_")
    output_file = output_dir / f"missing_pattern_{safe_ric}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def _plot_overview_heatmap(
    aligned_data: pd.DataFrame,
    bank_universe: pd.DataFrame,
    output_dir: Path,
    show_plots: bool = False
):
    """Generate overview heatmap showing missing % for all banks x variables.

    Args:
        aligned_data: Aligned DataFrame with all data.
        bank_universe: Bank universe DataFrame.
        output_dir: Directory to save plot.
        show_plots: Whether to display plot.
    """
    bank_rics = bank_universe["bank_ric"].tolist()

    # Group columns by type (ret, price, mktcap, fund, bench)
    var_types = ["ret_", "price_", "mktcap_", "fund_"]

    # Calculate missing % for each bank x variable type
    missing_matrix = []

    for bank_ric in bank_rics:
        row_data = {"bank_ric": bank_ric}

        for var_type in var_types:
            # Find columns for this bank and variable type
            cols = [col for col in aligned_data.columns
                   if bank_ric in str(col) and str(col).startswith(var_type)]

            if cols:
                missing_pct = aligned_data[cols].isna().sum().sum() / aligned_data[cols].size * 100
            else:
                missing_pct = np.nan

            row_data[var_type.replace("_", "")] = missing_pct

        missing_matrix.append(row_data)

    df_missing = pd.DataFrame(missing_matrix)
    df_missing = df_missing.set_index("bank_ric")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, max(8, len(bank_rics) * 0.4)))

    sns.heatmap(
        df_missing,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Missing %"},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title("Missing Data Overview: Banks × Variable Types",
                fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Variable Type", fontsize=11)
    ax.set_ylabel("Bank RIC", fontsize=11)

    plt.tight_layout()

    # Save
    output_file = output_dir / "missing_overview_heatmap.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved overview heatmap: {output_file.name}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def _simplify_column_name(col_name: str) -> str:
    """Simplify column name for display.

    Removes common prefixes and shortens long names.

    Args:
        col_name: Full column name.

    Returns:
        Simplified name.
    """
    # Remove common prefixes
    for prefix in ["ret_", "price_", "mktcap_", "fund_", "bench_ret_"]:
        if col_name.startswith(prefix):
            col_name = col_name[len(prefix):]

    # Shorten long field names
    col_name = col_name.replace("TR.CompanyMarketCapitalization", "MktCap")
    col_name = col_name.replace("TR.PriceClose", "Price")
    col_name = col_name.replace("TR.TotalReturn", "TotRet")
    col_name = col_name.replace("TR.F.TotAssets", "Assets")
    col_name = col_name.replace("TR.F.ComEq", "Equity")
    col_name = col_name.replace("TR.F.TotLiab", "Liab")

    # Truncate if still too long
    if len(col_name) > 40:
        col_name = col_name[:37] + "..."

    return col_name


def print_data_quality_summary(
    aligned_data: pd.DataFrame,
    bank_universe: pd.DataFrame
):
    """Print text-based data quality summary to console.

    Args:
        aligned_data: Aligned DataFrame with all data.
        bank_universe: Bank universe DataFrame.
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)

    # Overall statistics
    total_cells = aligned_data.size
    missing_cells = aligned_data.isna().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100

    print(f"\nOverall Statistics:")
    print(f"  - Total cells: {total_cells:,}")
    print(f"  - Missing cells: {missing_cells:,}")
    print(f"  - Missing %: {missing_pct:.2f}%")
    print(f"  - Shape: {aligned_data.shape[0]:,} rows × {aligned_data.shape[1]} columns")

    # Per-bank summary
    bank_rics = bank_universe["bank_ric"].tolist()

    bank_stats = []
    for bank_ric in bank_rics:
        bank_cols = [col for col in aligned_data.columns if bank_ric in str(col)]
        if bank_cols:
            bank_missing = aligned_data[bank_cols].isna().sum().sum() / aligned_data[bank_cols].size * 100
            bank_stats.append((bank_ric, bank_missing, len(bank_cols)))

    bank_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\nPer-Bank Missing % (Top 10 worst):")
    for bank_ric, missing_pct, num_cols in bank_stats[:10]:
        print(f"  - {bank_ric:15s}: {missing_pct:6.2f}% ({num_cols} columns)")

    # Variable type summary
    print(f"\nMissing % by Variable Type:")
    for var_type in ["ret_", "price_", "mktcap_", "fund_", "bench_ret_"]:
        var_cols = [col for col in aligned_data.columns if str(col).startswith(var_type)]
        if var_cols:
            var_missing = aligned_data[var_cols].isna().sum().sum() / aligned_data[var_cols].size * 100
            print(f"  - {var_type:12s}: {var_missing:6.2f}%")