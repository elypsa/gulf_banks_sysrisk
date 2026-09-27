#!/usr/bin/env python3
"""Interactive dashboard for SRISK analysis.

This dashboard provides interactive visualization of SRISK results:
- Time series plots for individual banks
- Comparison of multiple banks
- Summary statistics

Requirements:
    - Run scripts/04_calculate_srisk.py first to generate srisk_timeseries.parquet
    - Install streamlit: uv add streamlit plotly

Usage:
    uv run streamlit run scripts/dashboard_srisk.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.utils.config import RESULTS_DATA_DIR, CONFIG
from src.data import load_dataframe

# Page configuration
st.set_page_config(
    page_title="GCC SRISK Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_srisk_data():
    """Load SRISK time series data with caching."""
    try:
        srisk = pd.read_parquet(RESULTS_DATA_DIR / "srisk_timeseries.parquet")
        # Load bank universe for country mapping
        bank_universe = pd.read_parquet(Path(__file__).parent.parent / "data/processed/banks_universe_clean.parquet")
        return srisk, bank_universe
    except FileNotFoundError as e:
        st.error(f"❌ Error: SRISK data not found. Please run scripts/04_calculate_srisk.py first.")
        st.stop()

def format_billions(value):
    """Format large numbers as billions."""
    return f"${value / 1e9:.2f}B"

def format_millions(value):
    """Format large numbers as millions."""
    return f"${value / 1e6:.1f}M"

def create_bank_country_map(bank_universe):
    """Create mapping of bank RIC to country."""
    country_map = {}
    country_code_map = {
        "0#.TRXFLDAEPBANK": "UAE",
        "0#.TRXFLDSAPBANK": "SAU",
        "0#.TRXFLDQAPBANK": "QAT",
        "0#.TRXFLDKWPBANK": "KWT",
        "0#.TRXFLDOMPBANK": "OMN",
        "0#.TRXFLDBHPBANK": "BAH"
    }

    for _, row in bank_universe.iterrows():
        bank_ric = row['bank_ric']
        country_chain = row['country_chain']
        country_map[bank_ric] = country_code_map.get(country_chain, country_chain)

    return country_map

def plot_bank_srisk_timeseries(srisk, selected_banks, country_map):
    """Create interactive time series plot for selected banks."""
    fig = go.Figure()

    for bank in selected_banks:
        if bank in srisk.columns:
            country = country_map.get(bank, "Unknown")
            fig.add_trace(go.Scatter(
                x=srisk.index,
                y=srisk[bank] / 1e9,  # Convert to billions
                mode='lines',
                name=f"{bank} ({country})",
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x}<br>' +
                              'SRISK: $%{y:.2f}B<br>' +
                              '<extra></extra>'
            ))

    fig.update_layout(
        title="SRISK Time Series",
        xaxis_title="Date",
        yaxis_title="SRISK ($ Billions)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig

def plot_bank_comparison(srisk, selected_banks, country_map):
    """Create bar chart comparing banks' latest SRISK."""
    latest_srisk = srisk[selected_banks].iloc[-1]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Bank': selected_banks,
        'SRISK': latest_srisk.values / 1e9,
        'Country': [country_map.get(bank, "Unknown") for bank in selected_banks]
    }).sort_values('SRISK', ascending=False)

    fig = px.bar(
        plot_data,
        x='Bank',
        y='SRISK',
        color='Country',
        title=f"Latest SRISK Comparison ({srisk.index[-1].strftime('%Y-%m-%d')})",
        labels={'SRISK': 'SRISK ($ Billions)'},
        template='plotly_white',
        height=400
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=True
    )

    return fig

def main():
    """Main dashboard application."""

    # Header
    st.title("📊 GCC Bank SRISK Dashboard")
    st.markdown("### Time-Varying Systemic Risk Analysis")

    # Load data
    with st.spinner("Loading SRISK data..."):
        srisk, bank_universe = load_srisk_data()

    country_map = create_bank_country_map(bank_universe)

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    # Display key parameters
    st.sidebar.markdown("### Model Parameters")
    st.sidebar.metric("Capital Ratio", f"{CONFIG.CAPITAL_RATIO_BASEL*100:.1f}%")
    st.sidebar.metric("Crisis Threshold", f"{CONFIG.CRISIS_THRESHOLD*100:.1f}%")
    st.sidebar.metric("Horizon", f"{CONFIG.CRISIS_HORIZON_WEEKS} weeks")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Info")
    st.sidebar.metric("Total Banks", len(srisk.columns))
    st.sidebar.metric("Date Range", f"{srisk.index[0].strftime('%Y-%m-%d')} to {srisk.index[-1].strftime('%Y-%m-%d')}")
    st.sidebar.metric("Observations", len(srisk))

    # Main content
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Bank Selection")

    # Country filter
    all_countries = sorted(set(country_map.values()))
    selected_countries = st.sidebar.multiselect(
        "Filter by Country",
        options=all_countries,
        default=all_countries
    )

    # Filter banks by country
    available_banks = [bank for bank, country in country_map.items()
                      if country in selected_countries and bank in srisk.columns]
    available_banks = sorted(available_banks)

    # Bank selection
    selection_mode = st.sidebar.radio(
        "Selection Mode",
        options=["Single Bank", "Multiple Banks", "Top N Banks"]
    )

    if selection_mode == "Single Bank":
        selected_bank = st.sidebar.selectbox(
            "Select Bank",
            options=available_banks,
            format_func=lambda x: f"{x} ({country_map.get(x, 'Unknown')})"
        )
        selected_banks = [selected_bank]

    elif selection_mode == "Multiple Banks":
        # Pre-select top 5 banks by latest SRISK
        latest_srisk = srisk[available_banks].iloc[-1].sort_values(ascending=False)
        default_banks = latest_srisk.head(5).index.tolist()

        selected_banks = st.sidebar.multiselect(
            "Select Banks",
            options=available_banks,
            default=default_banks,
            format_func=lambda x: f"{x} ({country_map.get(x, 'Unknown')})"
        )

    else:  # Top N Banks
        top_n = st.sidebar.slider("Number of Top Banks", min_value=1, max_value=20, value=5)
        latest_srisk = srisk[available_banks].iloc[-1].sort_values(ascending=False)
        selected_banks = latest_srisk.head(top_n).index.tolist()

        st.sidebar.markdown(f"**Selected Top {top_n} Banks:**")
        for i, bank in enumerate(selected_banks, 1):
            st.sidebar.text(f"{i}. {bank} ({country_map.get(bank, 'Unknown')})")

    # Main visualization area
    if not selected_banks:
        st.warning("⚠️ Please select at least one bank from the sidebar.")
        return

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)

    # Get latest non-NaN values
    latest_values = srisk[selected_banks].iloc[-1]

    with col1:
        total_latest = latest_values.sum()
        if pd.isna(total_latest):
            st.metric("Total SRISK (Latest)", "N/A")
        else:
            st.metric("Total SRISK (Latest)", format_billions(total_latest))

    with col2:
        avg_latest = latest_values.mean()
        if pd.isna(avg_latest):
            st.metric("Average SRISK (Latest)", "N/A")
        else:
            st.metric("Average SRISK (Latest)", format_billions(avg_latest))

    with col3:
        # Filter out NaN values before finding max
        valid_latest = latest_values.dropna()
        if len(valid_latest) > 0:
            max_bank = valid_latest.idxmax()
            max_value = valid_latest.max()
            st.metric(f"Highest: {max_bank}", format_billions(max_value))
        else:
            st.metric("Highest", "N/A")

    with col4:
        # Calculate change from first to last observation
        first_total = srisk[selected_banks].iloc[0].sum()
        last_total = srisk[selected_banks].iloc[-1].sum()

        if pd.isna(first_total) or pd.isna(last_total):
            st.metric("Total Change", "N/A")
        else:
            change = last_total - first_total
            change_pct = (change / first_total * 100) if first_total != 0 else 0
            st.metric("Total Change", format_billions(change), f"{change_pct:+.1f}%")

    # Time series plot
    st.markdown("---")
    st.subheader("📈 SRISK Time Series")
    fig_ts = plot_bank_srisk_timeseries(srisk, selected_banks, country_map)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Comparison bar chart
    if len(selected_banks) > 1:
        st.markdown("---")
        st.subheader("📊 Latest SRISK Comparison")
        fig_bar = plot_bank_comparison(srisk, selected_banks, country_map)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed statistics table
    st.markdown("---")
    st.subheader("📋 Detailed Statistics")

    stats_data = []
    for bank in selected_banks:
        if bank in srisk.columns:
            bank_data = srisk[bank]
            stats_data.append({
                'Bank': bank,
                'Country': country_map.get(bank, 'Unknown'),
                'Latest SRISK': bank_data.iloc[-1],
                'Mean SRISK': bank_data.mean(),
                'Max SRISK': bank_data.max(),
                'Min SRISK': bank_data.min(),
                'Std Dev': bank_data.std(),
                'Latest Date': srisk.index[-1].strftime('%Y-%m-%d')
            })

    stats_df = pd.DataFrame(stats_data)

    # Format currency columns, handling NaN values
    currency_cols = ['Latest SRISK', 'Mean SRISK', 'Max SRISK', 'Min SRISK', 'Std Dev']
    for col in currency_cols:
        stats_df[col] = stats_df[col].apply(lambda x: format_millions(x) if pd.notna(x) else 'N/A')

    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True
    )

    # Download data
    st.markdown("---")
    st.subheader("💾 Download Data")

    csv_data = srisk[selected_banks].copy()
    csv_data.index.name = 'Date'

    st.download_button(
        label="📥 Download Selected Banks Data (CSV)",
        data=csv_data.to_csv().encode('utf-8'),
        file_name=f"srisk_selected_banks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
