# SRISK Dashboard Guide

## Overview
The SRISK Dashboard provides an interactive visualization tool for analyzing time-varying systemic risk across GCC banks.

## Features

### 📊 Interactive Visualizations
- **Time Series Plots**: View SRISK evolution over time for selected banks
- **Comparison Charts**: Compare latest SRISK values across multiple banks
- **Country Filtering**: Filter banks by country (UAE, SAU, QAT, KWT, OMN, BAH)

### 🔍 Selection Modes
1. **Single Bank**: Analyze one bank in detail
2. **Multiple Banks**: Compare multiple banks simultaneously
3. **Top N Banks**: Automatically select top N banks by latest SRISK

### 📈 Key Metrics Displayed
- Total SRISK (Latest)
- Average SRISK (Latest)
- Highest SRISK Bank
- Total Change (from first to last observation)
- Detailed statistics table with mean, max, min, and standard deviation

### 💾 Data Export
- Download selected banks' data as CSV for further analysis

## Prerequisites

Before running the dashboard, ensure you have:

1. **Generated SRISK data**:
   ```bash
   uv run scripts/02_clean_and_align_data.py
   uv run scripts/03b_estimate_garch_dcc_rolling.py
   uv run scripts/04_calculate_srisk.py
   ```

2. **Installed required packages** (automatically installed):
   - `streamlit`
   - `plotly`

## Running the Dashboard

### Command Line
```bash
uv run streamlit run scripts/dashboard_srisk.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Alternative Port
If port 8501 is already in use:
```bash
uv run streamlit run scripts/dashboard_srisk.py --server.port 8502
```

## Using the Dashboard

### 1. Sidebar Configuration
The left sidebar contains all controls:

**Model Parameters** (read-only)
- Capital Ratio
- Crisis Threshold
- Horizon

**Data Info** (read-only)
- Total Banks
- Date Range
- Number of Observations

**Bank Selection**
- Country filter (multi-select)
- Selection mode (radio buttons)
- Bank picker (varies by mode)

### 2. Main Display Area

**Summary Metrics** (top row)
- Four key metrics showing latest SRISK values

**Time Series Plot**
- Interactive plot with zoom, pan, and hover features
- Multiple banks shown as different colored lines
- Hover to see exact values at any date

**Comparison Chart** (when multiple banks selected)
- Bar chart comparing latest SRISK values
- Color-coded by country

**Statistics Table**
- Detailed statistics for each selected bank
- Values formatted in millions/billions for readability

**Download Button**
- Export selected banks' time series data as CSV

### 3. Interactive Features

**Plotly Controls** (appear on hover)
- 🔍 **Zoom**: Click and drag to zoom into specific time periods
- 🏠 **Home**: Reset to original view
- 📷 **Camera**: Download plot as PNG
- 🔄 **Pan**: Move around the zoomed plot
- ☐ **Box Select**: Select data points
- 📊 **Compare**: Toggle comparison mode

**Legend**
- Click legend items to show/hide specific banks
- Double-click to isolate a single bank

## Tips & Best Practices

### Performance
- **Start Small**: Begin with single or few banks, then expand
- **Use Country Filter**: Filter by country to reduce the bank list
- **Top N Mode**: Useful for quick overview of most systemically important banks

### Analysis Workflows

**1. Individual Bank Deep Dive**
- Select "Single Bank" mode
- Choose bank of interest
- Examine time series for trends, spikes, and patterns

**2. Country Comparison**
- Use country filter to select one country
- Select "Top N Banks" (e.g., N=5)
- Compare systemic importance within country

**3. Regional Overview**
- Select multiple countries
- Use "Top N Banks" (e.g., N=10)
- Identify region-wide systemic risk leaders

**4. Custom Comparison**
- Select "Multiple Banks" mode
- Choose specific banks of interest
- Compare evolution and current levels

### Interpreting Results

**SRISK Values**
- **Positive Values**: Bank needs capital in crisis scenario
- **Higher Values**: Greater systemic importance/vulnerability
- **Zero/Negative**: Bank is sufficiently capitalized

**Time Trends**
- **Increasing SRISK**: Growing systemic risk contribution
- **Decreasing SRISK**: Improving capital position or reduced size/leverage
- **Spikes**: Often correspond to market stress periods

**Country Patterns**
- Compare banking systems across GCC countries
- Identify regional vs. country-specific trends

## Troubleshooting

### Dashboard Won't Start
```
Error: SRISK data not found
```
**Solution**: Run the SRISK calculation pipeline first:
```bash
uv run scripts/04_calculate_srisk.py
```

### No Banks Showing
**Solution**: Check country filter - ensure at least one country is selected

### Slow Performance
**Solution**:
- Reduce number of selected banks
- Use smaller date ranges (future feature)
- Close other browser tabs

### Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution**: Use alternative port:
```bash
uv run streamlit run scripts/dashboard_srisk.py --server.port 8502
```

## Configuration

### Customizing Model Parameters
To change SRISK calculation parameters, edit `src/utils/config.py`:
- `CAPITAL_RATIO_BASEL`: Prudential capital ratio (default: 8%)
- `CRISIS_THRESHOLD`: Market decline threshold (default: -15%)
- `CRISIS_HORIZON_WEEKS`: Crisis horizon (default: 22 weeks)

After changing parameters, re-run:
```bash
uv run scripts/03b_estimate_garch_dcc_rolling.py
uv run scripts/04_calculate_srisk.py
```

## Data Files

The dashboard reads from:
- `data/results/srisk_timeseries.parquet` - Main SRISK time series
- `data/processed/banks_universe_clean.parquet` - Bank metadata

## Future Enhancements

Potential additions:
- [ ] Date range selector
- [ ] Multiple capital ratio scenarios
- [ ] System-wide aggregates visualization
- [ ] Country-level aggregation view
- [ ] Export to PDF report
- [ ] Statistical tests and correlations
- [ ] Crisis period highlighting
- [ ] Decomposition analysis integration

## Support

For issues or questions:
1. Check that all prerequisite scripts have been run
2. Verify data files exist in `data/results/`
3. Check console for error messages
4. Review Streamlit logs in terminal

## Quick Reference

```bash
# Complete pipeline
uv run scripts/02_clean_and_align_data.py
uv run scripts/03b_estimate_garch_dcc_rolling.py
uv run scripts/04_calculate_srisk.py

# Launch dashboard
uv run streamlit run scripts/dashboard_srisk.py

# Stop dashboard
# Press Ctrl+C in terminal
```
