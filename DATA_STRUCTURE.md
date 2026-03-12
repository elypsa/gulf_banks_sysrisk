# GCC SRISK Data Structure Documentation

## Overview
This document describes the data structure for the GCC SRISK calculator based on the LSEG prototype analysis.

## Data Sources

### 1. Bank Universe
**File**: `data/raw/gcc_banks_universe.parquet`

**Structure**:
```
Columns: bank_ric, country_chain, country_code
Example:
  ENBD.DU, 0#.TRXFLDAEPBANK, AEP  (Emirates NBD - UAE)
  1180.SE, 0#.TRXFLDSAPBANK, SAP  (Arab National Bank - Saudi)
```

**Coverage**: ~30-40 banks across 6 GCC countries
- UAE: ~8 banks (AJBNK.DU, DISB.DU, ENBD.DU, etc.)
- Saudi Arabia: ~10 banks (1030.SE, 1180.SE, 1183.SE, etc.)
- Kuwait: ~8 banks (KIBK.KW, KPRO.KW, GBKK.KW, etc.)
- Qatar: ~4 banks
- Oman: ~3 banks
- Bahrain: ~5 banks

### 2. Quarterly Fundamentals
**File**: `data/raw/fundamentals_quarterly.parquet`

**Fields**:
- `TR.F.TotAssets` - Total Assets
- `TR.F.ComEqTot` - Common Equity Total (book equity)
- `TR.F.TotLiab` - Total Liabilities (used as Debt for SRISK)
- `TR.F.TotLiabEq` - Total Liabilities & Equity (balance check)

**Time Period**: 2010-Q1 to present (~60 quarters)
**Currency**: Native (AED, SAR, QAR, KWD, OMR, BHD)
**Frequency**: Quarterly (Q1, Q2, Q3, Q4)

**Usage in SRISK**:
- Total Liabilities → Debt (D_i,t) in SRISK formula
- Forward-filled to daily frequency with 45-day reporting lag

### 3. Daily Market Data
**File**: `data/raw/prices_daily.parquet`

**Fields** (Multi-index: RIC × Field):
- `TR.PriceClose` - Adjusted closing price
- `TR.TotalReturn` - Total return index (includes dividends)
- `TR.CompanyMarketCapitalization` - Market cap in native currency

**Time Period**: 2009-12-31 to present (~4,000 trading days, 16 years)
**Currency**: Native
**Frequency**: Daily

**Data Structure**:
```python
# Multi-index columns: (RIC, Field)
Date        | (ENBD.DU, TR.PriceClose) | (ENBD.DU, TR.CompanyMarketCapitalization) | ...
2025-12-31  | 27.85                    | 175917261346.05                            | ...
2026-01-05  | 29.20                    | 184444668987.60                            | ...
```

**Usage in SRISK**:
- PriceClose → Calculate log returns r_i,t for GARCH-DCC
- MarketCapitalization → Equity (W_i,t) in SRISK formula

### 4. Benchmark Indices
**File**: `data/raw/benchmarks_daily.parquet`

**Primary Benchmark**:
- `.GPDGC` - S&P GCC Composite Index (broad market)
  - Available from: 2009-12-31
  - Used for LRMES calculation (market decline threshold)

**Secondary Benchmarks** (Country Banking Sectors):
- `.TRXFLDGCPUBANK` - GCC Banking Index
- `.TRXFLDAEPBANK` - UAE Banking Index
- `.TRXFLDSAPBANK` - Saudi Banking Index
- `.TRXFLDQAPBANK` - Qatar Banking Index
- `.TRXFLDKWPBANK` - Kuwait Banking Index
- `.TRXFLDOMPBANK` - Oman Banking Index
- `.TRXFLDBHPBANK` - Bahrain Banking Index

**Time Period**: 2009-12-31 to present (daily)

**Usage in SRISK**:
- S&P GCC Composite → Market returns (R_m,t) for GARCH-DCC
- Banking indices → Calculate "Financial Sector Sensitivity" (alternative LRMES)

### 5. Holiday Calendar
**File**: `data/raw/holidays_calendar.parquet`

**Fields**:
- `date` - Holiday date
- `name` - Holiday name (e.g., "Eid al-Fitr", "UAE National Day")
- `calendars` - List of countries observing (["UAE", "SAU", "QAT", ...])

**Time Period**: 2016-01-01 to present (from LSEG API)

**Note**: Pre-2016 holidays are reconstructed using `hijri-converter` library:
- Islamic holidays: Eid al-Fitr, Eid al-Adha, Islamic New Year, Prophet's Birthday
- National holidays: Fixed Gregorian dates (UAE National Day, Saudi National Day, etc.)

**Usage**:
- Filter out non-trading days from return calculation
- Ensure GARCH estimation only uses actual trading days

### 6. USD FX Rates
**File**: `data/raw/fx_rates_daily.parquet`

**Currency Pairs**:
- `AED=` - USD/AED rate (UAE Dirham)
- `SAR=` - USD/SAR rate (Saudi Riyal)
- `QAR=` - USD/QAR rate (Qatari Riyal)
- `KWD=` - USD/KWD rate (Kuwaiti Dinar)
- `OMR=` - USD/OMR rate (Omani Rial)
- `BHD=` - USD/BHD rate (Bahraini Dinar)

**Format**: USD/CCY (how many local currency per 1 USD)
**Time Period**: 2009-12-31 to present (daily)

**Usage**:
- Convert market cap and debt to USD for system-wide aggregation
- Formula: USD_value = Local_value / FX_rate

---

## Processed Data

### 7. Aligned Daily Data
**File**: `data/processed/aligned_daily_data.parquet`

**Description**: Master dataset combining all sources at daily frequency

**Columns**:
- `fund_{RIC}_TotLiab` - Daily debt (forward-filled from quarterly)
- `fund_{RIC}_ComEqTot` - Daily book equity
- `mktcap_{RIC}` - Daily market cap (USD)
- `price_{RIC}` - Daily price
- `ret_{RIC}` - Daily log returns
- `bench_ret_.GPDGC` - Daily S&P GCC Composite return
- `bench_ret_.TRXFLD*` - Daily banking sector returns

**Date Range**: Common trading days across all banks
**Currency**: USD (after conversion)
**Frequency**: Daily (trading days only, holidays excluded)

**Creation Process**:
1. Forward-fill quarterly fundamentals with 45-day lag
2. Calculate log returns from prices
3. Convert market cap and debt to USD
4. Merge with benchmark returns
5. Filter to common trading days

### 8. Returns (Clean)
**File**: `data/processed/returns_clean.parquet`

**Description**: Log returns for GARCH-DCC estimation

**Columns**:
- Bank returns: `ret_{bank_ric}`
- Market return: `ret_market` (S&P GCC Composite)

**Properties**:
- Only trading days (holidays excluded)
- No missing values within trading days
- Properly aligned across all banks
- Ready for bivariate GARCH-DCC

---

## Data Quality Considerations

### Missing Values
- **Pattern**: Rolling NaN in daily data due to country-specific holidays
- **Handling**:
  - Prices: Forward-fill on non-trading days
  - Returns: Only calculate on trading days (skip holidays)

### Date Alignment Issues
- **UAE Calendar Transition (Jan 2022)**:
  - Pre-2022: Sunday-Thursday trading
  - Post-2022: Monday-Friday trading
  - **Solution**: Separate handling in `calendar.py`

- **Cross-Country Misalignment**:
  - Example: UAE trades on day X, Bahrain closed (holiday)
  - **Solution**: Use GCC master calendar (intersection of trading days)

### Currency Heterogeneity
- All raw data in native currencies
- **Mapping**:
  ```
  UAE banks (*.DU) → AED=
  Saudi banks (*.SE) → SAR=
  Kuwait banks (*.KW) → KWD=
  Qatar banks (*.QA) → QAR=
  Oman banks (*.OM) → OMR=
  Bahrain banks (*.BH) → BHD=
  ```

### Data Completeness
- **Start Dates**:
  - Fundamentals: 2010-Q1 (earliest reliable data)
  - Market data: 2009-12-31 (16 years)
  - Benchmark: 2009-12-31
  - Holidays: 2016-01-01 (API limitation, pre-2016 reconstructed)

---

## SRISK Calculation Mappings

### Required Fields → Data Sources

| SRISK Component | Data Source | Field | Notes |
|-----------------|-------------|-------|-------|
| **Debt (D_i,t)** | Fundamentals | `TR.F.TotLiab` | Forward-filled to daily |
| **Equity (W_i,t)** | Market Data | `TR.CompanyMarketCapitalization` | Daily market cap |
| **Bank Returns (r_i,t)** | Market Data | `TR.PriceClose` | Log returns calculated |
| **Market Returns (R_m,t)** | Benchmarks | `.GPDGC` | S&P GCC Composite returns |
| **LRMES** | GARCH-DCC | Estimated | From returns + Monte Carlo |

### SRISK Formula Mapping
```
SRISK_i,t = k × Debt_i,t - (1 - k) × (1 - LRMES_i,t) × Equity_i,t

Where:
- k = 0.08 (Basel) or 0.055 (IFRS) [from CONFIG]
- Debt_i,t = fund_{RIC}_TotLiab (USD, forward-filled)
- Equity_i,t = mktcap_{RIC} (USD)
- LRMES_i,t = Estimated from GARCH-DCC + Monte Carlo
```

---

## File Access Patterns

### Loading Raw Data
```python
from src.data import load_dataframe
from src.utils.config import RAW_DATA_DIR

# Load universe
universe = load_dataframe("gcc_banks_universe", RAW_DATA_DIR)

# Load market data
prices = load_dataframe("prices_daily", RAW_DATA_DIR)

# Load benchmarks
benchmarks = load_dataframe("benchmarks_daily", RAW_DATA_DIR)
```

### Loading Processed Data
```python
from src.utils.config import PROCESSED_DATA_DIR

# Load aligned data
aligned = load_dataframe("aligned_daily_data", PROCESSED_DATA_DIR)

# Access specific bank
enbd_returns = aligned["ret_ENBD.DU"]
enbd_mktcap = aligned["mktcap_ENBD.DU"]
enbd_debt = aligned["fund_ENBD.DU_TotLiab"]

# Access market benchmark
market_returns = aligned["bench_ret_.GPDGC"]
```

---

## Summary Statistics (Expected)

Based on prototype analysis:

- **Total Banks**: ~30-40 (exact count TBD after quality filtering)
- **Time Period**: 2010-Q1 to present (~16 years)
- **Trading Days**: ~4,000 days
- **Quarterly Periods**: ~60 quarters
- **Data Size**:
  - Fundamentals: ~60 rows × ~120 columns (~30 banks × 4 fields)
  - Market Data: ~4,000 rows × ~90 columns (~30 banks × 3 fields)
  - Aligned: ~4,000 rows × ~200 columns (all data merged)

---

## Next Steps

After data fetching and processing:
1. **GARCH-DCC Estimation** → Estimate LRMES for each bank
2. **SRISK Calculation** → Apply formula with k=8% and k=5.5%
3. **System Aggregation** → Sum positive SRISK across all banks
4. **Reporting** → Generate dashboards and technical reports

---

*Document Version: 1.0*
*Last Updated: 2026-03-08*
*Part of: GCC SRISK Calculator Project*
