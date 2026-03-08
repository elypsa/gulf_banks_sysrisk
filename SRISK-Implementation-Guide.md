# SRISK Implementation Guide for Gulf Banking Sector Analysis

## Executive Summary

SRISK (Systemic Risk) measures the **expected capital shortfall** of a financial institution conditional on a severe market decline. It quantifies how much capital a bank would need to maintain minimum prudential requirements during a crisis.

**Key Formula:**
```
SRISK_i,t = max(0, k × Debt_i,t - (1 - k) × (1 - LRMES_i,t) × Equity_i,t)
```

Where:
- **k** = Prudential capital ratio (typically 8% under Basel, adjustable to 5.5% for IFRS)
- **Debt** = Book value of debt/liabilities
- **Equity** = Market value of equity (market capitalization)
- **LRMES** = Long-Run Marginal Expected Shortfall (expected equity loss in crisis)

**Interpretation:**
- Positive SRISK = Bank undercapitalized in crisis (needs external capital)
- Negative SRISK = Bank adequately capitalized (ignored in system aggregation)
- Higher SRISK = Greater systemic importance and vulnerability

---

## 1. Conceptual Framework

### 1.1 What SRISK Measures

SRISK answers: **"How much capital would this bank need if we have another financial crisis?"**

It combines three key dimensions:
1. **Size** (larger banks → larger shortfall)
2. **Leverage** (higher debt-to-equity → larger shortfall)  
3. **Risk** (higher correlation with market crashes → larger shortfall)

### 1.2 The Crisis Scenario

**Standard Definition:**
- **Horizon (h):** 6 months (22 trading weeks)
- **Threshold (C):** -40% decline in broad market index
- **Interpretation:** If the market falls 40% over 6 months, what is bank's expected capital shortfall?

## 2. Mathematical Framework

### 2.1 Core Equations

**Step 1: Capital Shortfall**
```
CS_i,t+h = k × A_i,t+h - W_i,t+h
```
Where:
- A_i,t+h = Quasi-assets = Debt + Equity at t+h
- W_i,t+h = Market equity value at t+h
- k = Prudential capital fraction (0.08)

**Step 2: SRISK Conditional on Crisis**
```
SRISK_i,t = E_t[CS_i,t+h | R_m,t:t+h < -40%]
```

**Step 3: Decomposition (assuming debt constant)**
```
SRISK_i,t = k × Debt_i,t - (1 - k) × (1 - LRMES_i,t) × Equity_i,t
```

**Step 4: Long-Run MES (LRMES)**
```
LRMES_i,t = E_t[1 - W_i,t+h/W_i,t | R_m,t:t+h < -40%]
            = 1 - E_t[exp(r_i,t:t+h) | R_m,t:t+h < -40%]
```
Where r_i,t:t+h is cumulative log return over horizon h.

### 2.2 System-Wide Aggregation

**Total Systemic Risk:**
```
SRISK_system,t = Σ max(0, SRISK_i,t)  for all banks i
```

**Important:** Only sum **positive** SRISK values. Banks with negative SRISK (capital surplus) are excluded—capital cannot be easily mobilized across institutions in crisis.

**Bank's Contribution:**
```
Contribution_%_i,t = SRISK_i,t / SRISK_system,t × 100%
```

### 2.3 LRMES Estimation via GARCH-DCC

The critical component is estimating LRMES. The standard approach uses **bivariate GJR-GARCH(1,1) with Dynamic Conditional Correlation (DCC)**.

**Return Model:**
```
r_i,t = μ_i + ε_i,t
r_m,t = μ_m + ε_m,t

where ε_i,t = σ_i,t × z_i,t
      ε_m,t = σ_m,t × z_m,t
```

**Volatility (GJR-GARCH):**
```
σ²_i,t = ω_i + α_i ε²_i,t-1 + γ_i I[ε_i,t-1<0] ε²_i,t-1 + β_i σ²_i,t-1
```
Where I[·] is indicator for asymmetric (leverage) effect.

**Dynamic Correlation (DCC):**
```
Q_t = (1 - a - b) Q̄ + a (z_t-1 z'_t-1) + b Q_t-1
R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
```
Where R_t is time-varying correlation matrix.

**LRMES Computation:**
```
LRMES_i,t = 1 - E_t[exp(Σ r_i,s) | Σ r_m,s < -40%]
```
Estimated via **Monte Carlo simulation** (50,000+ paths):
1. Generate h-step ahead returns using GARCH-DCC
2. Keep only paths where R_m,t:t+h < -40%
3. Average firm equity decline across crisis scenarios

---

## 3. Data Requirements

### 3.1 Market Data (Daily Frequency)

**For Each Bank:**
- **Stock prices** (adjusted for splits/dividends)
- **Shares outstanding** → Market cap = Price × Shares


**For Benchmark Index:**
- **GCC Composite Index** or **country-specific banking index**


**Time Period:**
- **Minimum:** 252 trading days (~1 year)
- **Recommended:** 1,000+ days (~4 years) for stable GARCH estimates
- **Rolling window:** Update daily/weekly with expanding or 3-5 year rolling window

### 3.2 Balance Sheet Data (Quarterly/Annual)

**Required:**
- **Total Debt/Liabilities** (book value): D_i,t
- **Market value of Equity:** W_i,t = Price × Shares
- **Quasi-Leverage Ratio:** LVG = (Debt + Equity) / Equity

**Frequency:**
- Balance sheet: Quarterly
- Interpolate between reporting dates (assume constant until next release)

### 3.3 Gulf Region Specifics

**Listed Banks by Country (approx counts):**
- **Saudi Arabia:** ~10 listed banks (dominant weight)
- **UAE:** ~8 banks (Abu Dhabi, Dubai markets)
- **Qatar:** ~4 banks
- **Kuwait:** ~8 banks
- **Bahrain:** ~5 banks
- **Oman:** ~3 banks

**Total:** ~40 listed banks in GCC region

## 5. Numerical Issues and Pitfalls

### 5.2 DCC Estimation Challenges

**Problem:** DCC correlation matrix not positive definite

**Solutions:**
- Impose constraints: a + b < 1, both > 0
- Use Ledoit-Wolf shrinkage for Q̄ (unconditional correlation)
- Check if standardized residuals are truly standardized (mean 0, var 1)
- Consider simpler CCC (constant correlation) if DCC fails


### 5.4 Sparse Crisis Scenarios

**Problem:** Too few simulations meet crisis threshold (R_m < -40%)

**Typical:** 5-10% of simulations → ~2,500-5,000 crisis paths out of 50,000

**If fewer than 500 crisis scenarios:**
- Increase n_sim to 100,000
- Consider less extreme threshold (-30% instead of -40%)
- Check if market volatility is too low (market calm → few crises simulated)
- Use importance sampling (bias simulations toward tail)

**-40% may be too extreme for Gulf markets:**
- Check historical GCC index drawdowns
- Largest decline in sample 
- Consider **regional calibration:** Use 5th percentile of historical 6-month returns


## 6. Optimal Presentation and Outputs

### 6.1 For Senior Management (Executive Dashboard)

**Key Metrics:**
1. **System-Wide SRISK**
   - Total capital shortfall (USD billions or % of GDP)
   - Time series plot (monthly/quarterly) with crisis periods highlighted
   - Traffic light: Green (<2% GDP), Yellow (2-5%), Red (>5%)

2. **Top 10 Systemically Important Banks**
   - Ranked table with SRISK, % contribution, trend arrows
   - Breakdown: Size effect vs Leverage effect vs Risk effect

3. **Heatmap by Country**
   - GCC countries × Time periods
   - Color intensity = SRISK contribution

4. **Leading Indicators**
   - Change in aggregate SRISK (ΔSRISKₜ - ΔSRISKₜ₋₁)
   - Correlation with macro indicators (oil prices, credit growth)

### 6.2 Technical Report (For Analysts)

**Section 1: Executive Summary**
- Current system SRISK and change from previous quarter
- Top 5 banks and their contributions
- Key drivers (leverage increases, equity declines, volatility spikes)

**Section 2: Individual Bank Profiles**
For each bank:
- SRISK level and ranking
- LRMES (expected equity loss in crisis)
- Leverage ratio (Debt/Equity)
- Market beta and correlation with GCC index
- 12-month trend chart

**Section 3: Decomposition Analysis**
```
ΔSRISK = ΔSize effect + ΔLeverage effect + ΔRisk effect
```
Quantify contribution of each factor to changes in SRISK.

**Section 4: Scenario Analysis**
- Alternative benchmarks (global vs regional index)

**Section 5: Cross-Country Comparison**
- SRISK by country


**Section 6: Model Diagnostics**
- GARCH parameter estimates and standard errors
- DCC correlation dynamics (plot ρₜ over time)
- Goodness-of-fit: Ljung-Box tests on standardized residuals
- Monte Carlo convergence checks

