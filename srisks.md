# SRISK Mathematical and Econometric Foundations

**Purpose:** Detailed statistical and econometric derivation of SRISK methodology  
**Audience:** Technical implementation with full mathematical rigor  
**Complements:** SRISK-Implementation-Guide.md

---

## Table of Contents

1. [Theoretical Foundation: Capital Shortfall in Crisis](#1-theoretical-foundation)
2. [Mathematical Framework: From MES to LRMES](#2-mathematical-framework)
3. [Econometric Specification: GJR-GARCH Model](#3-econometric-specification)
4. [Dynamic Conditional Correlation (DCC)](#4-dynamic-conditional-correlation)
5. [Monte Carlo Simulation Methodology](#5-monte-carlo-simulation)
6. [Statistical Properties and Inference](#6-statistical-properties)
7. [Complete Estimation Algorithm](#7-complete-algorithm)
8. [Numerical Implementation Details](#8-numerical-implementation)

---

## 1. Theoretical Foundation: Capital Shortfall in Crisis

### 1.1 Regulatory Capital Framework

**Definition:** A financial institution is adequately capitalized if its equity exceeds a prudential fraction k of its assets.

**Capital Adequacy Condition:**
$$W_{i,t} \geq k \cdot A_{i,t}$$

Where:
- $W_{i,t}$ = Market value of equity for bank $i$ at time $t$
- $A_{i,t}$ = Total assets (quasi-market value)
- $k$ = Prudential capital ratio (typically 0.08 under Basel III)

**Assets Decomposition:**
$$A_{i,t} = D_{i,t} + W_{i,t}$$

Where $D_{i,t}$ is book value of debt (assumed constant in short run).

### 1.2 Capital Shortfall Definition

**At time $t+h$ (future horizon), capital shortfall is:**

$$\text{CS}_{i,t+h} = k \cdot A_{i,t+h} - W_{i,t+h}$$

**Substituting $A_{i,t+h} = D_{i,t} + W_{i,t+h}$:**

$$\text{CS}_{i,t+h} = k(D_{i,t} + W_{i,t+h}) - W_{i,t+h}$$

$$= k \cdot D_{i,t} + k \cdot W_{i,t+h} - W_{i,t+h}$$

$$= k \cdot D_{i,t} - (1-k) \cdot W_{i,t+h}$$

**This is the capital needed at $t+h$ to meet regulatory requirements.**

### 1.3 SRISK: Expected Shortfall in Crisis

**SRISK is the expected capital shortfall conditional on a systemic crisis:**

$$\text{SRISK}_{i,t} = \mathbb{E}_t[\text{CS}_{i,t+h} \mid \text{Crisis at } t{:}t+h]$$

**Define crisis as market decline exceeding threshold $C$:**
$$\text{Crisis} = \{R_{m,t:t+h} < C\}$$

Where $R_{m,t:t+h} = \frac{M_{t+h}}{M_t} - 1$ is cumulative market return over horizon $h$.

**Full SRISK Formula:**
$$\text{SRISK}_{i,t} = \mathbb{E}_t[k \cdot D_{i,t} - (1-k) \cdot W_{i,t+h} \mid R_{m,t:t+h} < C]$$

**Key assumption:** Debt is constant: $D_{i,t+h} \approx D_{i,t}$ (reasonable for short horizons like 6 months).

### 1.4 Decomposition Using LRMES

**Define Long-Run Marginal Expected Shortfall (LRMES):**

$$\text{LRMES}_{i,t} = \mathbb{E}_t\left[1 - \frac{W_{i,t+h}}{W_{i,t}} \bigg| R_{m,t:t+h} < C\right]$$

**This is the expected percentage decline in equity conditional on crisis.**

**Rewrite $W_{i,t+h}$ in terms of LRMES:**
$$\frac{W_{i,t+h}}{W_{i,t}} = 1 - \text{LRMES}_{i,t} \text{ (in expectation under crisis)}$$

$$\mathbb{E}_t[W_{i,t+h} \mid \text{Crisis}] = W_{i,t}(1 - \text{LRMES}_{i,t})$$

**Substitute into SRISK:**
$$\text{SRISK}_{i,t} = k \cdot D_{i,t} - (1-k) \cdot W_{i,t}(1 - \text{LRMES}_{i,t})$$

$$= k \cdot D_{i,t} - (1-k)(1 - \text{LRMES}_{i,t}) \cdot W_{i,t}$$

**Final SRISK Formula:**
$$\boxed{\text{SRISK}_{i,t} = \max\left(0, k \cdot D_{i,t} - (1-k)(1 - \text{LRMES}_{i,t}) \cdot W_{i,t}\right)}$$

**The max with 0 ensures we only count capital needs, not surpluses.**

### 1.5 Interpretation of Components

**High SRISK arises from:**
1. **Large Size:** High $D_{i,t}$ and $W_{i,t}$ → large absolute shortfall
2. **High Leverage:** High $D_{i,t}/W_{i,t}$ → more debt relative to equity
3. **High Risk:** High $\text{LRMES}_{i,t}$ → large expected equity loss in crisis

**Example:**
- Bank with $D = 100$bn, $W = 10$bn, $k = 0.08$, $\text{LRMES} = 0.50$
- $\text{SRISK} = 0.08 \times 100 - 0.92 \times 0.50 \times 10$
- $= 8 - 4.6 = 3.4$ billion USD needed in crisis

---

## 2. Mathematical Framework: From MES to LRMES

### 2.1 Marginal Expected Shortfall (MES)

**1-Day MES:** Expected equity return when market has extreme negative return.

**Definition:**
$$\text{MES}_{i,t} = \mathbb{E}_t[R_{i,t+1} \mid R_{m,t+1} < C_{\text{1-day}}]$$

Where $C_{\text{1-day}}$ is typically -2% or 5th percentile.

**Relationship to Conditional Value-at-Risk:**
$$\text{MES}_{i,t} = -\text{CVaR}_{i,t}(\alpha) \text{ at tail probability } \alpha$$

### 2.2 From 1-Day MES to Long-Run MES

**Multi-period returns (log scale):**
$$r_{i,t:t+h} = \sum_{s=t+1}^{t+h} r_{i,s} = \ln\left(\frac{W_{i,t+h}}{W_{i,t}}\right)$$

**LRMES in log returns:**
$$\text{LRMES}_{i,t} = \mathbb{E}_t\left[1 - e^{r_{i,t:t+h}} \bigg| r_{m,t:t+h} < \ln(1+C)\right]$$

**For $C = -0.40$:**
$$\ln(1 - 0.40) = \ln(0.60) \approx -0.511$$

So crisis condition becomes:
$$r_{m,t:t+h} < -0.511$$

### 2.3 Why We Can't Use Simple Aggregation

**Naive approach (WRONG):**
$$\text{LRMES} \neq \sum_{s=t+1}^{t+h} \text{MES}_{i,s}$$

**Why this fails:**
1. **Time-varying volatility:** $\sigma_{i,t}$ changes over crisis
2. **Dynamic correlation:** $\rho_{i,m,t}$ increases during stress
3. **Path dependence:** Volatility clustering affects cumulative returns
4. **Non-linearity:** $e^{\sum r_s} \neq \sum e^{r_s}$

**Correct approach:** **Monte Carlo simulation** that respects dynamic volatility and correlation.

### 2.4 Connecting to Bivariate Distribution

**Joint return distribution at horizon $h$:**
$$\begin{bmatrix} r_{i,t:t+h} \\ r_{m,t:t+h} \end{bmatrix} \bigg| \mathcal{F}_t \sim F_{i,m}(\cdot | \Theta_t)$$

Where $\Theta_t$ contains GARCH parameters and current volatility states.

**LRMES is conditional expectation over joint distribution:**
$$\text{LRMES}_{i,t} = \int_{-\infty}^{0.60} \int_{-\infty}^{\infty} \left(1 - e^{r_i}\right) \cdot f_{i,m}(r_i, r_m | r_m < -0.511) \, dr_i \, dr_m$$

**This integral has no closed form → use simulation.**

---

## 3. Econometric Specification: GJR-GARCH Model

### 3.1 Return Dynamics

**Assume zero mean for simplicity (can include AR terms):**
$$r_{i,t} = \mu_i + \epsilon_{i,t}$$
$$r_{m,t} = \mu_m + \epsilon_{m,t}$$

Where $\epsilon_{i,t}$ are innovations with conditional variance $\sigma_{i,t}^2$.

**Standardized residuals:**
$$z_{i,t} = \frac{\epsilon_{i,t}}{\sigma_{i,t}}, \quad z_{m,t} = \frac{\epsilon_{m,t}}{\sigma_{m,t}}$$

With $\mathbb{E}_t[z_{i,t}] = 0$, $\text{Var}_t[z_{i,t}] = 1$.

### 3.2 GJR-GARCH(1,1) Specification

**Volatility equation (Glosten-Jagannathan-Runkle):**

$$\sigma_{i,t}^2 = \omega_i + \alpha_i \epsilon_{i,t-1}^2 + \gamma_i \epsilon_{i,t-1}^2 \mathbb{I}[\epsilon_{i,t-1} < 0] + \beta_i \sigma_{i,t-1}^2$$

**Components:**
- $\omega_i > 0$: Constant (long-run variance component)
- $\alpha_i \geq 0$: Symmetric ARCH effect
- $\gamma_i \geq 0$: **Asymmetric effect** (leverage effect)
- $\beta_i \geq 0$: GARCH persistence
- $\mathbb{I}[\cdot]$: Indicator function for negative shocks

**Interpretation of $\gamma_i$:**
- If $\gamma_i > 0$: Negative shocks increase volatility more than positive shocks
- Captures "leverage effect" in equities: bad news → higher volatility
- Typically $\gamma_i > 0$ for bank stocks

**Total impact of negative shock:**
$$\sigma_{i,t}^2 = \omega_i + (\alpha_i + \gamma_i) \epsilon_{i,t-1}^2 + \beta_i \sigma_{i,t-1}^2 \quad \text{if } \epsilon_{i,t-1} < 0$$

**Stationarity condition:**
$$\alpha_i + \frac{\gamma_i}{2} + \beta_i < 1$$

**Unconditional variance (if stationary):**
$$\mathbb{E}[\sigma_{i,t}^2] = \frac{\omega_i}{1 - \alpha_i - \gamma_i/2 - \beta_i}$$

### 3.3 Why GJR-GARCH for SRISK?

**Advantages over standard GARCH(1,1):**
1. **Captures leverage effect:** Banks' volatility spikes more in downturns
2. **Better tail behavior:** More realistic crisis scenario simulation
3. **Empirically superior:** Fits financial data better (lower AIC)

**Comparison:**
- **GARCH(1,1):** Symmetric response to shocks
- **GJR-GARCH(1,1):** Negative shocks have $(\alpha + \gamma)$ impact vs $\alpha$ for positive
- **EGARCH:** Alternative, but GJR easier to estimate and simulate

### 3.4 Maximum Likelihood Estimation

**Log-likelihood for sample $t = 1, \ldots, T$:**

$$\mathcal{L}(\theta_i) = -\frac{1}{2} \sum_{t=1}^{T} \left[\ln(2\pi) + \ln(\sigma_{i,t}^2) + \frac{\epsilon_{i,t}^2}{\sigma_{i,t}^2}\right]$$

Where $\theta_i = (\mu_i, \omega_i, \alpha_i, \gamma_i, \beta_i)$.

**Estimation:**
1. Initialize $\sigma_{1}^2$ (use sample variance or pre-sample average)
2. For $t = 2, \ldots, T$:
   - Compute $\epsilon_{t} = r_{t} - \mu$
   - Update $\sigma_t^2$ using GJR-GARCH equation
   - Accumulate log-likelihood
3. Maximize $\mathcal{L}(\theta)$ numerically (quasi-Newton methods)

**Practical considerations:**
- Use robust standard errors (Bollerslev-Wooldridge)
- Check parameter bounds during optimization
- Verify convergence with multiple starting values

### 3.5 Forecasting Volatility

**1-step ahead:**
$$\sigma_{i,t+1}^2 = \omega_i + \alpha_i \epsilon_{i,t}^2 + \gamma_i \epsilon_{i,t}^2 \mathbb{I}[\epsilon_{i,t} < 0] + \beta_i \sigma_{i,t}^2$$

**Multi-step ahead (for simulation at $t+s$, $s > 1$):**

Given path $\{\epsilon_{t+1}, \ldots, \epsilon_{t+s-1}\}$:
$$\sigma_{i,t+s}^2 = \omega_i + \alpha_i \epsilon_{i,t+s-1}^2 + \gamma_i \epsilon_{i,t+s-1}^2 \mathbb{I}[\epsilon_{i,t+s-1} < 0] + \beta_i \sigma_{i,t+s-1}^2$$

**This recursion is used in Monte Carlo simulation.**

---

## 4. Dynamic Conditional Correlation (DCC)

### 4.1 Motivation

**Joint modeling challenge:**
- Need time-varying correlation between bank and market
- Correlations increase during crises (contagion)
- Full multivariate GARCH too complex (too many parameters)

**Solution:** Two-step estimation (Engle 2002)
1. **Step 1:** Univariate GARCH for each series → get $\sigma_{i,t}$, $\sigma_{m,t}$
2. **Step 2:** Model correlation of standardized residuals

### 4.2 DCC Model Specification

**Standardized residuals from Step 1:**
$$z_{i,t} = \frac{\epsilon_{i,t}}{\sigma_{i,t}}, \quad z_{m,t} = \frac{\epsilon_{m,t}}{\sigma_{m,t}}$$

**Stack into vector:**
$$\mathbf{z}_t = \begin{bmatrix} z_{i,t} \\ z_{m,t} \end{bmatrix}$$

**DCC dynamics:**

$$\mathbf{Q}_t = (1 - a - b) \bar{\mathbf{Q}} + a (\mathbf{z}_{t-1} \mathbf{z}_{t-1}^T) + b \mathbf{Q}_{t-1}$$

Where:
- $\mathbf{Q}_t$ is pseudo-correlation matrix (not standardized yet)
- $\bar{\mathbf{Q}} = \mathbb{E}[\mathbf{z}_t \mathbf{z}_t^T]$ is unconditional correlation of standardized residuals
- $a, b \geq 0$ with $a + b < 1$ (stationarity)

**Standardize to get correlation matrix:**
$$\mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \mathbf{Q}_t \text{diag}(\mathbf{Q}_t)^{-1/2}$$

**In 2×2 case:**
$$\mathbf{Q}_t = \begin{bmatrix} q_{11,t} & q_{12,t} \\ q_{21,t} & q_{22,t} \end{bmatrix}$$

$$\mathbf{R}_t = \begin{bmatrix} 1 & \rho_t \\ \rho_t & 1 \end{bmatrix}$$

Where:
$$\rho_t = \frac{q_{12,t}}{\sqrt{q_{11,t} q_{22,t}}}$$

**This is the time-varying correlation between bank and market.**

### 4.3 DCC Evolution Equation

**For correlation $\rho_t$, expand the DCC dynamics:**

$$q_{12,t} = (1 - a - b) \bar{q}_{12} + a z_{i,t-1} z_{m,t-1} + b q_{12,t-1}$$

$$q_{11,t} = (1 - a - b) \bar{q}_{11} + a z_{i,t-1}^2 + b q_{11,t-1}$$

$$q_{22,t} = (1 - a - b) \bar{q}_{22} + a z_{m,t-1}^2 + b q_{22,t-1}$$

**Interpretation:**
- **a**: Sensitivity to recent co-movements (short-run dynamics)
- **b**: Persistence of correlation (long-run memory)
- High $b$ → correlations change slowly
- High $a$ → correlations react quickly to shocks

**Typical estimates for financials:**
- $a \approx 0.01$ to $0.05$
- $b \approx 0.90$ to $0.98$
- Sum $a + b \approx 0.95$ (very persistent)

### 4.4 Maximum Likelihood Estimation of DCC

**Log-likelihood (conditional on Step 1 volatilities):**

$$\mathcal{L}(a, b) = -\frac{1}{2} \sum_{t=1}^{T} \left[\ln|\mathbf{R}_t| + \mathbf{z}_t^T \mathbf{R}_t^{-1} \mathbf{z}_t - \mathbf{z}_t^T \mathbf{z}_t\right]$$

**For 2×2 case, simplifies to:**

$$\mathcal{L}(a, b) = -\frac{1}{2} \sum_{t=1}^{T} \left[\ln(1 - \rho_t^2) + \frac{z_{i,t}^2 + z_{m,t}^2 - 2\rho_t z_{i,t} z_{m,t}}{1 - \rho_t^2} - (z_{i,t}^2 + z_{m,t}^2)\right]$$

**Estimation procedure:**
1. Compute $\bar{\mathbf{Q}} = \frac{1}{T} \sum_{t=1}^T \mathbf{z}_t \mathbf{z}_t^T$
2. Initialize $\mathbf{Q}_1 = \bar{\mathbf{Q}}$
3. For candidate $(a, b)$:
   - Recursively compute $\mathbf{Q}_t$ for $t = 2, \ldots, T$
   - Compute $\mathbf{R}_t$ from $\mathbf{Q}_t$
   - Evaluate log-likelihood
4. Maximize over $(a, b)$ subject to:
   - $a, b \geq 0$
   - $a + b < 1$
   - $\mathbf{R}_t$ positive definite for all $t$

**Optimizer:** Use constrained optimization (L-BFGS-B, SLSQP)

### 4.5 Properties of DCC Correlation

**Stationarity:**
If $a + b < 1$, then $\rho_t$ converges to:
$$\bar{\rho} = \frac{\bar{q}_{12}}{\sqrt{\bar{q}_{11} \bar{q}_{22}}}$$

**Persistence:**
Half-life of correlation shock = $\frac{\ln(0.5)}{\ln(b)}$

Example: If $b = 0.95$, half-life = 13.5 days

**Crisis behavior:**
- During crisis: Negative shocks → $z_{i,t-1}, z_{m,t-1}$ both negative
- Product $z_{i,t-1} z_{m,t-1} > 0$ and large
- $q_{12,t}$ increases → $\rho_t$ increases
- **Correlation rises during joint stress (desirable property)**

---

## 5. Monte Carlo Simulation Methodology

### 5.1 Simulation Objective

**Goal:** Estimate LRMES by simulating future paths under GARCH-DCC dynamics.

$$\text{LRMES}_{i,t} = \mathbb{E}_t\left[1 - e^{r_{i,t:t+h}} \bigg| r_{m,t:t+h} < -0.511\right]$$

**Procedure:**
1. Simulate $N$ paths of $(r_{i,t+1}, \ldots, r_{i,t+h})$ and $(r_{m,t+1}, \ldots, r_{m,t+h})$
2. Keep only paths where $\sum_{s=1}^h r_{m,t+s} < -0.511$ (crisis paths)
3. For these crisis paths, compute bank equity decline: $1 - e^{\sum_{s=1}^h r_{i,t+s}}$
4. Average over crisis paths → LRMES estimate

### 5.2 Multi-Step Path Generation

**For each simulation path $n = 1, \ldots, N$:**

**Initialize at current state:**
- Volatilities: $\sigma_{i,t}^{(n)} = \sigma_{i,t}$, $\sigma_{m,t}^{(n)} = \sigma_{m,t}$
- Correlation matrix: $\mathbf{Q}_t^{(n)} = \mathbf{Q}_t$, $\mathbf{R}_t^{(n)} = \mathbf{R}_t$
- Cumulative returns: $r_{i,\text{cum}}^{(n)} = 0$, $r_{m,\text{cum}}^{(n)} = 0$

**For each step $s = 1, \ldots, h$:**

**Step 1:** Generate correlated standardized shocks
$$\mathbf{z}_{t+s}^{(n)} = \begin{bmatrix} z_{i,t+s}^{(n)} \\ z_{m,t+s}^{(n)} \end{bmatrix} \sim N(\mathbf{0}, \mathbf{R}_{t+s-1}^{(n)})$$

**Step 2:** Compute returns
$$r_{i,t+s}^{(n)} = \mu_i + \sigma_{i,t+s}^{(n)} \cdot z_{i,t+s}^{(n)}$$
$$r_{m,t+s}^{(n)} = \mu_m + \sigma_{m,t+s}^{(n)} \cdot z_{m,t+s}^{(n)}$$

**Step 3:** Accumulate
$$r_{i,\text{cum}}^{(n)} \leftarrow r_{i,\text{cum}}^{(n)} + r_{i,t+s}^{(n)}$$
$$r_{m,\text{cum}}^{(n)} \leftarrow r_{m,\text{cum}}^{(n)} + r_{m,t+s}^{(n)}$$

**Step 4:** Update volatilities (GJR-GARCH)
$$\epsilon_{i,t+s}^{(n)} = r_{i,t+s}^{(n)} - \mu_i = \sigma_{i,t+s}^{(n)} \cdot z_{i,t+s}^{(n)}$$

$$\sigma_{i,t+s+1}^{2,(n)} = \omega_i + \alpha_i (\epsilon_{i,t+s}^{(n)})^2 + \gamma_i (\epsilon_{i,t+s}^{(n)})^2 \mathbb{I}[\epsilon_{i,t+s}^{(n)} < 0] + \beta_i (\sigma_{i,t+s}^{(n)})^2$$

Similarly for market:
$$\sigma_{m,t+s+1}^{2,(n)} = \omega_m + \alpha_m (\epsilon_{m,t+s}^{(n)})^2 + \gamma_m (\epsilon_{m,t+s}^{(n)})^2 \mathbb{I}[\epsilon_{m,t+s}^{(n)} < 0] + \beta_m (\sigma_{m,t+s}^{(n)})^2$$

**Step 5:** Update correlation (DCC)
$$\mathbf{Q}_{t+s}^{(n)} = (1-a-b) \bar{\mathbf{Q}} + a (\mathbf{z}_{t+s}^{(n)} \mathbf{z}_{t+s}^{(n)T}) + b \mathbf{Q}_{t+s-1}^{(n)}$$

$$\mathbf{R}_{t+s}^{(n)} = \text{diag}(\mathbf{Q}_{t+s}^{(n)})^{-1/2} \mathbf{Q}_{t+s}^{(n)} \text{diag}(\mathbf{Q}_{t+s}^{(n)})^{-1/2}$$

**After $h$ steps, we have:**
- Bank cumulative return: $r_{i,t:t+h}^{(n)}$
- Market cumulative return: $r_{m,t:t+h}^{(n)}$

### 5.3 Generating Correlated Normal Shocks

**Need:** Draw $\mathbf{z}_t = (z_{i,t}, z_{m,t})^T \sim N(\mathbf{0}, \mathbf{R}_t)$

**Method 1: Cholesky Decomposition**

Given $\mathbf{R}_t = \begin{bmatrix} 1 & \rho_t \\ \rho_t & 1 \end{bmatrix}$

Compute Cholesky: $\mathbf{R}_t = \mathbf{L}_t \mathbf{L}_t^T$

$$\mathbf{L}_t = \begin{bmatrix} 1 & 0 \\ \rho_t & \sqrt{1 - \rho_t^2} \end{bmatrix}$$

**Algorithm:**
1. Draw independent $\mathbf{u} = (u_1, u_2)^T \sim N(\mathbf{0}, \mathbf{I})$
2. Compute $\mathbf{z}_t = \mathbf{L}_t \mathbf{u}$

$$z_{i,t} = u_1$$
$$z_{m,t} = \rho_t u_1 + \sqrt{1 - \rho_t^2} \cdot u_2$$

**Verification:**
$$\mathbb{E}[z_{i,t} z_{m,t}] = \mathbb{E}[u_1 (\rho_t u_1 + \sqrt{1-\rho_t^2} u_2)] = \rho_t \mathbb{E}[u_1^2] = \rho_t$$

**Method 2: Direct Bivariate Normal**

Use numpy/scipy:
```python
import numpy as np
R_t = np.array([[1, rho_t], [rho_t, 1]])
z_t = np.random.multivariate_normal(mean=[0, 0], cov=R_t)
```

### 5.4 LRMES Calculation from Simulations

**After $N$ simulations, we have:**
- Market cumulative returns: $\{r_{m,t:t+h}^{(n)}\}_{n=1}^N$
- Bank cumulative returns: $\{r_{i,t:t+h}^{(n)}\}_{n=1}^N$

**Step 1: Identify crisis scenarios**
$$\mathcal{C} = \{n : r_{m,t:t+h}^{(n)} < -0.511\}$$

Let $N_c = |\mathcal{C}|$ = number of crisis paths.

**Step 2: Compute equity returns in crisis**
For $n \in \mathcal{C}$:
$$\text{Equity Return}^{(n)} = e^{r_{i,t:t+h}^{(n)}} - 1$$

**Step 3: Compute losses in crisis**
$$\text{Loss}^{(n)} = 1 - e^{r_{i,t:t+h}^{(n)}}$$

**Step 4: LRMES estimate**
$$\widehat{\text{LRMES}}_{i,t} = \frac{1}{N_c} \sum_{n \in \mathcal{C}} \left(1 - e^{r_{i,t:t+h}^{(n)}}\right)$$

**Alternative (equivalent):**
$$\widehat{\text{LRMES}}_{i,t} = 1 - \frac{1}{N_c} \sum_{n \in \mathcal{C}} e^{r_{i,t:t+h}^{(n)}}$$

### 5.5 Simulation Standard Error

**LRMES is sample average over crisis scenarios:**
$$\widehat{\text{LRMES}}_{i,t} = \frac{1}{N_c} \sum_{n \in \mathcal{C}} Y^{(n)}$$

Where $Y^{(n)} = 1 - e^{r_{i,t:t+h}^{(n)}}$.

**Variance:**
$$\text{Var}[\widehat{\text{LRMES}}_{i,t}] = \frac{\sigma_Y^2}{N_c}$$

Where $\sigma_Y^2 = \text{Var}[Y | \text{Crisis}]$.

**Standard error:**
$$\text{SE}[\widehat{\text{LRMES}}_{i,t}] = \frac{\hat{\sigma}_Y}{\sqrt{N_c}}$$

Where:
$$\hat{\sigma}_Y^2 = \frac{1}{N_c - 1} \sum_{n \in \mathcal{C}} \left(Y^{(n)} - \widehat{\text{LRMES}}_{i,t}\right)^2$$

**Example:**
- $N = 50,000$ simulations
- $N_c \approx 5,000$ crisis scenarios (10%)
- If $\hat{\sigma}_Y = 0.15$:
  - $\text{SE} = 0.15 / \sqrt{5000} \approx 0.0021 = 0.21\%$

**Confidence interval (95%):**
$$\text{LRMES}_{i,t} \in \widehat{\text{LRMES}}_{i,t} \pm 1.96 \times \text{SE}$$

### 5.6 Importance Sampling (Optional)

**Problem:** If crisis rare, $N_c$ small → high variance.

**Solution:** Bias simulations toward tail.

**Tilted distribution:**
Instead of $z_t \sim N(0, 1)$, use:
$$z_t^* \sim N(\mu_{\text{tilt}}, 1) \text{ where } \mu_{\text{tilt}} < 0$$

**Correction via importance weights:**
$$\widehat{\text{LRMES}}_{i,t}^{\text{IS}} = \frac{\sum_{n=1}^N w^{(n)} Y^{(n)} \mathbb{I}[n \in \mathcal{C}]}{\sum_{n=1}^N w^{(n)} \mathbb{I}[n \in \mathcal{C}]}$$

Where:
$$w^{(n)} = \frac{\phi(z_t^{(n)}; 0, 1)}{\phi(z_t^{(n)}; \mu_{\text{tilt}}, 1)}$$

$\phi(\cdot)$ is normal density.

**Not necessary if $N = 50,000$ is sufficient.**

---

## 6. Statistical Properties and Inference

### 6.1 SRISK as Composite Estimator

**SRISK combines three estimated components:**
1. Balance sheet data: $D_{i,t}$, $W_{i,t}$ (observed)
2. GARCH-DCC parameters: $\theta_i, \theta_m, a, b$ (estimated)
3. LRMES: Simulated expectation (Monte Carlo)

**Full uncertainty should account for:**
- Parameter estimation error in GARCH-DCC
- Simulation error in LRMES
- Model misspecification

### 6.2 Bootstrap Confidence Intervals

**Parametric bootstrap for SRISK:**

**Step 1:** Obtain point estimates $\hat{\theta}_i, \hat{\theta}_m, \hat{a}, \hat{b}$

**Step 2:** For $b = 1, \ldots, B$ bootstrap iterations:
1. Resample residuals $\{\epsilon_t^*\}$ from empirical distribution of $\{\hat{\epsilon}_t\}$
2. Re-estimate GARCH-DCC on bootstrapped sample → $\hat{\theta}_i^{(b)}, \hat{a}^{(b)}, \hat{b}^{(b)}$
3. Simulate LRMES with bootstrapped parameters → $\widehat{\text{LRMES}}_{i,t}^{(b)}$
4. Compute $\text{SRISK}_{i,t}^{(b)}$

**Step 3:** Compute 95% confidence interval:
$$\text{SRISK}_{i,t} \in \left[Q_{0.025}\{\text{SRISK}^{(b)}\}, Q_{0.975}\{\text{SRISK}^{(b)}\}\right]$$

**Practical note:** Computationally intensive. For surveillance, focus on point estimates and track changes over time.

### 6.3 Hypothesis Tests

**Test 1: Is bank systemically important?**
$$H_0: \text{SRISK}_{i,t} = 0 \text{ vs } H_1: \text{SRISK}_{i,t} > 0$$

Use bootstrap distribution to test.

**Test 2: Has SRISK increased significantly?**
$$H_0: \text{SRISK}_{i,t} = \text{SRISK}_{i,t-s} \text{ vs } H_1: \text{SRISK}_{i,t} > \text{SRISK}_{i,t-s}$$

**Practical approach:** Consider increase significant if:
$$\Delta \text{SRISK} > 2 \times \text{SE}[\text{SRISK}_{i,t}]$$

### 6.4 Model Diagnostics

**GARCH adequacy:**

**Ljung-Box test on standardized residuals $\hat{z}_t$:**
$$Q(m) = T(T+2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m)$$

Where $\hat{\rho}_k$ is sample autocorrelation at lag $k$.

**Null:** No autocorrelation (model adequate)  
**Reject if:** $Q(m) > \chi^2_{1-\alpha}(m)$

**Test on $\hat{z}_t^2$ (ARCH effects):**
Should also show no autocorrelation.

**DCC adequacy:**

**Test if DCC improves over CCC (constant correlation):**
- Likelihood ratio test: $\text{LR} = 2(\mathcal{L}_{\text{DCC}} - \mathcal{L}_{\text{CCC}})$
- Under $H_0$: $\text{LR} \sim \chi^2(2)$ (2 parameters: $a, b$)
- Reject CCC if $\text{LR} > 5.99$ (5% level)

**Backtest LRMES:**
- Compute LRMES for historical dates
- Check if realized losses in actual crises align with LRMES predictions
- ROC curve: Does high LRMES predict actual distress?

---

## 7. Complete Estimation Algorithm

### 7.1 Full Workflow

**Input:**
- Daily returns: $\{r_{i,t}\}_{t=1}^T$, $\{r_{m,t}\}_{t=1}^T$
- Balance sheet: $D_{i,t}$, $W_{i,t}$ at time $t$
- Parameters: $h = 126$, $C = -0.40$, $k = 0.08$, $N = 50,000$

**Output:**
- $\text{SRISK}_{i,t}$ and $\text{LRMES}_{i,t}$

---

**STEP 1: Estimate Univariate GJR-GARCH Models**

**1.1 For bank $i$:**
- Specify model: $r_{i,t} = \mu_i + \epsilon_{i,t}$
- $\sigma_{i,t}^2 = \omega_i + \alpha_i \epsilon_{i,t-1}^2 + \gamma_i \epsilon_{i,t-1}^2 \mathbb{I}[\epsilon_{i,t-1}<0] + \beta_i \sigma_{i,t-1}^2$
- Maximize log-likelihood → $\hat{\theta}_i = (\hat{\mu}_i, \hat{\omega}_i, \hat{\alpha}_i, \hat{\gamma}_i, \hat{\beta}_i)$
- Extract: $\{\hat{\sigma}_{i,t}\}_{t=1}^T$, $\{\hat{z}_{i,t}\}_{t=1}^T = \{\hat{\epsilon}_{i,t} / \hat{\sigma}_{i,t}\}$

**1.2 For market $m$:**
- Same procedure → $\hat{\theta}_m$, $\{\hat{\sigma}_{m,t}\}$, $\{\hat{z}_{m,t}\}$

**1.3 Diagnostics:**
- Check stationarity: $\hat{\alpha}_i + \hat{\gamma}_i/2 + \hat{\beta}_i < 1$
- Ljung-Box test on $\hat{z}_{i,t}$ and $\hat{z}_{i,t}^2$
- If fails: Try different specification or longer sample

---

**STEP 2: Estimate DCC Parameters**

**2.1 Compute unconditional correlation matrix:**
$$\bar{\mathbf{Q}} = \frac{1}{T} \sum_{t=1}^T \begin{bmatrix} \hat{z}_{i,t} \\ \hat{z}_{m,t} \end{bmatrix} \begin{bmatrix} \hat{z}_{i,t} & \hat{z}_{m,t} \end{bmatrix}$$

**2.2 Maximize DCC log-likelihood:**
- Optimize over $(a, b)$ subject to: $a, b \geq 0$, $a + b < 1$
- Initialize: $a_0 = 0.01$, $b_0 = 0.95$
- Use numerical optimizer (L-BFGS-B)
- Output: $\hat{a}, \hat{b}$

**2.3 Compute filtered correlations:**
For $t = 1, \ldots, T$:
- $\mathbf{Q}_t = (1 - \hat{a} - \hat{b})\bar{\mathbf{Q}} + \hat{a} (\mathbf{z}_{t-1}\mathbf{z}_{t-1}^T) + \hat{b} \mathbf{Q}_{t-1}$
- $\rho_t = q_{12,t} / \sqrt{q_{11,t} q_{22,t}}$

**2.4 Current state at time $T$:**
- $\sigma_{i,T}$, $\sigma_{m,T}$, $\mathbf{Q}_T$, $\rho_T$

---

**STEP 3: Monte Carlo Simulation for LRMES**

**3.1 Initialize:**
- Set random seed for reproducibility
- Create arrays: `bank_cumret[N]`, `mkt_cumret[N]`

**3.2 For each simulation $n = 1, \ldots, N$:**

Initialize state:
- $\sigma_{i,0}^{(n)} = \sigma_{i,T}$
- $\sigma_{m,0}^{(n)} = \sigma_{m,T}$
- $\mathbf{Q}_0^{(n)} = \mathbf{Q}_T$
- $r_{i,\text{cum}}^{(n)} = 0$, $r_{m,\text{cum}}^{(n)} = 0$

For $s = 1, \ldots, h = 126$:

**(a) Generate correlated shocks:**
- Compute $\mathbf{R}_{s-1}^{(n)}$ from $\mathbf{Q}_{s-1}^{(n)}$
- Draw $\mathbf{z}_s^{(n)} \sim N(\mathbf{0}, \mathbf{R}_{s-1}^{(n)})$ using Cholesky

**(b) Compute returns:**
- $r_{i,s}^{(n)} = \hat{\mu}_i + \sigma_{i,s-1}^{(n)} z_{i,s}^{(n)}$
- $r_{m,s}^{(n)} = \hat{\mu}_m + \sigma_{m,s-1}^{(n)} z_{m,s}^{(n)}$

**(c) Accumulate:**
- $r_{i,\text{cum}}^{(n)} \mathrel{+}= r_{i,s}^{(n)}$
- $r_{m,\text{cum}}^{(n)} \mathrel{+}= r_{m,s}^{(n)}$

**(d) Update volatilities:**
- $\epsilon_{i,s}^{(n)} = r_{i,s}^{(n)} - \hat{\mu}_i$
- $\sigma_{i,s}^{2,(n)} = \hat{\omega}_i + \hat{\alpha}_i (\epsilon_{i,s}^{(n)})^2 + \hat{\gamma}_i (\epsilon_{i,s}^{(n)})^2 \mathbb{I}[\epsilon_{i,s}^{(n)}<0] + \hat{\beta}_i (\sigma_{i,s-1}^{(n)})^2$
- Similarly for $\sigma_{m,s}^{2,(n)}$

**(e) Update DCC:**
- $\mathbf{Q}_s^{(n)} = (1-\hat{a}-\hat{b})\bar{\mathbf{Q}} + \hat{a}(\mathbf{z}_s^{(n)}\mathbf{z}_s^{(n)T}) + \hat{b}\mathbf{Q}_{s-1}^{(n)}$

Store: `bank_cumret[n] = ` $r_{i,\text{cum}}^{(n)}$, `mkt_cumret[n] = ` $r_{m,\text{cum}}^{(n)}$

**3.3 Identify crisis scenarios:**
```python
crisis_mask = (mkt_cumret < np.log(1 + C))  # C = -0.40
crisis_returns = bank_cumret[crisis_mask]
N_crisis = len(crisis_returns)
```

**3.4 Compute LRMES:**
```python
LRMES = 1 - np.exp(crisis_returns).mean()
LRMES_SE = np.exp(crisis_returns).std() / np.sqrt(N_crisis)
```

---

**STEP 4: Calculate SRISK**

**4.1 Extract balance sheet data at time $t$:**
- $D_i = $ `total_debt`
- $W_i = $ `market_cap`

**4.2 SRISK formula:**
```python
k = 0.08
SRISK = k * D_i - (1 - k) * (1 - LRMES) * W_i
SRISK = max(0, SRISK)  # Floor at zero
```

**4.3 Additional metrics:**
```python
leverage = D_i / W_i
size = W_i
```

---

**STEP 5: Store Results**

```python
results = {
    'bank': bank_name,
    'date': current_date,
    'SRISK': SRISK,
    'LRMES': LRMES,
    'LRMES_SE': LRMES_SE,
    'leverage': leverage,
    'market_cap': W_i,
    'debt': D_i,
    'garch_params': theta_i,
    'dcc_params': (a, b),
    'n_crisis_scenarios': N_crisis
}
```

---

### 7.2 Computational Complexity

**Time complexity:**

- **GARCH estimation:** $O(T \cdot K)$ where $K$ is iterations to convergence  
  Typical: $T = 1000$ days, $K = 100$ iterations → ~100K operations per bank

- **DCC estimation:** $O(T \cdot K)$  
  Similar magnitude

- **Monte Carlo:** $O(N \cdot h \cdot C)$ where $C$ is cost per step  
  Typical: $N = 50,000$, $h = 126$, $C \approx 50$ operations  
  → $50000 \times 126 \times 50 = 315$ million operations per bank

**Total per bank:** ~320 million operations  
**For 40 Gulf banks:** ~13 billion operations  
**Wall time:** ~10-30 minutes on modern CPU (parallelizable across banks)

**Memory:**
- GARCH: $O(T)$ for storing volatilities
- DCC: $O(T)$ for correlations
- Simulation: $O(N \cdot h)$ if storing all paths (can reduce to $O(N)$)

---

## 8. Numerical Implementation Details

### 8.1 Optimization Tips

**GARCH convergence:**
```python
from arch import arch_model
from arch.univariate import ConstantMean, GARCH

# Robust fitting with error handling
def fit_gjr_garch(returns, max_attempts=3):
    starting_values = [
        None,  # Default
        np.array([0.01, 0.05, 0.05, 0.90]),
        np.array([0.05, 0.03, 0.03, 0.92])
    ]
    
    for sv in starting_values:
        try:
            am = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='normal')
            res = am.fit(
                starting_values=sv,
                disp='off',
                options={'maxiter': 1000}
            )
            
            # Check stationarity
            alpha = res.params['alpha[1]']
            gamma = res.params['gamma[1]']
            beta = res.params['beta[1]']
            
            if alpha + gamma/2 + beta < 0.9999:
                return res
        except:
            continue
    
    raise ValueError("GARCH estimation failed")
```

**DCC convergence:**
```python
from scipy.optimize import minimize

def dcc_loglik(params, z1, z2, Q_bar):
    a, b = params
    T = len(z1)
    
    # Initialize
    Q = Q_bar.copy()
    loglik = 0
    
    for t in range(1, T):
        # Update Q
        z = np.array([z1[t-1], z2[t-1]])
        Q = (1 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
        
        # Correlation
        rho = Q[0,1] / np.sqrt(Q[0,0] * Q[1,1])
        
        # Ensure valid correlation
        rho = np.clip(rho, -0.999, 0.999)
        
        # Log-likelihood contribution
        loglik += -0.5 * (np.log(1 - rho**2) + 
                          (z1[t]**2 + z2[t]**2 - 2*rho*z1[t]*z2[t]) / (1 - rho**2) -
                          z1[t]**2 - z2[t]**2)
    
    return -loglik

# Optimize
Q_bar = np.cov(np.column_stack([z1, z2]).T)
result = minimize(
    dcc_loglik,
    x0=[0.01, 0.95],
    args=(z1, z2, Q_bar),
    method='L-BFGS-B',
    bounds=[(0.001, 0.3), (0.7, 0.999)]
)

a_hat, b_hat = result.x
```

### 8.2 Efficient Simulation Code

**Vectorized simulation (faster):**
```python
def simulate_lrmes_vectorized(sigma_i_0, sigma_m_0, Q_0, 
                               params_i, params_m, dcc_params,
                               horizon=126, threshold=-0.40, n_sim=50000):
    
    mu_i, omega_i, alpha_i, gamma_i, beta_i = params_i
    mu_m, omega_m, alpha_m, gamma_m, beta_m = params_m
    a, b = dcc_params
    Q_bar = Q_0  # Approximate
    
    # Storage
    bank_cumret = np.zeros(n_sim)
    mkt_cumret = np.zeros(n_sim)
    
    # Current state
    sigma_i = np.full(n_sim, sigma_i_0)
    sigma_m = np.full(n_sim, sigma_m_0)
    Q = np.repeat(Q_0[np.newaxis, :, :], n_sim, axis=0)
    
    for h in range(horizon):
        # Extract correlations
        rho = Q[:, 0, 1] / np.sqrt(Q[:, 0, 0] * Q[:, 1, 1])
        rho = np.clip(rho, -0.999, 0.999)
        
        # Generate correlated shocks
        u1 = np.random.randn(n_sim)
        u2 = np.random.randn(n_sim)
        z_i = u1
        z_m = rho * u1 + np.sqrt(1 - rho**2) * u2
        
        # Returns
        r_i = mu_i + sigma_i * z_i
        r_m = mu_m + sigma_m * z_m
        
        # Accumulate
        bank_cumret += r_i
        mkt_cumret += r_m
        
        # Update volatilities
        eps_i = sigma_i * z_i
        eps_m = sigma_m * z_m
        
        sigma_i_sq_new = (omega_i + alpha_i * eps_i**2 + 
                          gamma_i * eps_i**2 * (eps_i < 0) + 
                          beta_i * sigma_i**2)
        sigma_m_sq_new = (omega_m + alpha_m * eps_m**2 + 
                          gamma_m * eps_m**2 * (eps_m < 0) + 
                          beta_m * sigma_m**2)
        
        sigma_i = np.sqrt(sigma_i_sq_new)
        sigma_m = np.sqrt(sigma_m_sq_new)
        
        # Update DCC (vectorized)
        z_vec = np.stack([z_i, z_m], axis=1)  # (n_sim, 2)
        Q = ((1 - a - b) * Q_bar + 
             a * np.einsum('ni,nj->nij', z_vec, z_vec) + 
             b * Q)
    
    # LRMES calculation
    crisis_mask = (mkt_cumret < np.log(1 + threshold))
    crisis_returns = bank_cumret[crisis_mask]
    
    if len(crisis_returns) < 100:
        warnings.warn(f"Only {len(crisis_returns)} crisis scenarios")
    
    lrmes = 1 - np.exp(crisis_returns).mean()
    lrmes_se = np.exp(crisis_returns).std() / np.sqrt(len(crisis_returns))
    
    return lrmes, lrmes_se, len(crisis_returns)
```

### 8.3 Parallelization

**Parallel across banks:**
```python
from multiprocessing import Pool
from functools import partial

def process_bank(bank_name, data, config):
    # Full SRISK calculation for one bank
    returns = data['returns'][bank_name]
    benchmark = data['benchmark']
    balance_sheet = data['balance_sheet'][bank_name]
    
    # GARCH-DCC estimation
    model_bank = fit_gjr_garch(returns)
    model_mkt = fit_gjr_garch(benchmark)
    a, b = fit_dcc(model_bank.std_resid, model_mkt.std_resid)
    
    # Simulation
    lrmes, lrmes_se, n_crisis = simulate_lrmes_vectorized(...)
    
    # SRISK
    srisk = calculate_srisk(balance_sheet, lrmes, config['k'])
    
    return {'bank': bank_name, 'srisk': srisk, 'lrmes': lrmes}

# Parallel execution
with Pool(processes=8) as pool:
    process_func = partial(process_bank, data=data, config=config)
    results = pool.map(process_func, bank_names)
```

### 8.4 Numerical Stability

**Avoiding overflow/underflow:**
```python
# When computing exp(cumret), may overflow
# Use log-sum-exp trick

def stable_mean_exp(log_values):
    """Compute mean(exp(log_values)) stably"""
    max_val = np.max(log_values)
    return np.exp(max_val) * np.mean(np.exp(log_values - max_val))

# LRMES calculation
lrmes = 1 - stable_mean_exp(crisis_returns)
```

**Correlation bounds:**
```python
# Ensure correlation in (-1, 1)
rho = np.clip(rho, -0.9999, 0.9999)

# For Cholesky, ensure positive definite
sqrt_term = np.sqrt(np.maximum(1 - rho**2, 1e-8))
```

---

## Summary: Key Statistical Insights

### What SRISK Measures
Expected capital shortfall in crisis = function of size, leverage, and tail risk

### How LRMES Captures Tail Risk
Via simulation that respects:
- Time-varying volatility (GJR-GARCH)
- Dynamic correlation that increases in crises (DCC)
- Path dependence of returns over 6-month horizon

### Why Monte Carlo is Necessary
No closed form for multi-period conditional expectation under GARCH-DCC dynamics

### Statistical Requirements
1. **Sufficient data:** 1000+ days for stable GARCH estimates
2. **Sufficient simulations:** 50,000+ paths to get stable LRMES
3. **Crisis scenarios:** Need 500+ crisis paths for low standard error
4. **Model adequacy:** Diagnostic tests to verify GARCH/DCC fit

### Econometric Sophistication
- Handles volatility clustering (GARCH)
- Captures leverage effect (GJR asymmetry)
- Models time-varying correlation (DCC)
- Respects non-normal tails (via simulation)

### Implementation Rigor
Every step has statistical justification:
- MLE for parameters → asymptotically efficient
- Monte Carlo for expectations → consistent estimator
- Bootstrap for inference → valid confidence intervals

---

**This completes the mathematical and econometric foundation of SRISK. The methodology is theoretically sound, empirically validated, and computationally feasible for regional banking surveillance.**
