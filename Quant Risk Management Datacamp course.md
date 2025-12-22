# Quantitative Risk Management in Python — Concise Notes

## 1. Portfolio Returns & Volatility

**Daily Returns:** `asset_returns = prices.pct_change()`  
**Portfolio Returns:** `portfolio_returns = asset_returns.dot(weights)`

**Covariance Matrix (Annualized):** `cov = asset_returns.cov() * 252`  
**Portfolio Volatility:** `σ_p = √(w'Σw)` → `np.sqrt(weights.T @ cov @ weights)`

**Rolling Volatility:** `returns.rolling(30).std() * np.sqrt(252)`

---

## 2. Frequency Resampling

Convert high-frequency → low-frequency data for factor models:
```python
returns.resample('Q').mean()  # Quarterly average
returns.resample('W').min()   # Weekly minimum
```

---

## 3. Factor Models (OLS Regression)

```python
import statsmodels.api as sm
X = sm.add_constant(risk_factor)
results = sm.OLS(returns, X).fit()
```
Key outputs: coefficients, t-statistics, R-squared, sum-of-squared residuals (SSR)

---

## 4. Efficient Frontier (PyPortfolioOpt)

**Expected Returns:** `mean_historical_return(prices, frequency=252)`  
**Covariance Shrinkage:** `CovarianceShrinkage(prices).ledoit_wolf()` — reduces extreme event overweighting

**Critical Line Algorithm:**
```python
from pypfopt import CLA
ef = CLA(expected_returns, cov_matrix)
ef.min_volatility()
ef.efficient_frontier()  # Returns (ret, vol, weights)
```

---

## 5. Value at Risk (VaR)

VaR = maximum loss at confidence level (e.g., 95% VaR = 5% chance of exceeding)

| Method | Approach |
|--------|----------|
| **Parametric** | Fit distribution → `norm.ppf(0.95, loc=μ, scale=σ)` |
| **Historical Simulation** | `np.quantile(losses, 0.95)` |
| **Monte Carlo** | Simulate correlated paths → compute quantile |

**Risk Exposure:** `P(loss) × Loss Amount` → `0.01 × VaR_99 × Portfolio_Value`

---

## 6. Conditional VaR (CVaR) / Expected Shortfall

CVaR = expected loss *given* losses exceed VaR (captures tail risk)

```python
tail_loss = dist.expect(lambda x: x, loc=μ, scale=σ, lb=VaR_95)
CVaR_95 = (1 / (1 - 0.95)) * tail_loss
```

**CVaR Minimization:**
```python
from pypfopt.efficient_frontier import EfficientCVaR
ec = EfficientCVaR(None, returns)
optimal_weights = ec.min_cvar()
```

---

## 7. Options & Hedging (Black-Scholes)

**Volatility:** `σ = np.sqrt(252) * returns.std()`

**Option Pricing:** `black_scholes(S, X, T, r, sigma, option_type)`  
- S = spot price, X = strike, T = time to maturity, r = risk-free rate

**Delta Hedging:** Compute option delta → create delta-neutral portfolio  
`delta = bs_delta(...)` → hedge ratio = `1/delta`

---

## 8. Parameter Estimation for VaR

| Distribution | Use Case |
|--------------|----------|
| Normal | Symmetric losses |
| Skew-Normal | Asymmetric losses (crisis periods) |
| Student's t | Fat tails |

**Goodness-of-fit:** `anderson(losses)` (Anderson-Darling test)  
**Skewness test:** `skewtest(losses)`

---

## 9. Structural Breaks (Chow Test)

Tests if regression relationship changed at a breakpoint:

```python
# Compute SSR for full period and sub-periods
F = [(SSR_total - (SSR_before + SSR_after)) / k] / [(SSR_before + SSR_after) / (n - 2k)]
```
Compare F-statistic to critical value (e.g., F_crit ≈ 5.85 at 99%)

---

## 10. Extreme Value Theory

**Block Maxima:** `losses.resample('W').max()` — weekly/monthly/quarterly extremes

**GEV Distribution:**
```python
from scipy.stats import genextreme
params = genextreme.fit(weekly_maxima)
VaR_99 = genextreme.ppf(0.99, *params)
CVaR_99 = (1/(1-0.99)) * genextreme.expect(lambda x: x, args=(params[0],), 
                                            loc=params[1], scale=params[2], lb=VaR_99)
```

---

## 11. Kernel Density Estimation (KDE)

Non-parametric approach for fat-tailed distributions:
```python
from scipy.stats import gaussian_kde
kde = gaussian_kde(losses)
kde.pdf(x)  # Evaluate density
kde.resample(size=1000)  # Draw samples
```

---

## 12. Neural Networks for Risk Management

**Asset Price Prediction:**
```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(16, input_dim=3, activation='sigmoid'),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
model.fit(X_train, y_train, epochs=100)
```

**Real-time Portfolio Optimization:** Train NN on rolling window efficient portfolios → predict optimal weights from new asset returns

---

## Key Python Libraries

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations, quantiles |
| `pandas` | Time series, resampling |
| `scipy.stats` | Distributions (norm, t, skewnorm, genextreme), KDE |
| `statsmodels` | OLS regression |
| `pypfopt` | Efficient frontier, CVaR optimization |
| `keras` | Neural networks |
