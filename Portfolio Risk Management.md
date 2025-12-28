# Portfolio Risk Management in Python
## Comprehensive Study Notes

---

# Chapter 1: Financial Returns

## What is Financial Risk?

**Financial risk** is a measure of the uncertainty of future returns. When analyzing returns, risk is essentially the dispersion or variance of those returns. All definitions of financial risk hinge on two fundamental concepts: **returns** and **probability**.

## Types of Financial Returns

Financial returns are derived from stock prices and expressed as percentages in decimal form. There are two common types:

### Discrete (Simple) Returns

**Definition**: The percentage change in price from one period to the next. Discrete returns aggregate across assets, making them ideal for portfolio analysis.

**Formula**:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Where:
- $R_t$ = Return at time t
- $P_t$ = Price at time t
- $P_{t-1}$ = Price at time t-1

### Log (Continuous) Returns

**Definition**: The natural logarithm of the price ratio. Log returns aggregate across time, making them useful for advanced financial models.

**Formula**:
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Key Insight**: For portfolio construction with multiple assets, discrete returns are preferred because they aggregate linearly across assets.

## Calculating Returns in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read in stock price data
StockPrices = pd.read_csv(fpath_csv, parse_dates=['Date'])
StockPrices = StockPrices.sort_values(by='Date')

# Calculate daily returns using the Adjusted Close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Note: First period has NaN because there's no previous price
print(StockPrices.head())
```

## Visualizing Return Distributions

Return distributions help identify outliers. Left-tail outliers represent large negative returns (to be avoided), while right-tail outliers represent positive events like earnings surprises.

```python
# Convert to percentage returns
percent_return = StockPrices['Returns'] * 100

# Drop missing values (first period has no return)
returns_plot = percent_return.dropna()

# Plot histogram
plt.hist(returns_plot, bins=75)
plt.show()
```

---

# Chapter 2: Moments of Distributions

## What are Moments?

**Moments** are common properties of probability distributions that can be analyzed and compared. They describe the shape and characteristics of a distribution.

## First Moment: Mean (μ)

**Definition**: The average outcome of a random sample from the distribution. In finance, this represents the expected return.

**Annualization Formula**:
$$\mu_{annual} = (1 + \mu_{daily})^{252} - 1$$

Where 252 is the typical number of trading days per year.

```python
import numpy as np

# Calculate average daily return
mean_return_daily = np.mean(StockPrices['Returns'])

# Annualize the return (compounding daily for 252 days)
mean_return_annualized = ((1 + mean_return_daily)**252) - 1
```

**Example**: A daily return of 0.03% compounds to approximately 7.85% annually.

## Second Moment: Variance (σ²)

**Definition**: A measure of the variability in outcomes. The square of the standard deviation (volatility).

**Volatility (σ)** is one of the most important concepts in risk management. Higher volatility indicates higher risk. Traders often call it "vol" for short.

```python
# Calculate daily standard deviation (volatility)
sigma_daily = np.std(StockPrices['Returns'])

# Calculate daily variance
variance_daily = sigma_daily**2
```

### Scaling Volatility

**Critical Rule**: Volatility scales with the square root of time, not linearly.

$$\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$$

```python
# Annualize volatility
sigma_annualized = sigma_daily * np.sqrt(252)

# Annualize variance
variance_annualized = sigma_annualized**2
```

**Example**: A daily volatility of 1.93% translates to approximately 30.7% annualized volatility.

## Third Moment: Skewness

**Definition**: A measure of asymmetry—how much a distribution "leans" to the left or right.

- **Negative skewness**: Right-leaning curve (long left tail)
- **Positive skewness**: Left-leaning curve (long right tail)
- **Zero skewness**: Symmetric distribution

In finance, positive skewness is desirable—it indicates a higher probability of large positive returns while negative returns are compressed and predictable.

```python
from scipy.stats import skew

clean_returns = StockPrices['Returns'].dropna()
returns_skewness = skew(clean_returns)

# Normal distribution has skewness ≈ 0
# Financial returns typically show positive skewness
```

## Fourth Moment: Kurtosis

**Definition**: A measure of the thickness of the tails of a distribution—a proxy for the probability of outliers.

- Normal distributions have kurtosis ≈ 3
- **Leptokurtic**: Kurtosis > 3 (fat tails, more outliers)
- **Platykurtic**: Kurtosis < 3 (thin tails, fewer outliers)

Most financial returns are **leptokurtic** with positive excess kurtosis.

**Excess Kurtosis**: Sample kurtosis minus 3. If positive, the distribution has fatter tails than normal.

```python
from scipy.stats import kurtosis

# scipy.stats.kurtosis() returns EXCESS kurtosis by default
excess_kurtosis = kurtosis(clean_returns)

# True kurtosis = excess kurtosis + 3
true_kurtosis = excess_kurtosis + 3
```

**Risk Implication**: High kurtosis means extreme returns (both positive and negative) occur more frequently than expected under a normal distribution. This is dangerous for portfolios if movements are in the wrong direction.

## Testing for Normality

The **Shapiro-Wilk test** provides a statistical method to determine if data is normally distributed.

- **Null hypothesis**: Data is normally distributed
- **Rejection criterion**: p-value ≤ 0.05

```python
from scipy.stats import shapiro

shapiro_results = shapiro(clean_returns.dropna())
p_value = shapiro_results[1]

if p_value <= 0.05:
    print("Reject null hypothesis: Data is non-normal")
else:
    print("Cannot reject null hypothesis: Data may be normal")
```

### Comparing Returns to Normal Distribution

| Property | Normal Distribution | Financial Returns |
|----------|-------------------|-------------------|
| Skewness | ≈ 0 | Typically positive |
| Kurtosis | ≈ 3 | Typically > 3 |
| Outliers | Predictable | More frequent |

---

# Chapter 3: Portfolio Composition and Backtesting

## What is a Portfolio?

**Definition**: A bundle of individual stocks with different weights for each position. The portfolio return is a linear combination of the weights and returns of each position.

$$R_p = \sum_{i=1}^{n} w_i \times R_i$$

Where:
- $R_p$ = Portfolio return
- $w_i$ = Weight of asset i
- $R_i$ = Return of asset i

## Calculating Portfolio Returns in Python

```python
import numpy as np

# Define portfolio weights (must sum to 1.0)
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate weighted returns and sum across assets
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)
```

## Types of Portfolio Weighting

### Equally Weighted Portfolio

**Definition**: Each stock receives the same weight (1/n where n is the number of stocks).

**Characteristics**: Tends to outperform when largest companies are underperforming, since even tiny companies have equal weight as giants like Apple or Amazon.

```python
numstocks = 9
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)

StockReturns['Portfolio_EW'] = StockReturns.iloc[:, 0:numstocks].mul(
    portfolio_weights_ew, axis=1
).sum(axis=1)
```

### Market Capitalization Weighted Portfolio

**Definition**: Weight each stock by its market cap relative to the total market cap of the portfolio.

**Market Capitalization**: The total value of all outstanding publicly traded shares of a company.

$$w_n = \frac{MarketCap_n}{\sum_{i=1}^{n} MarketCap_i}$$

**Characteristics**: Tends to outperform when large companies are doing well. The S&P 500 index follows this methodology.

```python
# Market caps in billions
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 
                                    356.94, 268.88, 331.57, 246.09])

# Calculate market cap weights
mcap_weights = market_capitalizations / sum(market_capitalizations)

StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(
    mcap_weights, axis=1
).sum(axis=1)
```

## Cumulative Returns

**Definition**: The total growth of an investment over time, accounting for compounding.

```python
# Calculate cumulative returns
CumulativeReturns = ((1 + StockReturns["Portfolio"]).cumprod() - 1)
CumulativeReturns.plot()
plt.show()
```

---

# Chapter 4: Correlation and Covariance

## The Foundation of Portfolio Theory

Modern portfolio theory teaches that you can build a portfolio with less risk than any of its underlying assets alone. The key lies in **correlation** and **covariance**.

## Pearson Correlation

**Definition**: A measure of the linear relationship between the returns of two variables, ranging from -1 to +1.

| Value | Interpretation |
|-------|---------------|
| +1 | Perfect positive correlation |
| 0 | No correlation |
| -1 | Perfect negative correlation |

```python
import seaborn as sns

# Calculate correlation matrix
correlation_matrix = StockReturns.corr()

# Visualize with heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidths=0.3)
plt.show()
```

**Key Properties**:
- Diagonals are always 1 (perfect correlation with self)
- Matrix is symmetric (correlation is bidirectional)

## Covariance Matrix

**Definition**: Measures the joint variability of two random variables. Covariance is the un-normalized form of correlation and is essential for portfolio optimization.

**Relationship**: Correlation is a normalized measure of covariance.

```python
# Calculate covariance matrix
cov_mat = StockReturns.cov()

# Annualize (variance scales linearly with time)
cov_mat_annual = cov_mat * 252
```

**Key Insight**: The covariance matrix contains information about both variance (diagonal elements) and the relationships between assets (off-diagonal elements).

## Portfolio Standard Deviation

Using the covariance matrix, portfolio volatility can be calculated efficiently:

$$\sigma_p = \sqrt{w^T \cdot \Sigma \cdot w}$$

Where:
- $\sigma_p$ = Portfolio volatility
- $w$ = Vector of portfolio weights
- $\Sigma$ = Covariance matrix
- $w^T$ = Transpose of weight vector

```python
# Calculate portfolio volatility
portfolio_volatility = np.sqrt(
    np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights))
)
```

### Matrix Operations Explained

**Transpose (T)**: Flipping a matrix by 90 degrees—rows become columns.

**Dot Product**: The sum of element-wise products of two arrays.

---

# Chapter 5: Markowitz Portfolios

## Modern Portfolio Theory

**Harry Markowitz** pioneered modern portfolio theory in 1952, establishing that there exists an **efficient frontier** of portfolios, each offering the highest expected return for a given level of risk.

## The Efficient Frontier

**Definition**: The set of optimal portfolios that offer the highest expected return for each level of risk. All portfolios below the frontier are sub-optimal.

## Sharpe Ratio

**Definition**: A measure of risk-adjusted return—how much return an investor receives per unit of risk.

$$S = \frac{R_p - R_f}{\sigma_p}$$

Where:
- $S$ = Sharpe ratio
- $R_p$ = Portfolio return
- $R_f$ = Risk-free rate
- $\sigma_p$ = Portfolio volatility

Higher Sharpe ratios indicate better risk-adjusted performance.

```python
risk_free = 0  # Assume 0 for simplicity

RandomPortfolios['Sharpe'] = (
    RandomPortfolios['Returns'] - risk_free
) / RandomPortfolios['Volatility']
```

## Key Markowitz Portfolios

### Maximum Sharpe Ratio (MSR) Portfolio

**Definition**: The tangency portfolio on the efficient frontier with the highest Sharpe ratio. Crosses the capital allocation line according to CAPM theory.

**Characteristics**: Often erratic—high historical Sharpe doesn't guarantee future performance.

```python
# Find MSR portfolio
sorted_portfolios = RandomPortfolios.sort_values(by=['Sharpe'], ascending=False)
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]
MSR_weights_array = np.array(MSR_weights)

StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(
    MSR_weights_array, axis=1
).sum(axis=1)
```

### Global Minimum Volatility (GMV) Portfolio

**Definition**: The portfolio with the lowest possible risk (standard deviation) on the efficient frontier.

**Characteristics**: Tends to be more stable over time than MSR. Volatility and correlations are more stable than returns, so GMV often outperforms out-of-sample.

```python
# Find GMV portfolio
sorted_portfolios = RandomPortfolios.sort_values(by=['Volatility'], ascending=True)
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]
GMV_weights_array = np.array(GMV_weights)

StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(
    GMV_weights_array, axis=1
).sum(axis=1)
```

**Important Caveat**: Past performance is not a guarantee of future returns. Sharpe ratios change dramatically over time.

---

# Chapter 6: Factor Models

## What is Factor Analysis?

**Definition**: The practice of using known factors (such as market returns, size, or value) as independent variables to explain portfolio returns.

## Excess Returns

**Definition**: Portfolio return minus the risk-free rate of return.

$$R_{excess} = R_p - R_f$$

This adjustment is crucial for comparing investments across regions with different risk-free rates.

```python
FamaFrenchData['Portfolio_Excess'] = (
    FamaFrenchData['Portfolio'] - FamaFrenchData['RF']
)
```

## Capital Asset Pricing Model (CAPM)

**Definition**: The foundational asset pricing model that explains returns using a single factor: market beta.

$$R_P - R_f = \beta_M (R_M - R_f) + \alpha$$

Where:
- $R_P$ = Portfolio return
- $R_f$ = Risk-free rate
- $\beta_M$ = Market beta (exposure to broad market)
- $R_M$ = Market return
- $\alpha$ = Unexplained return (alpha)

### Beta (β)

**Definition**: A measure of systematic risk—the exposure to the broad market benchmark. High beta means high market exposure.

**Calculation using Covariance**:
$$\beta = \frac{Cov(R_P, R_B)}{Var(R_B)}$$

```python
# Calculate beta using covariance method
covariance_matrix = FamaFrenchData[['Portfolio_Excess', 'Market_Excess']].cov()
covariance_coefficient = covariance_matrix.iloc[0, 1]
benchmark_variance = FamaFrenchData['Market_Excess'].var()

portfolio_beta = covariance_coefficient / benchmark_variance
```

**Calculation using Regression**:

```python
import statsmodels.formula.api as smf

CAPM_model = smf.ols(
    formula='Portfolio_Excess ~ Market_Excess', 
    data=FamaFrenchData
)
CAPM_fit = CAPM_model.fit()

regression_beta = CAPM_fit.params['Market_Excess']
adj_rsquared = CAPM_fit.rsquared_adj
```

**Interpretation**: A beta of 0.97 means for every 1% rise (or fall) in the market, the portfolio rises (falls) approximately 0.97%.

### R-Squared and Adjusted R-Squared

**R-squared**: The percentage of variance in returns explained by the model (0-100%).

**Adjusted R-squared**: Penalizes for added variables to avoid overfitting. Use this for comparing models.

## Fama-French 3-Factor Model

**Definition**: Extends CAPM by adding two additional factors developed by Eugene Fama and Kenneth French in the 1990s.

$$R_P = R_f + \beta_M(R_M - R_f) + b_{SMB} \cdot SMB + b_{HML} \cdot HML + \alpha$$

### Additional Factors

**SMB (Small Minus Big)**: The size factor—returns of small-cap stocks minus large-cap stocks. Small stocks tend to outperform large stocks over time.

**HML (High Minus Low)**: The value factor—returns of high book-to-market (value) stocks minus low book-to-market (growth) stocks.

```python
FamaFrench_model = smf.ols(
    formula='Portfolio_Excess ~ Market_Excess + SMB + HML', 
    data=FamaFrenchData
)
FamaFrench_fit = FamaFrench_model.fit()

regression_adj_rsq = FamaFrench_fit.rsquared_adj
```

**Performance**: The Fama-French model explains over 90% of portfolio variance on average, compared to ~70% for CAPM.

### Extracting Coefficients and P-Values

```python
# P-values for statistical significance
smb_pval = FamaFrench_fit.pvalues['SMB']

# Coefficients (factor exposures)
smb_coeff = FamaFrench_fit.params['SMB']

# P-value < 0.05 indicates statistical significance
```

**Interpretation Examples**:
- Negative SMB coefficient = exposure to large-cap stocks
- Positive HML coefficient = exposure to value stocks

## Alpha (α)

**Definition**: The unexplained performance after accounting for all factors. Positive alpha is interpreted as outperformance due to skill, luck, or timing.

```python
portfolio_alpha = FamaFrench_fit.params['Intercept']
portfolio_alpha_annualized = ((1 + portfolio_alpha)**252) - 1
```

### The Efficient Market Hypothesis

One school of thought: Alpha is simply a missing factor. When all economic factors are discovered, all returns can be explained.

Alternative view: Some unexplainable performance exists due to skill, timing, or luck.

## Fama-French 5-Factor Model

**Definition**: The 2015 extension adding two more factors:

$$R_P = R_f + \beta_M(R_M - R_f) + b_{SMB} \cdot SMB + b_{HML} \cdot HML + b_{RMW} \cdot RMW + b_{CMA} \cdot CMA + \alpha$$

**RMW (Robust Minus Weak)**: Profitability factor—returns of companies with high operating profitability minus those with low profitability.

**CMA (Conservative Minus Aggressive)**: Investment factor—returns of conservative investors minus aggressive investors.

```python
FamaFrench5_model = smf.ols(
    formula='Portfolio_Excess ~ Market_Excess + SMB + HML + RMW + CMA', 
    data=FamaFrenchData
)
FamaFrench5_fit = FamaFrench5_model.fit()
```

---

# Chapter 7: Tail Risk Estimation

## What is Tail Risk?

**Definition**: The risk of extreme outcomes in the tails of the return distribution, particularly the left tail (large negative returns).

## Historical Drawdown

**Definition**: The percentage loss from the highest cumulative point (peak-to-trough decline).

$$Drawdown = \frac{r_t}{RM} - 1$$

Where:
- $r_t$ = Cumulative return at time t
- $RM$ = Running maximum

Great investments have minimal drawdown, showing consistent growth.

```python
# Calculate running maximum
running_max = np.maximum.accumulate(cum_rets)

# Ensure running max never drops below 1 (starting point)
running_max[running_max < 1] = 1

# Calculate drawdown
drawdown = (cum_rets) / running_max - 1

drawdown.plot()
plt.show()
```

## Value at Risk (VaR)

**Definition**: A measure of the minimum loss expected in the worst X% of scenarios over a given time period.

**Common Quantiles**: VaR(95), VaR(99), VaR(99.9)

**Interpretation**: VaR(95) = -2.3% means "In the worst 5% of scenarios, losses will exceed 2.3%." Equivalently: "95% certain losses won't exceed 2.3%."

### Historical VaR

Uses actual historical returns to estimate percentiles.

```python
# VaR(95) - the 5th percentile of returns
var_95 = np.percentile(StockReturns_perc, 5)

# Visualize
plt.hist(sorted(StockReturns_perc), density=True)
plt.axvline(x=var_95, color='r', label=f"VaR 95: {var_95:.2f}%")
plt.show()
```

### Parametric VaR

Assumes a probability distribution (typically normal) rather than using historical values directly.

```python
from scipy.stats import norm

mu = np.mean(StockReturns)
vol = np.std(StockReturns)
confidence_level = 0.05  # For VaR(95)

var_95 = norm.ppf(confidence_level, mu, vol)
```

**Advantage**: Allows for a continuous range of outcomes, not limited to historical events.

## Conditional Value at Risk (CVaR) / Expected Shortfall

**Definition**: The expected (average) loss in the worst X% of scenarios. Always more extreme than VaR at the same quantile.

**Interpretation**: CVaR(95) = -2.5% means "In the worst 5% of cases, average losses will be 2.5%."

```python
# CVaR is the mean of all returns worse than VaR
cvar_95 = StockReturns_perc[StockReturns_perc <= var_95].mean()
```

## Scaling VaR Over Time

VaR for a single day can be scaled to longer horizons:

$$VaR(95)_{t\ days} = VaR(95)_{1\ day} \times \sqrt{t}$$

```python
# Scale 1-day VaR to multiple days
forecasted_values = np.empty([100, 2])

for i in range(0, 100):
    forecasted_values[i, 0] = i
    forecasted_values[i, 1] = var_95 * np.sqrt(i + 1)
```

**Example**: A 1-day VaR(95) of -2.35% becomes approximately -5.25% over 5 days.

### Comparing VaR Quantiles

| Metric | 90% | 95% | 99% |
|--------|-----|-----|-----|
| VaR | Less extreme | Moderate | More extreme |
| CVaR | Always ≥ VaR(90) | Always ≥ VaR(95) | Always ≥ VaR(99) |

---

# Chapter 8: Monte Carlo Simulations

## Random Walks in Finance

**Definition**: A mathematical model for random (stochastic) movements, widely used to simulate stock prices and other financial variables.

**Key Insight**: Even though stock movements appear random, they can be modeled mathematically to forecast a range of possible outcomes.

## Building a Random Walk

```python
# Parameters from historical data
mu = np.mean(StockReturns)  # Average return
vol = np.std(StockReturns)  # Volatility
T = 252  # Trading days (1 year)
S0 = 10  # Starting price

# Generate random returns from normal distribution
rand_rets = np.random.normal(mu, vol, T) + 1

# Create price path
forecasted_values = S0 * rand_rets.cumprod()

plt.plot(range(0, T), forecasted_values)
plt.show()
```

**Important**: Each run produces a different outcome due to the random sampling.

## What is Monte Carlo Simulation?

**Definition**: A technique that generates thousands or millions of random simulations to create a comprehensive range of possible outcomes for analysis.

**Applications**:
- Option pricing
- Portfolio optimization
- Risk management
- Trading strategy backtesting

## Implementing Monte Carlo Simulations

```python
# Generate 100 Monte Carlo paths
for i in range(0, 100):
    rand_rets = np.random.normal(mu, vol, T) + 1
    forecasted_values = S0 * rand_rets.cumprod()
    plt.plot(range(T), forecasted_values)

plt.show()
```

## Monte Carlo VaR

Use aggregated simulation returns to calculate parametric VaR:

```python
sim_returns = []

for i in range(100):
    rand_rets = np.random.normal(mu, vol, T)
    sim_returns.append(rand_rets)

# Calculate VaR(99) from simulated returns
var_99 = np.percentile(sim_returns, 1)
print(f"Parametric VaR(99): {round(100*var_99, 2)}%")
```

**Benefits**:
- Creates unlimited scenarios
- Can model outcomes that haven't occurred historically
- Allows parameter tweaking for sensitivity analysis

**Trade-offs**:
- More simulations = more stable estimates but longer computation
- Results vary between runs unless a seed is set

---

# Quick Reference: Key Formulas

## Returns and Risk

| Concept | Formula |
|---------|---------|
| Simple Return | $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ |
| Annualized Return | $(1 + R_{daily})^{252} - 1$ |
| Annualized Volatility | $\sigma_{daily} \times \sqrt{252}$ |
| Portfolio Return | $\sum w_i R_i$ |
| Portfolio Volatility | $\sqrt{w^T \Sigma w}$ |

## Risk Metrics

| Concept | Formula |
|---------|---------|
| Sharpe Ratio | $\frac{R_p - R_f}{\sigma_p}$ |
| Beta | $\frac{Cov(R_p, R_m)}{Var(R_m)}$ |
| VaR Scaling | $VaR_{t\ days} = VaR_{1\ day} \times \sqrt{t}$ |
| Drawdown | $\frac{r_t}{RM} - 1$ |

## Key Python Functions

| Task | Function |
|------|----------|
| Daily returns | `df['Price'].pct_change()` |
| Mean | `np.mean(returns)` |
| Std deviation | `np.std(returns)` |
| Skewness | `scipy.stats.skew(returns)` |
| Kurtosis | `scipy.stats.kurtosis(returns)` |
| Correlation matrix | `df.corr()` |
| Covariance matrix | `df.cov()` |
| Percentile (VaR) | `np.percentile(returns, q)` |
| Normal PPF (VaR) | `scipy.stats.norm.ppf(q, mu, sigma)` |
| OLS Regression | `smf.ols(formula, data).fit()` |

---

# Summary: Key Takeaways

1. **Financial risk** is the uncertainty of returns, measured through variance, volatility, and distribution moments.

2. **Returns are typically non-normal** with positive skewness and excess kurtosis (fat tails).

3. **Portfolio diversification** can reduce risk below that of any individual asset through low correlation between holdings.

4. **The efficient frontier** represents optimal risk-return combinations; the GMV portfolio often outperforms out-of-sample.

5. **Factor models** (CAPM, Fama-French) explain portfolio returns through systematic exposures; unexplained returns are called alpha.

6. **Tail risk** can be measured through VaR (minimum loss at a confidence level) and CVaR (expected loss in worst cases).

7. **Monte Carlo simulations** allow comprehensive risk analysis by generating thousands of possible scenarios.


