# Quantitative Risk Management in Python

## Core Concepts and Advanced Techniques

This document summarizes the key quantitative risk management concepts, with emphasis on sophisticated techniques used in portfolio risk analysis, VaR estimation, and extreme value modeling.

---

## 1. Portfolio Volatility and Covariance

### Portfolio Variance Formula

Portfolio variance quantifies the total risk of a multi-asset portfolio by accounting for both individual asset volatilities and their co-movements. For a portfolio with weights **w** and covariance matrix **Σ**, the portfolio variance is computed as a quadratic form:

$$\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$$

The annualized volatility is then $\sigma_p \sqrt{252}$, where 252 represents trading days per year. This measure captures how assets move together—highly correlated assets provide less diversification benefit.

### Covariance Shrinkage (Ledoit-Wolf)

Sample covariance matrices suffer from estimation error, particularly when the number of observations is small relative to the number of assets. The **Ledoit-Wolf shrinkage estimator** addresses this by blending the sample covariance with a structured target (often the identity matrix), reducing extreme eigenvalues that arise from sampling noise. This produces a more stable, efficient covariance estimate that improves portfolio optimization outcomes.

```python
from pypfopt.risk_models import CovarianceShrinkage
cs = CovarianceShrinkage(prices)
efficient_cov = cs.ledoit_wolf()
```

---

## 2. Efficient Frontier and Portfolio Optimization

### Modern Portfolio Theory

The efficient frontier represents the set of portfolios offering the highest expected return for each level of risk. Points below this frontier are suboptimal because an investor could achieve higher returns for the same risk. The **Critical Line Algorithm (CLA)** is a specialized method for tracing the entire efficient frontier analytically, avoiding the numerical instabilities of general-purpose optimizers.

### CVaR Minimization

Rather than minimizing variance (which treats upside and downside deviations symmetrically), **CVaR minimization** directly targets tail risk. This approach constructs portfolios that minimize expected losses in the worst α% of scenarios, making it particularly relevant for risk-averse investors or regulatory compliance.

```python
from pypfopt.efficient_frontier import EfficientCVaR
ec = EfficientCVaR(None, returns)
optimal_weights = ec.min_cvar()  # Minimizes 95% CVaR by default
```

---

## 3. Value at Risk (VaR)

### Definition and Interpretation

Value at Risk represents the maximum loss not exceeded with a given confidence level over a specified time horizon. Formally, the α% VaR is the α-th quantile of the loss distribution:

$$\text{VaR}_\alpha = F_L^{-1}(\alpha)$$

where $F_L^{-1}$ is the inverse CDF (quantile function) of the loss distribution. A 95% VaR of 3% means there is only a 5% probability of losing more than 3% of portfolio value on any given day.

### Parametric VaR (Normal Distribution)

When losses are assumed normally distributed with mean μ and standard deviation σ, VaR has a closed-form solution using the standard normal quantile:

$$\text{VaR}_\alpha = \mu + \sigma \cdot \Phi^{-1}(\alpha)$$

However, the **Anderson-Darling test** often reveals that financial returns violate normality assumptions, exhibiting heavier tails and skewness—particularly during crisis periods.

### VaR with Student's t-Distribution

The Student's t-distribution captures fat tails better than the Normal, making it more appropriate for financial losses. With ν degrees of freedom, the t-distribution has heavier tails (higher kurtosis), and VaR is computed as:

$$\text{VaR}_\alpha = \mu + \sigma \cdot t_\nu^{-1}(\alpha)$$

Rolling window estimation allows the VaR to adapt to changing market conditions, with parameters (μ, σ) re-estimated at each time step.

### Skewed Normal Distribution

Financial losses often exhibit asymmetry—during crises, large negative returns become more probable than large positive returns. The **skew-normal distribution** extends the normal by adding a shape parameter α that captures this asymmetry. The `skewtest()` from scipy.stats can validate whether skewness is statistically significant before fitting.

---

## 4. Conditional Value at Risk (CVaR) / Expected Shortfall

### Definition

While VaR answers "what is the worst loss at confidence level α?", CVaR answers "if we exceed VaR, what is the expected loss?" CVaR is the conditional expectation of losses exceeding the VaR threshold:

$$\text{CVaR}_\alpha = \mathbb{E}[L \mid L > \text{VaR}_\alpha] = \frac{1}{1-\alpha} \int_{\text{VaR}_\alpha}^{\infty} L \cdot f_L(L) \, dL$$

CVaR is a **coherent risk measure** (satisfying subadditivity, positive homogeneity, monotonicity, and translation invariance), whereas VaR is not. This makes CVaR preferable for portfolio optimization and regulatory purposes.

### Risk Exposure

Risk exposure combines probability of loss with loss magnitude. For a portfolio of value V, the expected loss at confidence α is:

$$\text{Risk Exposure} = (1 - \alpha) \cdot \text{VaR}_\alpha \cdot V$$

CVaR directly provides risk exposure in monetary terms when multiplied by portfolio value, since it already incorporates tail probability.

---

## 5. VaR Estimation Methods

### Historical Simulation

Historical simulation assumes future losses follow the same distribution as past losses. VaR is simply the empirical quantile of historical loss data:

$$\text{VaR}_\alpha^{HS} = \text{Quantile}_\alpha(\{L_1, L_2, \ldots, L_T\})$$

This method is non-parametric and captures fat tails naturally, but assumes stationarity—a problematic assumption when market regimes shift (e.g., pre-crisis vs. crisis periods).

### Monte Carlo Simulation

Monte Carlo generates thousands of simulated portfolio paths using estimated mean returns μ and covariance matrix Σ. For correlated assets, random shocks are generated via Cholesky decomposition:

$$\mathbf{r}_t = \boldsymbol{\mu} \cdot \Delta t + \mathbf{L} \cdot \mathbf{z} \cdot \sqrt{\Delta t}$$

where **L** is the Cholesky factor of Σ (i.e., $\Sigma = \mathbf{L}\mathbf{L}^T$) and **z** is a vector of independent standard normal draws. VaR is then the empirical quantile of simulated portfolio losses. This method is flexible and can incorporate complex dynamics, but is computationally intensive.

---

## 6. Extreme Value Theory (EVT)

### Block Maxima Method

EVT focuses on the tail behavior of distributions. The **block maxima** approach partitions data into non-overlapping blocks (e.g., weekly, monthly) and retains only the maximum loss from each block. The Fisher-Tippett-Gnedenko theorem guarantees that these maxima converge to a **Generalized Extreme Value (GEV)** distribution regardless of the underlying loss distribution.

### GEV Distribution

The GEV distribution has CDF:

$$F(x; \xi, \mu, \sigma) = \exp\left(-\left[1 + \xi\left(\frac{x-\mu}{\sigma}\right)\right]^{-1/\xi}\right)$$

where μ is location, σ is scale, and ξ is the shape parameter. The shape parameter determines tail behavior: ξ > 0 (Fréchet) indicates heavy tails, ξ = 0 (Gumbel) indicates light tails, and ξ < 0 (Weibull) indicates bounded tails. Financial data typically exhibits ξ > 0.

```python
from scipy.stats import genextreme
weekly_maxima = losses.resample("W").max()
params = genextreme.fit(weekly_maxima)
VaR_99 = genextreme.ppf(0.99, *params)
```

---

## 7. Options and Hedging

### Black-Scholes Option Pricing

The Black-Scholes formula provides the theoretical price of European options. For a call option:

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

where:
- $d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
- $d_2 = d_1 - \sigma\sqrt{T}$

Here S is spot price, K is strike price, r is risk-free rate, T is time to maturity, σ is volatility, and N(·) is the standard normal CDF. Volatility σ is the key risk input—doubling volatility significantly increases option value.

### Delta Hedging

The option **delta** (Δ) measures sensitivity of option price to changes in the underlying asset price:

$$\Delta = \frac{\partial V}{\partial S}$$

A **delta-neutral portfolio** holds 1/Δ shares of stock for each option, creating a position whose value is locally invariant to small price movements. This hedge must be continuously rebalanced as delta changes with the underlying price (gamma risk).

---

## 8. Structural Break Analysis

### Chow Test

The **Chow test** determines whether regression coefficients differ significantly across two sub-periods, indicating a structural break. The test statistic is:

$$F = \frac{(\text{SSR}_{total} - (\text{SSR}_1 + \text{SSR}_2)) / k}{(\text{SSR}_1 + \text{SSR}_2) / (n - 2k)}$$

where SSR denotes sum of squared residuals, k is the number of parameters, and n is total observations. If F exceeds the critical value (e.g., ~5.85 at 99% confidence for typical df), we reject the null hypothesis of stable parameters, confirming a structural change—critical for understanding regime shifts like the 2008 financial crisis.

---

## 9. Kernel Density Estimation (KDE)

### Non-Parametric Density Estimation

KDE estimates the probability density function without assuming a parametric form. The **Gaussian KDE** places a kernel (typically Gaussian) at each data point and sums them:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where h is the bandwidth controlling smoothness. KDE adapts to the data's actual shape, capturing fat tails and multimodality that parametric distributions might miss. For risk management, the distribution yielding the highest CVaR provides the most conservative loss reserve.

---

## 10. Neural Networks for Risk Management

### Asset Price Prediction

Neural networks can learn complex, non-linear relationships between asset prices. A typical architecture uses multiple dense layers with activation functions:

```python
model = Sequential()
model.add(Dense(16, input_dim=3, activation='sigmoid'))  # Hidden layer 1
model.add(Dense(8, activation='relu'))                    # Hidden layer 2
model.add(Dense(1))                                       # Output layer
model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
```

The network learns to predict one asset's price from others, capturing cross-asset dependencies that linear models might miss.

### Real-Time Portfolio Optimization

A powerful application trains neural networks to predict optimal portfolio weights from recent asset returns. The network learns the mapping from a rolling window of returns to minimum-volatility portfolio weights, enabling near-instantaneous portfolio rebalancing without solving the full optimization problem at each time step. This is crucial for high-frequency risk management where computational speed matters.

---

## Summary: Key Risk Measures Comparison

| Measure | Strengths | Limitations |
|---------|-----------|-------------|
| **Volatility** | Simple, intuitive | Symmetric (penalizes gains equally) |
| **VaR** | Regulatory standard, intuitive | Not coherent, ignores tail severity |
| **CVaR** | Coherent, captures tail risk | Harder to backtest |
| **EVT/GEV** | Robust for extreme events | Requires sufficient tail data |

---

## Python Libraries Used

- `numpy`, `pandas`: Data manipulation and numerical computation
- `scipy.stats`: Statistical distributions (norm, t, skewnorm, genextreme)
- `statsmodels`: OLS regression, structural break tests
- `pypfopt`: Portfolio optimization (efficient frontier, CVaR minimization)
- `keras/tensorflow`: Neural network implementation for predictive models
