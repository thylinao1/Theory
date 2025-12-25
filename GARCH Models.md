# GARCH Models in Python: Comprehensive Notes

## Chapter 1: Introduction to Volatility and GARCH

### 1.1 What is Volatility?

Volatility measures the degree of variation in a financial instrument's price over time. It quantifies uncertainty and risk—higher volatility means prices swing more dramatically, while lower volatility indicates more stable, predictable movements.

In quantitative finance, volatility is typically measured as the standard deviation of returns. Returns are calculated as percentage price changes:

$$\text{Return}_t = \frac{P_t - P_{t-1}}{P_{t-1}} \times 100$$

```python
# Calculate daily returns as percentage price changes
sp_price['Return'] = 100 * (sp_price['Close'].pct_change())
```

### 1.2 Converting Volatility Across Time Horizons

Volatility scales with the square root of time. This relationship comes from the properties of variance: if daily returns are independent, then the variance of n-day returns equals n times the daily variance. Taking square roots gives us the volatility scaling rule.

$$\sigma_{\text{period}} = \sigma_{\text{daily}} \times \sqrt{\text{number of days}}$$

**Standard Conversions:**

| Conversion | Multiplier | Typical Days |
|------------|-----------|--------------|
| Daily → Monthly | √21 ≈ 4.58 | 21 trading days |
| Daily → Annual | √252 ≈ 15.87 | 252 trading days |
| Monthly → Annual | √12 ≈ 3.46 | 12 months |

```python
import math

# Daily volatility
std_daily = sp_data['Return'].std()

# Convert to monthly (21 trading days)
std_monthly = math.sqrt(21) * std_daily

# Convert to annual (252 trading days)
std_annual = math.sqrt(252) * std_daily
```

### 1.3 Volatility Clustering

One of the most important empirical observations in financial markets is that volatility clusters. Large price movements tend to be followed by large movements (of either sign), and small movements tend to be followed by small movements. This creates periods of high volatility and periods of low volatility.

Standard models like simple moving averages or constant-volatility assumptions fail to capture this phenomenon. GARCH models were specifically designed to model this clustering behavior by making today's volatility depend on yesterday's volatility and yesterday's shocks.

**Visual Signature:** When you plot financial returns over time, you'll notice that the "envelope" of returns expands and contracts. During crises (2008, 2020), the envelope widens dramatically; during calm periods, it narrows.

---

## Chapter 2: ARCH and GARCH Model Structure

### 2.1 The ARCH Model

ARCH stands for **Autoregressive Conditional Heteroskedasticity**, introduced by Robert Engle in 1982 (Nobel Prize 2003).

"Heteroskedasticity" means non-constant variance. "Conditional" means the variance depends on past information. "Autoregressive" means the variance depends on its own past values.

**ARCH(1) Model:**

The return follows:
$$r_t = \mu + \varepsilon_t$$

Where the residual has time-varying variance:
$$\varepsilon_t = \sigma_t \cdot z_t, \quad z_t \sim N(0,1)$$

The conditional variance evolves as:
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2$$

**Interpretation of Parameters:**
- **ω (omega):** The baseline or long-run variance. Must be positive.
- **α (alpha):** The ARCH coefficient. Controls how much yesterday's squared shock affects today's variance. Must be non-negative, and α < 1 for stationarity.

**How ARCH Captures Clustering:** When a large shock occurs (large |ε_{t-1}|), the squared term ε²_{t-1} becomes large, which increases σ²_t. Higher variance means the next shock is likely to be larger in magnitude, perpetuating the high-volatility period.

### 2.2 The GARCH Model

GARCH stands for **Generalized ARCH**, introduced by Tim Bollerslev in 1986. It extends ARCH by adding a "memory" component—today's variance also depends on yesterday's variance, not just yesterday's shock.

**GARCH(1,1) Model:**

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Interpretation of Parameters:**
- **ω (omega):** Baseline variance contribution.
- **α (alpha):** The ARCH term. Controls the immediate impact of new shocks on variance.
- **β (beta):** The GARCH term. Controls the persistence of past variance. Higher β means volatility persists longer.

**Stationarity Condition:** For the process to be stationary (variance doesn't explode to infinity), we need α + β < 1.

**The Role of β:** The β parameter creates "volatility memory." If β = 0.85, then 85% of yesterday's variance carries forward to today. This is why GARCH can capture prolonged high or low volatility regimes better than ARCH alone.

```python
# Simulate ARCH(1) - no beta term
arch_resid, arch_variance = simulate_GARCH(n=200, omega=0.1, alpha=0.7)

# Simulate GARCH(1,1) - includes beta term
garch_resid, garch_variance = simulate_GARCH(n=200, omega=0.1, alpha=0.7, beta=0.1)
```

### 2.3 Understanding the Model Components

**Return Equation:**
$$r_t = \mu + \varepsilon_t$$

This says returns equal an expected mean (μ) plus a random shock (ε_t). The shock is the "surprise" or unpredictable component.

**Variance Equation:**
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

This says today's variance is a weighted combination of:
1. A constant baseline (ω)
2. Yesterday's squared shock (α × ε²_{t-1}) — the "news" or "innovation" impact
3. Yesterday's variance (β × σ²_{t-1}) — the "persistence" component

**Standardized Residuals:**
$$z_t = \frac{\varepsilon_t}{\sigma_t} = \frac{r_t - \mu}{\sigma_t}$$

Standardized residuals strip out the predictable volatility pattern. If the model is correct, z_t should look like white noise with constant variance (typically 1) and no autocorrelation.

### 2.4 The Notation p and q

In GARCH(p, q):
- **p** = number of lagged variance terms (GARCH terms, the β coefficients)
- **q** = number of lagged squared residual terms (ARCH terms, the α coefficients)

**GARCH(1,1)** is by far the most common specification and often performs as well as more complex models. The "1,1" means one lag of variance and one lag of squared residuals.

---

## Chapter 3: Implementing GARCH in Python

### 3.1 The arch Package

Python's `arch` package provides a comprehensive toolkit for GARCH modeling.

```python
from arch import arch_model

# Define a GARCH(1,1) model
model = arch_model(returns, 
                   p=1,              # Number of GARCH lags
                   q=1,              # Number of ARCH lags
                   mean='constant',  # Mean model specification
                   vol='GARCH',      # Volatility model type
                   dist='normal')    # Distribution assumption

# Fit the model
result = model.fit(update_freq=4)

# View summary
print(result.summary())

# Plot fitted results
result.plot()
```

### 3.2 Key Model Outputs

After fitting a GARCH model, you can access several important outputs:

```python
# Model parameters (omega, alpha, beta, etc.)
result.params

# Standard errors of parameters
result.std_err

# P-values for parameter significance
result.pvalues

# T-statistics for parameters
result.tvalues

# Conditional volatility (time series of σ_t)
result.conditional_volatility

# Raw residuals (ε_t = r_t - μ)
result.resid

# Log-likelihood value
result.loglikelihood

# Information criteria
result.aic
result.bic
```

### 3.3 Standardized Residuals

Standardized residuals are crucial for model validation. They represent the "pure shocks" after accounting for the time-varying volatility.

```python
# Calculate standardized residuals
gm_resid = result.resid
gm_std = result.conditional_volatility
std_resid = gm_resid / gm_std
```

If the GARCH model is correctly specified, standardized residuals should:
1. Have approximately constant variance (around 1)
2. Show no autocorrelation
3. Resemble the assumed distribution (normal, t, etc.)

### 3.4 Making Forecasts

GARCH models can forecast future variance (and hence volatility).

```python
# Fit the model
result = model.fit()

# Forecast 5 periods ahead
forecast = result.forecast(horizon=5)

# Access variance forecasts
print(forecast.variance[-1:])  # Last row contains the forecasts

# Access mean forecasts (if applicable)
print(forecast.mean[-1:])
```

The forecast output contains variance predictions for each horizon (1-step, 2-step, ..., h-step ahead).

---

## Chapter 4: Model Specification Choices

### 4.1 Distribution Assumptions

GARCH models assume a distribution for the standardized residuals. The default is normal, but financial returns typically have fatter tails than the normal distribution.

**Available Distributions:**

| Distribution | Code | Characteristics | When to Use |
|-------------|------|-----------------|-------------|
| Normal | `'normal'` | Light tails, symmetric | Baseline, rarely optimal |
| Student's t | `'t'` | Fat tails, symmetric | Most common choice for returns |
| Skewed t | `'skewt'` | Fat tails, asymmetric | When returns show skewness |
| Generalized Error | `'ged'` | Flexible tail weight | Alternative to t-distribution |

```python
# Normal distribution (default)
model_normal = arch_model(returns, p=1, q=1, dist='normal')

# Student's t-distribution (fat tails)
model_t = arch_model(returns, p=1, q=1, dist='t')

# Skewed Student's t-distribution (fat tails + asymmetry)
model_skewt = arch_model(returns, p=1, q=1, dist='skewt')
```

**Why Fat Tails Matter:** Financial returns experience extreme events (crashes, spikes) more frequently than a normal distribution predicts. The t-distribution has heavier tails, meaning it assigns higher probability to extreme outcomes. This leads to more realistic risk estimates.

### 4.2 Mean Model Assumptions

The mean equation (r_t = μ + ε_t) can be specified in different ways:

| Mean Model | Code | Equation | When to Use |
|------------|------|----------|-------------|
| Constant | `'constant'` | r_t = μ + ε_t | Default, usually sufficient |
| Zero | `'zero'` | r_t = ε_t | When mean return is negligible |
| AR(1) | `'AR'` | r_t = φ₀ + φ₁r_{t-1} + ε_t | If returns show autocorrelation |

```python
# Constant mean (default)
model = arch_model(returns, mean='constant', vol='GARCH')

# Zero mean
model = arch_model(returns, mean='zero', vol='GARCH')

# AR(1) mean
model = arch_model(returns, mean='AR', lags=1, vol='GARCH')
```

**Key Insight:** Mean model assumptions typically have minimal impact on volatility estimates. The correlation between volatility estimates from constant-mean and AR-mean models is usually very high (>0.99). Focus more on the volatility specification.

### 4.3 Asymmetric GARCH Models

Standard GARCH treats positive and negative shocks symmetrically—only the magnitude (ε²) matters. But in reality, negative shocks (bad news) often increase volatility more than positive shocks of the same size. This is called the **leverage effect**.

**GJR-GARCH (Glosten-Jagannathan-Runkle):**

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 \cdot I_{[\varepsilon_{t-1} < 0]} + \beta \sigma_{t-1}^2$$

The indicator function I_{[ε < 0]} equals 1 when the shock is negative and 0 otherwise. So negative shocks get an extra impact of γ.

```python
# GJR-GARCH with o=1 for one asymmetric term
gjr_model = arch_model(returns, p=1, q=1, o=1, vol='GARCH', dist='t')
result = gjr_model.fit()
```

**EGARCH (Exponential GARCH):**

Models the log of variance, ensuring variance is always positive:

$$\ln(\sigma_t^2) = \omega + \alpha \left(\frac{|\varepsilon_{t-1}|}{\sigma_{t-1}} - \sqrt{2/\pi}\right) + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}} + \beta \ln(\sigma_{t-1}^2)$$

The γ term captures asymmetry. If γ < 0, negative shocks increase volatility more than positive shocks.

```python
# EGARCH model
egarch_model = arch_model(returns, p=1, q=1, o=1, vol='EGARCH', dist='t')
result = egarch_model.fit()
```

### 4.4 Model Comparison Table

| Model | Equation | Key Feature | Use Case |
|-------|----------|-------------|----------|
| ARCH(1) | σ²_t = ω + αε²_{t-1} | Basic volatility clustering | Simple benchmark |
| GARCH(1,1) | σ²_t = ω + αε²_{t-1} + βσ²_{t-1} | Adds volatility persistence | Standard choice |
| GJR-GARCH | σ²_t = ω + αε²_{t-1} + γε²_{t-1}I_{[ε<0]} + βσ²_{t-1} | Asymmetric response | Leverage effect |
| EGARCH | ln(σ²_t) = ω + α|z_{t-1}| + γz_{t-1} + βln(σ²_{t-1}) | Asymmetric, always positive variance | Volatile assets like crypto |

---

## Chapter 5: Rolling Window Forecasting

### 5.1 Why Rolling Windows?

Using all available data to fit a model and then make predictions has several problems:

1. **Lookback Bias:** In reality, you don't know future data when making predictions. Using future data for model fitting creates unrealistic performance estimates.

2. **Overfitting:** Model parameters may be calibrated to historical quirks that won't repeat. A model fit on crisis data will behave very differently from one fit on calm periods.

3. **Non-Stationarity:** Financial markets evolve. Parameters estimated from 1990s data may not apply to 2020s conditions.

Rolling window forecasts address these issues by continuously refitting the model as new data arrives, mimicking real-world forecasting conditions.

### 5.2 Expanding Window Forecast

The expanding window approach starts with a fixed initial sample and grows the sample as new observations arrive.

**Process:**
1. Fit model using observations 1 to T
2. Forecast observation T+1
3. Add observation T+1 to the sample
4. Fit model using observations 1 to T+1
5. Forecast observation T+2
6. Repeat...

```python
forecasts = {}

for i in range(30):
    # Fit model with expanding window (always starts from beginning)
    result = model.fit(first_obs=0, 
                       last_obs=100 + i,  # End point expands
                       update_freq=5)
    
    # Make 1-step ahead forecast
    temp_result = result.forecast(horizon=1).variance
    fcast = temp_result.iloc[100 + i]
    forecasts[fcast.name] = fcast

forecast_df = pd.DataFrame(forecasts).T
```

**Advantage:** Uses all available historical information.
**Disadvantage:** Old observations may become irrelevant or misleading.

### 5.3 Fixed Rolling Window Forecast

The fixed rolling window maintains a constant window size. As new observations enter, old observations exit.

**Process:**
1. Fit model using observations 1 to T (window size = T)
2. Forecast observation T+1
3. Drop observation 1, add observation T+1
4. Fit model using observations 2 to T+1 (window size still = T)
5. Forecast observation T+2
6. Repeat...

```python
forecasts = {}
window_size = 100

for i in range(30):
    # Fit model with fixed rolling window
    result = model.fit(first_obs=i,           # Start point shifts
                       last_obs=i + window_size,  # End point shifts
                       update_freq=5)
    
    # Make 1-step ahead forecast
    temp_result = result.forecast(horizon=1).variance
    fcast = temp_result.iloc[i + window_size]
    forecasts[fcast.name] = fcast

forecast_df = pd.DataFrame(forecasts).T
```

**Advantage:** More responsive to recent conditions; drops obsolete data.
**Disadvantage:** May miss long-term patterns if window is too short.

### 5.4 Choosing Window Size

There is no universal rule for optimal window size. It depends on the data and application.

**Trade-offs:**

| Window Size | Bias | Variance | Responsiveness |
|-------------|------|----------|----------------|
| Too narrow | High (misses relevant patterns) | Low | High (quick to adapt) |
| Too wide | Low | High (overfitting to old data) | Low (slow to adapt) |

**Practical Guidelines:**
- For daily data: 250-500 observations (1-2 years) is common
- For highly volatile assets: Shorter windows may be better
- For stable markets: Longer windows can be used
- Cross-validate different window sizes if possible

---

## Chapter 6: Model Validation

### 6.1 Parameter Significance Testing

Not all parameters in a GARCH model may be necessary. The principle of parsimony ("KISS: Keep It Simple, Stupid") suggests using the simplest model that adequately explains the data.

**Hypothesis Test Setup:**
- H₀: Parameter = 0 (parameter is unnecessary)
- Hₐ: Parameter ≠ 0 (parameter is necessary)

**Using P-Values:**
The p-value tells you the probability of observing the estimated parameter value if the true value were zero.

Decision Rule:
- If p-value < 0.05: Reject H₀, keep the parameter
- If p-value ≥ 0.05: Fail to reject H₀, consider dropping the parameter

```python
# View p-values
print(result.pvalues)

# Create summary DataFrame
para_summary = pd.DataFrame({
    'parameter': result.params,
    'p-value': result.pvalues
})
print(para_summary)
```

**Using T-Statistics:**
The t-statistic measures how many standard errors the parameter estimate is from zero:

$$t = \frac{\text{parameter estimate}}{\text{standard error}}$$

Rule of Thumb:
- If |t| > 2: Parameter is significantly different from zero, keep it
- If |t| < 2: Parameter may not be necessary

```python
# View t-statistics
print(result.tvalues)

# Verify by manual calculation
calculated_t = result.params / result.std_err
```

### 6.2 Checking Standardized Residuals

If the GARCH model correctly captures the volatility dynamics, the standardized residuals (z_t = ε_t / σ_t) should behave like white noise—independent, identically distributed random variables with no predictable patterns.

**Visual Inspection:**
```python
# Calculate standardized residuals
std_resid = result.resid / result.conditional_volatility

# Plot standardized residuals
plt.plot(std_resid)
plt.title('Standardized Residuals')
plt.show()
```

If the model is good, the plot should show no obvious clustering or patterns. Compare this to the raw returns, which typically show clear clustering.

### 6.3 ACF Plot (Autocorrelation Function)

The ACF plot shows the correlation between the series and its lagged values. For properly specified GARCH models, standardized residuals should have no significant autocorrelation.

**Interpretation:**
- Lag 0 is always 1.0 (perfect correlation with itself)
- Other lags should be near zero
- Blue bands indicate the 95% confidence interval
- Spikes outside the bands suggest remaining autocorrelation (model problems)

```python
from statsmodels.graphics.tsaplots import plot_acf

# Plot ACF of standardized residuals
plot_acf(std_resid, alpha=0.05)  # alpha=0.05 for 95% confidence bands
plt.show()
```

**What Bad ACF Looks Like:**
- Significant spikes at early lags → Model missed some autocorrelation structure
- Slowly decaying pattern → Non-stationarity or long memory not captured

### 6.4 Ljung-Box Test

The Ljung-Box test formally tests whether any of a group of autocorrelations are different from zero. It's a statistical complement to the visual ACF plot.

**Hypothesis:**
- H₀: Data is independently distributed (no autocorrelation)
- Hₐ: Data shows autocorrelation

**Test Statistic:**
$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}$$

Where ρ̂_k is the sample autocorrelation at lag k, and h is the number of lags tested.

**Decision Rule:**
- If p-value < 0.05: Reject H₀, autocorrelation exists (model is inadequate)
- If p-value ≥ 0.05: Fail to reject H₀, no significant autocorrelation (model is adequate)

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Perform Ljung-Box test on standardized residuals
lb_test = acorr_ljungbox(std_resid, lags=10, return_df=True)

# Check p-values
print(lb_test)
```

**Interpreting Results:**
Look at the p-values for multiple lags. If most are above 0.05, the model adequately captures the autocorrelation structure.

---

## Chapter 7: Model Selection

### 7.1 Log-Likelihood

GARCH models are estimated using **Maximum Likelihood Estimation (MLE)**. The likelihood function measures the probability of observing the data given the model parameters. MLE finds the parameters that maximize this probability.

**Log-Likelihood Interpretation:**
- Higher log-likelihood = better fit to the data
- The model with the highest log-likelihood best explains the observed data

```python
# Access log-likelihood
print(result.loglikelihood)
```

**Limitation:** Adding more parameters always increases (or maintains) log-likelihood, even if the extra parameters just capture noise. This leads to overfitting.

### 7.2 Information Criteria

Information criteria balance goodness of fit against model complexity. They penalize models with more parameters to prevent overfitting.

**AIC (Akaike Information Criterion):**
$$AIC = -2 \ln(L) + 2k$$

Where L is the likelihood and k is the number of parameters.

**BIC (Bayesian Information Criterion):**
$$BIC = -2 \ln(L) + k \ln(n)$$

Where n is the number of observations.

**Key Differences:**

| Criterion | Penalty | Tends to Select |
|-----------|---------|-----------------|
| AIC | 2k (lighter) | Slightly more complex models |
| BIC | k·ln(n) (heavier when n > 7) | More parsimonious models |

**Decision Rule:** Lower AIC/BIC indicates a better model (balancing fit and complexity).

```python
# Access AIC and BIC
print('AIC:', result.aic)
print('BIC:', result.bic)

# Compare two models
print('Model 1 AIC:', result1.aic, 'BIC:', result1.bic)
print('Model 2 AIC:', result2.aic, 'BIC:', result2.bic)
# Choose the model with lower values
```

### 7.3 Model Selection Summary

| Criterion | Formula | Interpretation | Usage |
|-----------|---------|----------------|-------|
| Log-Likelihood | ln(L) | Higher = better fit | Compare nested models |
| AIC | -2ln(L) + 2k | Lower = better | General model comparison |
| BIC | -2ln(L) + k·ln(n) | Lower = better | Prefer parsimony |
| P-values | P(data \| H₀) | < 0.05 → significant | Parameter necessity |
| T-statistics | param / SE | \|t\| > 2 → significant | Parameter necessity |

**Practical Workflow:**
1. Start with GARCH(1,1) with t-distribution
2. Check if all parameters are significant (p-values, t-stats)
3. Try asymmetric models (GJR-GARCH, EGARCH) if leverage effect suspected
4. Compare models using AIC/BIC
5. Validate chosen model with ACF plot and Ljung-Box test

---

## Chapter 8: Backtesting

### 8.1 What is Backtesting?

Backtesting assesses model quality by comparing predictions to actual historical outcomes. It answers: "How well would this model have performed if we had used it in the past?"

**The Setup:**
- **In-sample data:** Used for model fitting (training)
- **Out-of-sample data:** Reserved for backtesting (testing)

The key is that out-of-sample data must not be used during model fitting—this mimics real-world conditions where you can't see the future.

### 8.2 Mean Absolute Error (MAE)

MAE measures the average magnitude of forecast errors without considering direction.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Properties:**
- Always positive
- Same units as the original data
- Less sensitive to outliers than MSE
- Lower MAE = better forecast accuracy

### 8.3 Mean Squared Error (MSE)

MSE measures the average of squared forecast errors.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Properties:**
- Always positive
- Units are squared (variance for volatility forecasts)
- Heavily penalizes large errors due to squaring
- Lower MSE = better forecast accuracy

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    
    print(f'Mean Absolute Error (MAE): {mae:.3g}')
    print(f'Mean Squared Error (MSE): {mse:.3g}')
    
    return mae, mse

# Backtest model
evaluate(actual_variance, forecast_variance)
```

### 8.4 MAE vs. MSE

| Metric | Sensitivity to Outliers | Interpretation | Preferred When |
|--------|------------------------|----------------|----------------|
| MAE | Low | Average error magnitude | Outliers should not dominate |
| MSE | High | Penalizes large errors | Large errors are particularly bad |

In practice, report both. If they give conflicting rankings, investigate whether outliers are driving MSE differences.

---

## Chapter 9: Practical Applications

### 9.1 Value at Risk (VaR)

VaR quantifies the maximum expected loss over a given time horizon at a specified confidence level.

**VaR Statement:** "There is a 5% probability that the portfolio will lose at least $X over the next day."

**Components:**
- Time horizon (1 day, 10 days, etc.)
- Confidence level (95%, 99%)
- Loss amount (the VaR value)

**Dynamic VaR with GARCH:**

$$VaR_t = \mu_t + \sigma_t \times q_{\alpha}$$

Where:
- μ_t = forecasted mean return from GARCH
- σ_t = forecasted volatility from GARCH (square root of variance)
- q_α = quantile at confidence level α

### 9.2 Parametric vs. Empirical VaR

**Parametric VaR:**
Uses quantiles from the assumed distribution (normal, t, etc.).

```python
# Get quantile from assumed distribution (e.g., Student's t)
nu = result.params['nu']  # Degrees of freedom for t-distribution
q_parametric = model.distribution.ppf(0.05, nu)

# Calculate VaR
VaR_parametric = mean_forecast + np.sqrt(variance_forecast) * q_parametric
```

**Empirical VaR:**
Uses quantiles from the observed distribution of standardized residuals.

```python
# Get empirical quantile from historical standardized residuals
q_empirical = std_resid.quantile(0.05)

# Calculate VaR
VaR_empirical = mean_forecast + np.sqrt(variance_forecast) * q_empirical
```

**Comparison:**

| Approach | Pros | Cons |
|----------|------|------|
| Parametric | Consistent with model assumptions | May not capture true tail behavior |
| Empirical | Based on actual observed data | Requires sufficient historical data |

### 9.3 VaR Validation

A valid VaR model should have exceedances (actual losses exceeding VaR) at approximately the expected rate.

For 5% daily VaR over 250 trading days:
- Expected exceedances ≈ 0.05 × 250 = 12.5
- If you observe significantly more exceedances, the model underestimates risk
- If you observe significantly fewer, the model may be too conservative

### 9.4 Dynamic Covariance

Covariance measures how two asset returns move together. GARCH enables dynamic (time-varying) covariance estimation.

**Formula:**
$$Cov_t(r_1, r_2) = \rho \times \sigma_{1,t} \times \sigma_{2,t}$$

Where:
- ρ = correlation between standardized residuals of the two assets
- σ_{1,t}, σ_{2,t} = GARCH volatility estimates for each asset

```python
# Step 1: Fit GARCH models for each asset
vol_eur = garch_eur.conditional_volatility
vol_cad = garch_cad.conditional_volatility

# Step 2: Calculate standardized residuals
resid_eur = garch_eur.resid / vol_eur
resid_cad = garch_cad.resid / vol_cad

# Step 3: Calculate correlation between residuals
corr = np.corrcoef(resid_eur, resid_cad)[0, 1]

# Step 4: Calculate dynamic covariance
covariance = corr * vol_eur * vol_cad
```

### 9.5 Portfolio Variance

For a two-asset portfolio with weights w₁ and w₂:

$$\sigma_p^2 = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 Cov(r_1, r_2)$$

**Diversification Benefit:** When covariance is negative, the third term reduces portfolio variance. Assets that move in opposite directions provide natural hedging.

```python
# Define portfolio weights
w1 = 0.6
w2 = 1 - w1

# Calculate portfolio variance
portfolio_var = (w1**2 * variance_asset1 + 
                 w2**2 * variance_asset2 + 
                 2 * w1 * w2 * covariance)
```

### 9.6 Dynamic Beta

Beta measures a stock's sensitivity to market movements. It quantifies systematic risk—risk that cannot be diversified away.

**Formula:**
$$\beta = \rho \times \frac{\sigma_{stock}}{\sigma_{market}}$$

Where:
- ρ = correlation between stock and market standardized residuals
- σ_stock = stock volatility from GARCH
- σ_market = market volatility from GARCH

**Interpretation:**
- β = 1: Stock moves with the market
- β > 1: Stock is more volatile than the market (amplifies market moves)
- β < 1: Stock is less volatile than the market (dampens market moves)
- β < 0: Stock moves opposite to the market (rare)

```python
# Calculate correlation from standardized residuals
correlation = np.corrcoef(stock_resid, market_resid)[0, 1]

# Calculate dynamic beta
stock_beta = correlation * (stock_vol / market_vol)

# Plot beta over time
plt.plot(stock_beta)
plt.title('Dynamic Stock Beta')
plt.show()
```

**Why Dynamic Beta Matters:**
Static beta (single number) assumes the stock-market relationship is constant. In reality, beta varies over time—stocks may become more or less sensitive to market movements depending on economic conditions, company events, or market regimes.

---

## Summary Tables

### Model Specification Options

| Component | Options | Default | Notes |
|-----------|---------|---------|-------|
| p (GARCH lags) | 0, 1, 2, ... | 1 | Usually p=1 is sufficient |
| q (ARCH lags) | 1, 2, ... | 1 | Usually q=1 is sufficient |
| o (asymmetric lags) | 0, 1, ... | 0 | Set o=1 for GJR-GARCH |
| vol | 'GARCH', 'EGARCH', 'ARCH' | 'GARCH' | Choose based on data |
| dist | 'normal', 't', 'skewt', 'ged' | 'normal' | 't' recommended for finance |
| mean | 'constant', 'zero', 'AR' | 'constant' | Usually doesn't matter much |

### Validation Checklist

| Check | Method | Good Result | Bad Result |
|-------|--------|-------------|------------|
| Parameter significance | P-values < 0.05 | All key params significant | Insignificant params present |
| Parameter significance | \|T-stat\| > 2 | Parameters well above 2 | Parameters near 0 |
| Residual autocorrelation | ACF plot | No spikes outside bands | Spikes at early lags |
| Residual autocorrelation | Ljung-Box test | P-values > 0.05 | P-values < 0.05 |
| Model fit | Log-likelihood | Higher is better | — |
| Model selection | AIC/BIC | Lower is better | — |
| Forecast accuracy | MAE/MSE | Lower is better | — |

### Python Functions Quick Reference

```python
# Model specification
arch_model(data, p=1, q=1, o=0, mean='constant', vol='GARCH', dist='t')

# Model fitting
result = model.fit(first_obs=None, last_obs=None, update_freq=4)

# Forecasting
forecast = result.forecast(horizon=5)
forecast.variance  # Variance forecasts
forecast.mean      # Mean forecasts

# Model outputs
result.params                    # Parameter estimates
result.pvalues                   # P-values
result.tvalues                   # T-statistics
result.conditional_volatility    # Fitted volatility series
result.resid                     # Residuals
result.loglikelihood             # Log-likelihood
result.aic                       # AIC
result.bic                       # BIC

# Diagnostics
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

plot_acf(std_resid, alpha=0.05)
acorr_ljungbox(std_resid, lags=10, return_df=True)

# Backtesting
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
```

---

## Key Concepts Recap

1. **Volatility clusters** in financial markets—high volatility tends to persist, as does low volatility.

2. **GARCH models** capture this clustering by making current variance depend on past shocks (ARCH term) and past variance (GARCH term).

3. **Standardized residuals** (ε_t / σ_t) should be white noise if the model is correctly specified.

4. **Distribution choice** matters—use Student's t or skewed t for financial returns due to fat tails.

5. **Asymmetric models** (GJR-GARCH, EGARCH) capture the leverage effect where negative shocks increase volatility more than positive shocks.

6. **Rolling window forecasts** prevent lookback bias and allow the model to adapt to changing conditions.

7. **Model validation** involves checking parameter significance (p-values, t-stats) and residual behavior (ACF, Ljung-Box).

8. **Model selection** balances fit and complexity using AIC/BIC—lower is better.

9. **Backtesting** compares forecasts to actual outcomes using MAE and MSE.

10. **Applications** include VaR for risk management, dynamic covariance for portfolio optimization, and dynamic beta for measuring systematic risk.
