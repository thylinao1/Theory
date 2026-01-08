# ARIMA Models in Python: Comprehensive Notes

## Table of Contents
1. [Introduction to Time Series](#1-introduction-to-time-series)
2. [Stationarity and Transformations](#2-stationarity-and-transformations)
3. [AR, MA, and ARMA Models](#3-ar-ma-and-arma-models)
4. [Fitting Time Series Models](#4-fitting-time-series-models)
5. [Forecasting](#5-forecasting)
6. [ARIMA Models](#6-arima-models)
7. [Model Selection: ACF, PACF, AIC, and BIC](#7-model-selection-acf-pacf-aic-and-bic)
8. [Model Diagnostics](#8-model-diagnostics)
9. [The Box-Jenkins Methodology](#9-the-box-jenkins-methodology)
10. [Seasonal Time Series and SARIMA](#10-seasonal-time-series-and-sarima)
11. [Automation and Production](#11-automation-and-production)

---

## 1. Introduction to Time Series

### What is a Time Series?

A time series is a sequence of data points collected or recorded at successive points in time, typically at uniform intervals. Unlike cross-sectional data where observations are independent, time series data exhibits temporal dependence—meaning past values influence future values. This autocorrelation structure is precisely what ARIMA models exploit for forecasting.

Time series analysis finds applications across virtually every quantitative field: financial markets (stock prices, volatility), macroeconomics (GDP, inflation, unemployment), public health (disease incidence, hospital admissions), energy (demand forecasting, load balancing), and climate science (temperature anomalies, CO₂ concentrations).

### Key Characteristics of Time Series

**Trend** represents the long-term movement or direction in the data. Mathematically, if we denote our time series as $\{y_t\}_{t=1}^{T}$, a positive trend implies $\mathbb{E}[y_t]$ is increasing in $t$. Trends can be linear ($\mu_t = \alpha + \beta t$), polynomial, or even exponential. The presence of trend typically indicates non-stationarity, which must be addressed before applying ARMA-type models.

**Seasonality** refers to periodic fluctuations that repeat at fixed, known intervals. If a series has seasonal period $S$, we observe patterns where $y_t$ and $y_{t+S}$ exhibit systematic correlation. For monthly data with annual seasonality, $S = 12$; for daily data with weekly patterns, $S = 7$. Seasonality is deterministic in nature—we know when peaks and troughs will occur.

**Cyclicality** differs from seasonality in that cycles have no fixed period. Business cycles, for instance, typically last 2-10 years but with irregular duration. Unlike seasonal patterns, cycles emerge from complex economic dynamics and cannot be predicted based on calendar time alone.

### White Noise: The Building Block

White noise is the foundation of time series modeling. A white noise process $\{\varepsilon_t\}$ satisfies three conditions:

$$\mathbb{E}[\varepsilon_t] = 0 \quad \text{(zero mean)}$$

$$\text{Var}(\varepsilon_t) = \sigma^2 \quad \text{(constant variance)}$$

$$\text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \quad \text{for } t \neq s \quad \text{(no autocorrelation)}$$

If additionally $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$, we have Gaussian white noise. White noise represents the unpredictable "shock" or "innovation" component—the part of future values that cannot be forecast from past information.

### Train-Test Split for Time Series

Unlike cross-sectional machine learning where random splitting is appropriate, time series requires temporal splitting to prevent data leakage. We train on observations from $t = 1, \ldots, T_{\text{train}}$ and test on $t = T_{\text{train}} + 1, \ldots, T$. This respects the causal structure: we can only use past information to predict future values.

```python
# Time-based train-test split
train = df.loc[:'2006']  # All data up to end of 2006
test = df.loc['2007':]   # All data from 2007 onwards
```

---

## 2. Stationarity and Transformations

### Understanding Stationarity

Stationarity is the cornerstone assumption for ARMA modeling. A time series $\{y_t\}$ is (weakly/covariance) stationary if:

**Constant Mean**: $\mathbb{E}[y_t] = \mu$ for all $t$ — the expected value doesn't depend on time.

**Constant Variance**: $\text{Var}(y_t) = \sigma^2$ for all $t$ — the dispersion around the mean is stable.

**Time-Invariant Autocovariance**: $\text{Cov}(y_t, y_{t+k}) = \gamma_k$ depends only on the lag $k$, not on the time $t$.

Intuitively, stationarity means the statistical properties of the series don't change over time. If you took any window of observations, the distribution would look similar regardless of where in the series that window is located. Non-stationary series violate these conditions—they might have trends (violating constant mean), expanding variance, or changing correlation structures.

### The Augmented Dickey-Fuller Test

The Augmented Dickey-Fuller (ADF) test is the standard statistical test for detecting non-stationarity due to a unit root. The null hypothesis is that the series has a unit root (is non-stationary). The test is based on the regression:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t$$

where $\Delta y_t = y_t - y_{t-1}$ is the first difference. The test statistic is the t-statistic for $\hat{\gamma}$. Under the null hypothesis of a unit root ($\gamma = 0$), this statistic follows a non-standard distribution (Dickey-Fuller distribution), not the usual t-distribution.

**Interpretation**: If the p-value < 0.05, we reject the null hypothesis and conclude the series is stationary. The more negative the test statistic, the stronger the evidence against non-stationarity.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f'Test Statistic: {result[0]:.4f}')
print(f'P-value: {result[1]:.4f}')
print(f'Critical Values: {result[4]}')
# Reject null (conclude stationarity) if test statistic < critical value
```

**Important Caveat**: The ADF test only tests for trend stationarity (unit root). A series can pass the ADF test but still be non-stationary due to changing variance or structural breaks. Always complement statistical tests with visual inspection.

### Differencing: The Primary Transformation

Differencing is the most common transformation to achieve stationarity. The first difference operator $\nabla$ is defined as:

$$\nabla y_t = y_t - y_{t-1} = (1 - B)y_t$$

where $B$ is the backshift operator ($By_t = y_{t-1}$). If first differencing isn't sufficient, we can apply it again:

$$\nabla^2 y_t = \nabla(\nabla y_t) = y_t - 2y_{t-1} + y_{t-2}$$

This second difference captures acceleration or changes in the rate of change. In practice, we rarely need more than two differences for economic and financial time series.

```python
# First difference
df_diff1 = df.diff().dropna()

# Second difference
df_diff2 = df.diff().diff().dropna()

# Verify stationarity after differencing
result = adfuller(df_diff1['column'])
```

### Other Transformations

**Log Transformation**: For series with multiplicative structure or exponential growth, the logarithm stabilizes variance and linearizes exponential trends:

$$z_t = \log(y_t)$$

**Log-Returns**: Standard in finance, capturing percentage changes in a scale-independent way:

$$r_t = \log\left(\frac{y_t}{y_{t-1}}\right) = \log(y_t) - \log(y_{t-1})$$

This is approximately equal to the percentage return for small changes, has the attractive property of time-additivity (multi-period returns are sums of single-period returns), and often produces more stationary series than simple differences for asset prices.

**Box-Cox Transformation**: A parametric family that includes log as a special case:

$$y_t^{(\lambda)} = \begin{cases} \frac{y_t^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(y_t) & \text{if } \lambda = 0 \end{cases}$$

The parameter $\lambda$ can be estimated to optimally stabilize variance.

---

## 3. AR, MA, and ARMA Models

### Autoregressive (AR) Models

An autoregressive model of order $p$, denoted AR($p$), expresses the current value as a linear combination of its $p$ most recent values plus a white noise shock:

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t$$

where $c$ is a constant (related to the mean), $\phi_1, \ldots, \phi_p$ are the autoregressive coefficients, and $\varepsilon_t \sim WN(0, \sigma^2)$.

Using the backshift operator, this can be written compactly as:

$$\Phi(B)y_t = c + \varepsilon_t \quad \text{where} \quad \Phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \cdots - \phi_p B^p$$

**Stationarity Condition**: An AR($p$) process is stationary if and only if all roots of the characteristic polynomial $\Phi(z) = 0$ lie outside the unit circle in the complex plane. For AR(1), this simplifies to $|\phi_1| < 1$.

**Intuition**: AR models capture momentum or mean-reversion dynamics. A positive $\phi_1$ close to 1 indicates persistence—high values tend to be followed by high values. A negative $\phi_1$ indicates oscillatory behavior around the mean.

### Moving Average (MA) Models

A moving average model of order $q$, denoted MA($q$), expresses the current value as a linear combination of current and past shock terms:

$$y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}$$

where $\mu$ is the mean, $\theta_1, \ldots, \theta_q$ are the moving average coefficients, and $\varepsilon_t \sim WN(0, \sigma^2)$.

In operator notation:

$$y_t = \mu + \Theta(B)\varepsilon_t \quad \text{where} \quad \Theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \cdots + \theta_q B^q$$

**Key Properties**: MA processes are always stationary (they're finite linear combinations of white noise). However, for unique representation (invertibility), we require all roots of $\Theta(z) = 0$ to lie outside the unit circle.

**Intuition**: MA models capture how shocks propagate through time. The coefficient $\theta_j$ measures how much a shock $j$ periods ago still affects the current value. After $q$ periods, the effect of any shock vanishes entirely.

### ARMA Models: Combining AR and MA

The ARMA($p, q$) model combines both autoregressive and moving average components:

$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

Or equivalently:

$$\Phi(B)y_t = c + \Theta(B)\varepsilon_t$$

ARMA models provide a parsimonious representation—they can capture complex autocorrelation structures with fewer parameters than pure AR or MA models alone. The principle of parsimony suggests using the simplest adequate model.

### Generating Simulated ARMA Data

Understanding how to generate ARMA data helps build intuition about model behavior:

```python
from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np

np.random.seed(42)

# ARMA(1,1) with φ₁ = 0.5, θ₁ = 0.3
# Note: ar_coefs uses NEGATIVE of desired φ values after the leading 1
ar_coefs = [1, -0.5]  # Represents 1 - 0.5B
ma_coefs = [1, 0.3]   # Represents 1 + 0.3B

y = arma_generate_sample(ar_coefs, ma_coefs, nsample=200, scale=1.0)
```

---

## 4. Fitting Time Series Models

### Model Estimation

Fitting ARMA models involves estimating the parameters $(\phi_1, \ldots, \phi_p, \theta_1, \ldots, \theta_q, \sigma^2)$ from observed data. The statsmodels library uses Maximum Likelihood Estimation (MLE), which finds parameters that maximize the probability of observing the data.

For Gaussian innovations, the log-likelihood function is:

$$\ell(\boldsymbol{\theta}) = -\frac{T}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^{T}\log(\sigma_t^2) - \frac{1}{2}\sum_{t=1}^{T}\frac{\varepsilon_t^2}{\sigma_t^2}$$

The optimization is handled numerically—statsmodels uses state-space representations and the Kalman filter for efficient computation.

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit an ARMA(2,1) model (middle order parameter d=0 means no differencing)
model = ARIMA(data, order=(2, 0, 1))
results = model.fit()
print(results.summary())
```

### Interpreting the Summary Output

The `results.summary()` provides crucial information about your fitted model. The coefficient table shows estimated values with standard errors, z-statistics, and p-values. Significant coefficients (p-value < 0.05) indicate the corresponding lag contributes meaningfully to the model.

The `sigma2` parameter represents the estimated variance of the innovation terms $\hat{\sigma}^2$. Log-likelihood, AIC, and BIC are model selection criteria discussed later.

### ARMAX Models: Including Exogenous Variables

The ARMAX model extends ARMA by incorporating external predictors:

$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \beta_1 x_{1,t} + \cdots + \beta_k x_{k,t} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

This is a regression model with ARMA errors—the dependent variable depends on both its own past and contemporaneous external factors. For example, modeling daily electricity demand might include temperature as an exogenous variable while accounting for autocorrelation in the residuals.

```python
# ARMAX model with exogenous variable
model = ARIMA(y, order=(2, 0, 1), exog=X)
results = model.fit()
```

---

## 5. Forecasting

### One-Step-Ahead Predictions

One-step-ahead forecasting uses all available information up to time $t$ to predict $y_{t+1}$. For an AR(1) model:

$$\hat{y}_{t+1|t} = \hat{c} + \hat{\phi}_1 y_t$$

The forecast is simply applying the estimated model equation with the most recent observation. The forecast error is:

$$e_{t+1|t} = y_{t+1} - \hat{y}_{t+1|t} = \varepsilon_{t+1}$$

The uncertainty in this prediction equals the standard deviation of the innovations: $\sigma_\varepsilon$.

```python
# One-step-ahead predictions for last 30 observations
one_step_forecast = results.get_prediction(start=-30)
mean_forecast = one_step_forecast.predicted_mean
conf_int = one_step_forecast.conf_int()
```

### Multi-Step (Dynamic) Predictions

For horizons beyond one step, we must forecast recursively—using forecasts to generate further forecasts. For an AR(1) at horizon $h$:

$$\hat{y}_{t+h|t} = \hat{c}\sum_{j=0}^{h-1}\hat{\phi}_1^j + \hat{\phi}_1^h y_t$$

As $h \to \infty$, the forecast converges to the unconditional mean $\mu = c/(1-\phi_1)$ (assuming stationarity).

**Uncertainty Grows with Horizon**: The variance of the $h$-step forecast error increases with $h$ because each step accumulates uncertainty from unknown future shocks:

$$\text{Var}(\hat{y}_{t+h|t}) = \sigma^2 \sum_{j=0}^{h-1}\psi_j^2$$

where $\psi_j$ are the coefficients in the MA($\infty$) representation of the process. This is why confidence intervals fan out for longer horizons.

```python
# Dynamic forecasts (multi-step, using predicted values)
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Out-of-sample future forecasts
future_forecast = results.get_forecast(steps=10)
mean_future = future_forecast.predicted_mean
conf_int_future = future_forecast.conf_int()
```

### Visualizing Forecasts

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data.index, data, label='Observed')
plt.plot(mean_forecast.index, mean_forecast, color='red', label='Forecast')
plt.fill_between(conf_int.index, 
                 conf_int.iloc[:, 0], 
                 conf_int.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.legend()
plt.show()
```

---

## 6. ARIMA Models

### From ARMA to ARIMA

The ARIMA (AutoRegressive Integrated Moving Average) model handles non-stationary series by incorporating differencing directly into the model specification. An ARIMA($p, d, q$) model applies $d$ differences to achieve stationarity, then fits an ARMA($p, q$) to the differenced series.

The model equation is:

$$\Phi(B)(1-B)^d y_t = c + \Theta(B)\varepsilon_t$$

Or expanded:

$$\nabla^d y_t = c + \phi_1 \nabla^d y_{t-1} + \cdots + \phi_p \nabla^d y_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}$$

where $\nabla^d = (1-B)^d$ is the $d$-th difference operator.

### The "Integrated" in ARIMA

The term "integrated" refers to the inverse operation of differencing. If we difference $d$ times to achieve stationarity, we must "integrate" (cumulative sum) $d$ times to recover forecasts on the original scale. The ARIMA class handles this automatically—you pass in the original series and specify $d$, and forecasts are returned on the original scale.

### Manual Differencing vs. ARIMA

You could manually difference, fit ARMA, then invert—but ARIMA is cleaner:

**Manual Approach** (not recommended):
```python
# Difference the data
diff_data = data.diff().dropna()

# Fit ARMA to differenced data
model = ARIMA(diff_data, order=(2, 0, 2))
results = model.fit()

# Forecast differences
diff_forecast = results.get_forecast(steps=10).predicted_mean

# Integrate to get levels
level_forecast = np.cumsum(diff_forecast) + data.iloc[-1]
```

**ARIMA Approach** (recommended):
```python
# ARIMA handles differencing internally
model = ARIMA(data, order=(2, 1, 2))  # d=1 means first difference
results = model.fit()

# Forecasts are automatically on original scale
level_forecast = results.get_forecast(steps=10).predicted_mean
```

### Choosing the Differencing Order $d$

Determine $d$ empirically before fitting:

1. Plot the series and check for trend/non-constant mean
2. Apply ADF test
3. If non-stationary, difference and repeat
4. Stop when series passes stationarity tests

Typically $d \in \{0, 1, 2\}$. Using $d > 2$ is rare and often indicates over-differencing or other issues.

---

## 7. Model Selection: ACF, PACF, AIC, and BIC

### The Autocorrelation Function (ACF)

The ACF measures correlation between $y_t$ and $y_{t-k}$ at various lags $k$:

$$\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} = \frac{\gamma_k}{\gamma_0}$$

The sample ACF is estimated as:

$$\hat{\rho}_k = \frac{\sum_{t=k+1}^{T}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}$$

For stationary processes, the ACF decays toward zero. The pattern of decay reveals model structure.

### The Partial Autocorrelation Function (PACF)

The PACF measures the correlation between $y_t$ and $y_{t-k}$ after removing the linear dependence on intermediate lags $y_{t-1}, \ldots, y_{t-k+1}$. It's the coefficient $\phi_{kk}$ in the regression:

$$y_t = \phi_{k1}y_{t-1} + \phi_{k2}y_{t-2} + \cdots + \phi_{kk}y_{t-k} + \varepsilon_t$$

The PACF isolates the direct effect of lag $k$, uncontaminated by shorter lags.

### Identifying Model Orders from ACF/PACF

The theoretical patterns are:

| Model | ACF Behavior | PACF Behavior |
|-------|--------------|---------------|
| AR($p$) | Tails off exponentially/sinusoidally | Cuts off after lag $p$ |
| MA($q$) | Cuts off after lag $q$ | Tails off exponentially/sinusoidally |
| ARMA($p,q$) | Tails off after lag $q-p$ | Tails off after lag $p-q$ |

"Cuts off" means drops to zero (within confidence bands) abruptly. "Tails off" means gradual decay.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data, lags=20, zero=False, ax=ax1)
plot_pacf(data, lags=20, zero=False, ax=ax2)
plt.tight_layout()
plt.show()
```

### Information Criteria: AIC and BIC

When ACF/PACF patterns are ambiguous (especially for ARMA models), information criteria provide objective model comparison.

**Akaike Information Criterion (AIC)**:

$$\text{AIC} = -2\ln(\hat{L}) + 2k$$

where $\hat{L}$ is the maximized likelihood and $k$ is the number of estimated parameters.

**Bayesian Information Criterion (BIC)**:

$$\text{BIC} = -2\ln(\hat{L}) + k\ln(T)$$

where $T$ is the sample size.

Both criteria balance fit (log-likelihood) against complexity (number of parameters). Lower values indicate better models. BIC penalizes complexity more heavily than AIC, especially for large samples, thus favoring simpler models.

**When to use which**: AIC is asymptotically optimal for prediction; BIC is consistent for model identification (correctly identifies the true model as $T \to \infty$ if it's in the candidate set).

### Grid Search Over Model Orders

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

results_list = []

for p in range(4):
    for q in range(4):
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            results_list.append({
                'p': p, 'q': q,
                'AIC': results.aic,
                'BIC': results.bic
            })
        except:
            continue  # Skip problematic orders

results_df = pd.DataFrame(results_list)
print(results_df.sort_values('AIC').head())
```

---

## 8. Model Diagnostics

### Residual Analysis

After fitting, we examine residuals $\hat{\varepsilon}_t = y_t - \hat{y}_{t|t-1}$ to verify model adequacy. Well-specified model residuals should behave like white noise: zero mean, constant variance, no autocorrelation, and ideally normal distribution.

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{T}\sum_{t=1}^{T}|\hat{\varepsilon}_t|$$

MAE quantifies average forecast error magnitude in the original units, providing an intuitive measure of typical prediction accuracy.

```python
import numpy as np
mae = np.mean(np.abs(results.resid))
```

### Diagnostic Plots

The `plot_diagnostics()` method produces four essential plots:

**Standardized Residuals**: Time series plot of residuals divided by their estimated standard deviation. Look for any patterns, trends, or volatility clustering—ideally appears as random noise around zero.

**Histogram + KDE**: Compares the empirical distribution of residuals (histogram and kernel density estimate) against a normal distribution. Departures suggest non-normality, which affects confidence interval validity.

**Normal Q-Q Plot**: Plots residual quantiles against theoretical normal quantiles. Points should fall along the 45° line. Systematic deviations indicate skewness (S-curve) or heavy tails (deviations at extremes).

**Correlogram (ACF of Residuals)**: Shows residual autocorrelations. Significant spikes indicate remaining structure not captured by the model—suggesting additional AR or MA terms may be needed.

```python
results.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### Statistical Tests in the Summary

**Ljung-Box Test (Prob(Q))**: Tests the null hypothesis that residuals are uncorrelated up to a certain lag. P-value < 0.05 suggests significant autocorrelation remains—the model may be misspecified.

$$Q = T(T+2)\sum_{k=1}^{h}\frac{\hat{\rho}_k^2}{T-k}$$

**Jarque-Bera Test (Prob(JB))**: Tests normality based on skewness and kurtosis. P-value < 0.05 suggests non-normal residuals. While this doesn't invalidate point forecasts, it affects prediction interval coverage.

$$JB = \frac{T}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

where $S$ is skewness and $K$ is kurtosis.

---

## 9. The Box-Jenkins Methodology

### A Systematic Framework

The Box-Jenkins methodology provides a structured approach to ARIMA modeling, consisting of three iterative stages:

### Stage 1: Identification

The goal is to determine appropriate model orders $(p, d, q)$ and any necessary transformations.

**Steps**:
1. Plot the series to visually assess trend, seasonality, and variance stability
2. Apply transformations if needed (log for multiplicative patterns, Box-Cox for variance stabilization)
3. Test for stationarity using ADF test
4. Difference until stationary (determine $d$)
5. Examine ACF/PACF of stationary series to identify candidate $p$ and $q$

### Stage 2: Estimation

Fit the identified model(s) using maximum likelihood estimation. If multiple plausible specifications exist from the identification stage, fit all candidates.

```python
# Fit candidate models
model1 = ARIMA(data, order=(1, 1, 1)).fit()
model2 = ARIMA(data, order=(2, 1, 0)).fit()
model3 = ARIMA(data, order=(1, 1, 2)).fit()

# Compare using AIC
print(f"ARIMA(1,1,1): AIC = {model1.aic:.2f}")
print(f"ARIMA(2,1,0): AIC = {model2.aic:.2f}")
print(f"ARIMA(1,1,2): AIC = {model3.aic:.2f}")
```

### Stage 3: Diagnostic Checking

Verify the best model's residuals satisfy white noise assumptions:

1. Check residual plots for patterns
2. Verify ACF of residuals shows no significant autocorrelation
3. Confirm Ljung-Box and Jarque-Bera tests pass (or understand why not)

If diagnostics reveal problems, return to identification and consider alternative specifications.

### Iteration and Production

The methodology is inherently iterative. You may cycle through identification-estimation-diagnostics multiple times before finding a satisfactory model. Only when diagnostics pass should you proceed to production forecasting.

---

## 10. Seasonal Time Series and SARIMA

### Understanding Seasonality

Seasonal patterns are pervasive in real-world data: retail sales spike in December, air conditioning demand peaks in summer, website traffic follows weekly patterns. A seasonal time series can be decomposed additively:

$$y_t = T_t + S_t + R_t$$

where $T_t$ is the trend component, $S_t$ is the seasonal component (with $\sum_{j=1}^{S}S_{t+j} \approx 0$ over each cycle), and $R_t$ is the irregular/residual component.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data, period=12)  # Monthly data, annual seasonality
decomposition.plot()
plt.show()
```

### Detecting Seasonal Period from ACF

The ACF of seasonal data shows peaks at seasonal lags. For a series with period $S$, expect significant autocorrelation at lags $S, 2S, 3S, \ldots$. To better visualize seasonality, first detrend by subtracting a rolling mean:

```python
# Detrend to clarify seasonal ACF pattern
detrended = data - data.rolling(window=15).mean()
detrended = detrended.dropna()

plot_acf(detrended, lags=40)
# Look for peaks at multiples of the seasonal period
```

### The SARIMA Model

The Seasonal ARIMA model, SARIMA($p, d, q$)($P, D, Q$)$_S$, extends ARIMA by adding seasonal AR and MA terms that operate at the seasonal lag $S$.

The full model equation is:

$$\Phi_P(B^S)\phi_p(B)(1-B)^d(1-B^S)^D y_t = c + \Theta_Q(B^S)\theta_q(B)\varepsilon_t$$

where:

**Non-seasonal components**:
- $\phi_p(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR polynomial)
- $\theta_q(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA polynomial)
- $(1-B)^d$ is $d$-th order differencing

**Seasonal components**:
- $\Phi_P(B^S) = 1 - \Phi_1 B^S - \cdots - \Phi_P B^{PS}$ (seasonal AR polynomial)
- $\Theta_Q(B^S) = 1 + \Theta_1 B^S + \cdots + \Theta_Q B^{QS}$ (seasonal MA polynomial)
- $(1-B^S)^D$ is $D$-th order seasonal differencing

### Expanded Form Example

Consider a SARIMA(1,1,1)(1,1,1)$_{12}$ model for monthly data. Expanding the operators:

$$\underbrace{(1 - \Phi_1 B^{12})}_{\text{Seasonal AR}} \underbrace{(1 - \phi_1 B)}_{\text{Non-seasonal AR}} \underbrace{(1-B)}_{\text{Diff}} \underbrace{(1-B^{12})}_{\text{Seasonal Diff}} y_t = \underbrace{(1 + \Theta_1 B^{12})}_{\text{Seasonal MA}} \underbrace{(1 + \theta_1 B)}_{\text{Non-seasonal MA}} \varepsilon_t$$

After multiplication, this creates dependencies at lags 1, 12, 13 on both the AR and MA sides, capturing both local dynamics (adjacent months) and seasonal patterns (same month last year).

### Seasonal Differencing

The seasonal difference operator removes seasonal patterns:

$$\nabla_S y_t = y_t - y_{t-S} = (1 - B^S)y_t$$

For monthly data with $S = 12$, this compares each month to the same month last year, effectively removing annual seasonality. You might need both regular and seasonal differencing:

```python
# Both regular and seasonal differencing
diff_data = data.diff().diff(12).dropna()  # d=1, D=1 for monthly data
```

### Choosing Seasonal Orders

For seasonal ACF/PACF analysis, examine correlations at seasonal lags $S, 2S, 3S, \ldots$:

```python
# After appropriate differencing
diff_data = data.diff().diff(12).dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Non-seasonal orders: look at first few lags
plot_acf(diff_data, lags=11, zero=False, ax=ax1)
plot_pacf(diff_data, lags=11, zero=False, ax=ax2)
plt.show()

# Seasonal orders: look at seasonal lags specifically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(diff_data, lags=[12, 24, 36], zero=False, ax=ax1)
plot_pacf(diff_data, lags=[12, 24, 36], zero=False, ax=ax2)
plt.show()
```

Apply the same AR/MA identification rules to seasonal lags for $(P, Q)$ as you would for non-seasonal lags for $(p, q)$.

### Fitting SARIMA in Python

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(1,1,1)(1,1,1)₁₂ for monthly data
model = SARIMAX(data, 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12))
results = model.fit()
print(results.summary())
```

### Additive vs. Multiplicative Seasonality

**Additive seasonality**: Seasonal fluctuations are constant regardless of the level. The amplitude of seasonal swings doesn't change as the series trends up or down.

**Multiplicative seasonality**: Seasonal fluctuations scale with the level. As the series grows, seasonal swings become larger proportionally.

SARIMA inherently models additive seasonality. For multiplicative patterns, apply a log transformation first:

```python
import numpy as np

# Transform multiplicative to additive
log_data = np.log(data)

# Fit SARIMA to log-transformed data
model = SARIMAX(log_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecasts need to be exponentiated
log_forecast = results.get_forecast(steps=12).predicted_mean
forecast = np.exp(log_forecast)
```

### Practical Guidelines for Seasonal Modeling

1. **Never exceed $D = 1$** for seasonal differencing
2. **Rarely need $d + D > 2$** total orders of differencing
3. **Strong seasonality** (clear, consistent patterns) → always use $D = 1$
4. **Weak seasonality** (irregular amplitude) → try both with and without seasonal differencing
5. **Multiplicative patterns** → log-transform before modeling

---

## 11. Automation and Production

### Automated Order Selection with pmdarima

The `pmdarima` package implements automated ARIMA model selection, similar to R's `auto.arima()`:

```python
import pmdarima as pm

# Automated SARIMA selection
model = pm.auto_arima(
    data,
    seasonal=True,
    m=12,                    # Seasonal period
    d=None,                  # Let algorithm determine d
    D=None,                  # Let algorithm determine D
    start_p=0, max_p=3,      # AR order bounds
    start_q=0, max_q=3,      # MA order bounds
    start_P=0, max_P=2,      # Seasonal AR bounds
    start_Q=0, max_Q=2,      # Seasonal MA bounds
    information_criterion='aic',
    trace=True,              # Print search progress
    error_action='ignore',   # Skip failing models
    suppress_warnings=True,
    stepwise=True            # Use stepwise search (faster)
)

print(model.summary())
```

**Stepwise vs. Grid Search**: The `stepwise=True` option uses a smart search algorithm that explores the model space efficiently, starting from an initial model and moving to neighboring orders. This is much faster than exhaustive grid search but may miss the global optimum in rare cases.

### Saving and Loading Models

For production deployment, save trained models using `joblib`:

```python
import joblib

# Save the fitted model
joblib.dump(model, 'sarima_model.pkl')

# Later, load the model
loaded_model = joblib.load('sarima_model.pkl')

# Make predictions with loaded model
forecast = loaded_model.predict(n_periods=12)
```

### Updating Models with New Data

As new observations become available, update the model rather than refitting from scratch:

```python
# new_data contains recent observations
loaded_model.update(new_data)

# Make updated forecasts
updated_forecast = loaded_model.predict(n_periods=12)
```

**Important**: The `update()` method adjusts parameters incrementally—it doesn't re-select model orders. If significant structural changes occur (regime shifts, new seasonality), consider re-running the full Box-Jenkins procedure.

### Production Workflow Summary

1. **Development**: Apply Box-Jenkins methodology to identify and validate model
2. **Validation**: Test on held-out data, verify forecast accuracy meets requirements
3. **Deployment**: Save model object, set up automated data pipeline
4. **Monitoring**: Track forecast errors over time, flag deteriorating performance
5. **Maintenance**: Periodically update with new data, re-evaluate model specification if needed

---

## Quick Reference: Key Formulas

### Model Equations

| Model | Equation |
|-------|----------|
| AR($p$) | $y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \varepsilon_t$ |
| MA($q$) | $y_t = \mu + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}$ |
| ARMA($p,q$) | $y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j}$ |
| ARIMA($p,d,q$) | $\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$ |
| SARIMA | $\Phi(B^S)\phi(B)(1-B)^d(1-B^S)^D y_t = c + \Theta(B^S)\theta(B)\varepsilon_t$ |

### Key Operators

| Operator | Definition |
|----------|------------|
| Backshift | $By_t = y_{t-1}$ |
| Difference | $\nabla y_t = (1-B)y_t = y_t - y_{t-1}$ |
| Seasonal Difference | $\nabla_S y_t = (1-B^S)y_t = y_t - y_{t-S}$ |

### Model Selection Criteria

| Criterion | Formula | Use Case |
|-----------|---------|----------|
| AIC | $-2\ln(\hat{L}) + 2k$ | Prediction optimization |
| BIC | $-2\ln(\hat{L}) + k\ln(T)$ | Model identification |

---

## Key Python Functions Reference

```python
# Stationarity testing
from statsmodels.tsa.stattools import adfuller
result = adfuller(series)

# ACF/PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data, lags=20)
plot_pacf(data, lags=20)

# ARIMA modeling
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(data, order=(p, d, q))
results = model.fit()

# SARIMA modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data, order=(p,d,q), seasonal_order=(P,D,Q,S))
results = model.fit()

# Forecasting
forecast = results.get_forecast(steps=h)
mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Diagnostics
results.plot_diagnostics()
print(results.summary())

# Automated selection
import pmdarima as pm
model = pm.auto_arima(data, seasonal=True, m=S)

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(data, period=S)
```
