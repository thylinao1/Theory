# Survival Analysis: A Comprehensive Guide

## 1. Introduction to Time-to-Event Analysis

Survival analysis (also called time-to-event analysis, duration analysis, or reliability analysis) is a branch of statistics that deals with analyzing the expected duration until one or more events occur. Despite its name, survival analysis applies far beyond mortality studies:

- **Medicine**: Time until death, disease recurrence, or recovery
- **Finance**: Time until default, customer churn, or loan prepayment
- **Engineering**: Time until equipment failure (reliability analysis)
- **Economics**: Duration of unemployment, time until market entry
- **Social Sciences**: Time until marriage, divorce, or job transition

The fundamental question survival analysis answers is: *What is the probability that an individual survives (or an event does not occur) beyond time $t$?*

---

## 2. Core Concepts

### 2.1 Why Special Methods Are Required

Standard regression techniques (OLS, logistic regression) are inadequate for time-to-event data because:

1. **Non-negativity**: Duration times are always positive ($T \geq 0$), violating the normality assumption of linear regression
2. **Right-skewness**: Duration distributions are typically right-skewed, not symmetric
3. **Censoring**: We often don't observe the event for all subjects—this is the most critical issue

### 2.2 Types of Censoring

**Right Censoring** (most common): The event has not occurred by the end of the observation period. We know the subject survived at least until time $C$, but not when (or if) the event occurs afterward.

$$T_i^{obs} = \min(T_i, C_i)$$

where $T_i$ is the true event time and $C_i$ is the censoring time.

**Left Censoring**: The event occurred before observation began (e.g., disease onset before diagnosis).

**Interval Censoring**: The event is known to have occurred within a time interval, but the exact time is unknown.

### 2.3 The Survival Function

The survival function $S(t)$ gives the probability that the event has not occurred by time $t$:

$$S(t) = P(T > t) = 1 - F(t) = \int_t^{\infty} f(u) \, du$$

where:
- $F(t) = P(T \leq t)$ is the cumulative distribution function (CDF)
- $f(t)$ is the probability density function (PDF)

**Key properties of $S(t)$:**
- $S(0) = 1$ (everyone starts alive/uncensored)
- $S(\infty) = 0$ (eventually everyone experiences the event, assuming proper distribution)
- $S(t)$ is monotonically non-increasing
- The median survival time $t_{0.5}$ satisfies $S(t_{0.5}) = 0.5$

### 2.4 The Hazard Function

The hazard function $h(t)$ (also called the hazard rate, failure rate, or instantaneous risk) gives the instantaneous rate of event occurrence at time $t$, conditional on survival up to time $t$:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t \mid T \geq t)}{\Delta t} = \frac{f(t)}{S(t)} = -\frac{d}{dt}\ln S(t)$$

**Interpretation**: For small $\Delta t$, we have $h(t) \cdot \Delta t \approx P(\text{event in } [t, t+\Delta t) \mid \text{survived to } t)$.

**Relationship to Survival Function**:

$$S(t) = \exp\left(-\int_0^t h(u) \, du\right) = \exp(-H(t))$$

where $H(t) = \int_0^t h(u) \, du$ is the **cumulative hazard function**.

### 2.5 Common Hazard Shapes

The shape of the hazard function reveals important information about the underlying process:

| Hazard Shape | Interpretation | Example |
|--------------|----------------|---------|
| Constant | Memoryless process | Radioactive decay, Exponential distribution |
| Increasing | Aging/wear-out | Human mortality after age 30, mechanical wear |
| Decreasing | Early failures, "burn-in" | Infant mortality, electronics |
| Bathtub | Combination | Human lifetime, electronic devices |

---

## 3. Kaplan-Meier Estimator

### 3.1 Definition and Formula

The Kaplan-Meier (KM) estimator, also called the product-limit estimator, is a non-parametric method for estimating the survival function from censored data.

Let $t_1 < t_2 < \cdots < t_k$ be the distinct observed event times. The KM estimator is:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

where:
- $d_i$ = number of events at time $t_i$
- $n_i$ = number of individuals at risk just before time $t_i$ (alive and uncensored)

**Intuition**: The probability of surviving past time $t$ equals the product of conditional probabilities of surviving each time point up to $t$.

### 3.2 Variance Estimation (Greenwood's Formula)

The variance of $\hat{S}(t)$ is estimated using Greenwood's formula:

$$\widehat{\text{Var}}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{t_i \leq t} \frac{d_i}{n_i(n_i - d_i)}$$

This allows construction of pointwise confidence intervals:

$$\hat{S}(t) \pm z_{\alpha/2} \cdot \sqrt{\widehat{\text{Var}}[\hat{S}(t)]}$$

**Note**: For better coverage properties near 0 and 1, log-log transformed confidence intervals are often preferred:

$$\exp\left(-\exp\left[\log(-\log \hat{S}(t)) \pm \frac{z_{\alpha/2} \cdot \sigma(t)}{\log \hat{S}(t)}\right]\right)$$

### 3.3 Comparing Survival Curves: Log-Rank Test

To compare survival functions between two or more groups, the **log-rank test** (Mantel-Cox test) is the most widely used method.

**Null hypothesis**: $H_0: S_1(t) = S_2(t)$ for all $t$

**Test statistic**:

$$\chi^2 = \frac{\left(\sum_{i=1}^{k}(O_{1i} - E_{1i})\right)^2}{\sum_{i=1}^{k} V_i}$$

where at each event time $t_i$:
- $O_{1i}$ = observed events in group 1
- $E_{1i} = n_{1i} \cdot \frac{d_i}{n_i}$ = expected events under $H_0$
- $V_i$ = variance term

Under $H_0$, this follows a $\chi^2$ distribution with degrees of freedom equal to (number of groups - 1).

### 3.4 Advantages and Disadvantages

**Advantages:**
- Non-parametric: No distributional assumptions required
- Handles censoring naturally
- Produces intuitive step-function estimates
- Widely understood and easily visualized
- Excellent for descriptive/exploratory analysis

**Disadvantages:**
- Cannot directly incorporate covariates (must stratify)
- Step-function may be inappropriate for continuous underlying hazard
- Requires sufficient sample size for precise estimates
- No mechanism for extrapolation beyond observed data
- Median may not be estimable if $\hat{S}(t)$ never drops below 0.5

---

## 4. Parametric Models: The Weibull Model

### 4.1 The Weibull Distribution

The Weibull distribution is one of the most versatile distributions for survival analysis. A random variable $T$ follows a Weibull distribution with shape parameter $\gamma > 0$ and scale parameter $\lambda > 0$ if:

**PDF**:
$$f(t) = \frac{\gamma}{\lambda}\left(\frac{t}{\lambda}\right)^{\gamma-1} \exp\left[-\left(\frac{t}{\lambda}\right)^{\gamma}\right]$$

**Survival Function**:
$$S(t) = \exp\left[-\left(\frac{t}{\lambda}\right)^{\gamma}\right]$$

**Hazard Function**:
$$h(t) = \frac{\gamma}{\lambda}\left(\frac{t}{\lambda}\right)^{\gamma-1} = \frac{\gamma}{\lambda^{\gamma}} t^{\gamma-1}$$

### 4.2 Interpreting the Shape Parameter

The shape parameter $\gamma$ determines the hazard behavior:

| $\gamma$ Value | Hazard Behavior | Interpretation |
|----------------|-----------------|----------------|
| $\gamma < 1$ | Decreasing | Early failures, improving reliability |
| $\gamma = 1$ | Constant | Exponential distribution (memoryless) |
| $\gamma > 1$ | Increasing | Aging, wear-out |

### 4.3 The Accelerated Failure Time (AFT) Formulation

The Weibull model is typically parameterized as an **Accelerated Failure Time (AFT)** model:

$$\log(T_i) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \sigma \epsilon_i$$

where:
- $\epsilon_i$ follows a standard extreme value (Gumbel) distribution
- $\sigma > 0$ is the scale parameter
- $\exp(\beta_j)$ is the multiplicative effect on survival time

**Interpretation of coefficients**: A unit increase in covariate $x_j$ multiplies the expected survival time by $\exp(\beta_j)$. 
- If $\beta_j > 0$: longer survival time (protective effect)
- If $\beta_j < 0$: shorter survival time (harmful effect)

### 4.4 Relationship Between Parameterizations

The R function `survreg()` uses the AFT parameterization. The relationship to the traditional Weibull parameters is:

$$\gamma = \frac{1}{\sigma}, \quad \lambda = \exp(\beta_0)$$

### 4.5 Maximum Likelihood Estimation

For censored data, the likelihood function is:

$$L(\theta) = \prod_{i=1}^{n} \left[f(t_i; \theta)\right]^{\delta_i} \left[S(t_i; \theta)\right]^{1-\delta_i}$$

where $\delta_i = 1$ if subject $i$ experienced the event, and $\delta_i = 0$ if censored.

The log-likelihood is:

$$\ell(\theta) = \sum_{i=1}^{n} \left[\delta_i \log f(t_i; \theta) + (1-\delta_i) \log S(t_i; \theta)\right]$$

### 4.6 Advantages and Disadvantages

**Advantages:**
- Smooth survival curves (more realistic for continuous processes)
- Can incorporate covariates directly
- Allows extrapolation beyond observed data
- Coefficients have intuitive interpretation (time ratios)
- Computationally efficient
- Nested model testing (e.g., Weibull vs. Exponential)
- Can estimate any quantile of the survival distribution

**Disadvantages:**
- Requires distributional assumption (may not hold)
- Monotonic hazard only (no bathtub curves)
- Model misspecification leads to biased estimates
- Less robust than semi-parametric alternatives
- Assumes proportional hazards and AFT simultaneously (restrictive)

---

## 5. Other Parametric Distributions

### 5.1 Exponential Distribution

**Special case**: Weibull with $\gamma = 1$

$$S(t) = \exp(-\lambda t), \quad h(t) = \lambda \text{ (constant)}$$

**Use case**: Memoryless processes, baseline models, queuing theory

### 5.2 Log-Normal Distribution

If $\log(T) \sim N(\mu, \sigma^2)$, then $T$ follows a log-normal distribution.

**Hazard**: Non-monotonic (increases then decreases)—useful for diseases with peak mortality followed by recovery

$$h(t) = \frac{\phi\left(\frac{\log t - \mu}{\sigma}\right)}{t \cdot \sigma \cdot \Phi\left(-\frac{\log t - \mu}{\sigma}\right)}$$

### 5.3 Log-Logistic Distribution

$$S(t) = \frac{1}{1 + (\lambda t)^{\gamma}}$$

**Hazard**: Can be decreasing ($\gamma \leq 1$) or unimodal ($\gamma > 1$)

Useful when hazard peaks and then declines (e.g., certain cancers after treatment).

### 5.4 Gompertz Distribution

$$h(t) = \alpha e^{\beta t}$$

Commonly used in actuarial science for human mortality (hazard increases exponentially with age).

---

## 6. Cox Proportional Hazards Model

### 6.1 Model Formulation

The Cox model is a **semi-parametric** model that separates the baseline hazard from covariate effects:

$$h(t \mid \mathbf{x}) = h_0(t) \exp(\beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p) = h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x})$$

where:
- $h_0(t)$ is the **baseline hazard** (unspecified, non-parametric)
- $\exp(\boldsymbol{\beta}^T \mathbf{x})$ is the **relative risk** or **hazard ratio**

**Key insight**: The baseline hazard $h_0(t)$ is left completely unspecified—we only estimate the regression coefficients $\boldsymbol{\beta}$.

### 6.2 The Proportional Hazards Assumption

The ratio of hazards for two individuals with covariate vectors $\mathbf{x}_1$ and $\mathbf{x}_2$ is:

$$\frac{h(t \mid \mathbf{x}_1)}{h(t \mid \mathbf{x}_2)} = \frac{h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x}_1)}{h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x}_2)} = \exp(\boldsymbol{\beta}^T (\mathbf{x}_1 - \mathbf{x}_2))$$

**Critical**: This ratio is **constant over time**—the hazard functions are proportional. This implies:
- Survival curves for different covariate values should not cross
- The effect of covariates does not change over time

### 6.3 Interpreting Coefficients

| Coefficient | Hazard Ratio | Interpretation |
|-------------|--------------|----------------|
| $\beta_j > 0$ | $\exp(\beta_j) > 1$ | Increased hazard, shorter survival |
| $\beta_j < 0$ | $\exp(\beta_j) < 1$ | Decreased hazard, longer survival |
| $\beta_j = 0$ | $\exp(\beta_j) = 1$ | No effect on hazard |

**Example**: If $\beta = -0.5$ for a treatment indicator, then $HR = \exp(-0.5) \approx 0.61$, meaning the treatment reduces the hazard by 39%.

### 6.4 Partial Likelihood Estimation

Since $h_0(t)$ is unspecified, we cannot use full maximum likelihood. Cox's ingenious solution is the **partial likelihood**:

$$L(\boldsymbol{\beta}) = \prod_{i: \delta_i = 1} \frac{\exp(\boldsymbol{\beta}^T \mathbf{x}_i)}{\sum_{j \in R(t_i)} \exp(\boldsymbol{\beta}^T \mathbf{x}_j)}$$

where $R(t_i)$ is the risk set at time $t_i$ (all subjects still under observation).

**Intuition**: At each event time, we ask: "Given that someone failed, what's the probability it was the observed individual (versus others at risk)?"

This eliminates $h_0(t)$ and allows estimation of $\boldsymbol{\beta}$ without specifying the baseline hazard.

### 6.5 Breslow Estimator for Baseline Hazard

After estimating $\hat{\boldsymbol{\beta}}$, the cumulative baseline hazard can be estimated:

$$\hat{H}_0(t) = \sum_{t_i \leq t} \frac{d_i}{\sum_{j \in R(t_i)} \exp(\hat{\boldsymbol{\beta}}^T \mathbf{x}_j)}$$

The baseline survival function is then $\hat{S}_0(t) = \exp(-\hat{H}_0(t))$.

### 6.6 Survival Function for a Specific Covariate Vector

For a subject with covariates $\mathbf{x}$:

$$\hat{S}(t \mid \mathbf{x}) = \left[\hat{S}_0(t)\right]^{\exp(\hat{\boldsymbol{\beta}}^T \mathbf{x})}$$

### 6.7 Testing the Proportional Hazards Assumption

Several methods exist:

1. **Graphical**: Plot $\log(-\log(\hat{S}(t)))$ vs. $\log(t)$—should be parallel lines for different groups

2. **Schoenfeld Residuals**: Test for correlation between residuals and time
   - $H_0$: No correlation (PH assumption holds)
   - Use `cox.zph()` in R

3. **Include Time Interactions**: Add $x \cdot \log(t)$ terms and test significance

### 6.8 Advantages and Disadvantages

**Advantages:**
- Semi-parametric: No distributional assumption on baseline hazard
- Robust to misspecification of $h_0(t)$
- Partial likelihood eliminates nuisance parameters
- Hazard ratios have clear interpretation
- Most widely used model in biomedical research
- Handles time-varying covariates
- Extensive diagnostic tools available

**Disadvantages:**
- Cannot extrapolate beyond observed data (baseline unspecified)
- Coefficients less intuitive than AFT models (hazard ratios vs. time ratios)
- Proportional hazards assumption may not hold
- Less efficient than parametric models when distributional assumptions are correct
- Ties in event times require approximations (Breslow, Efron, exact)
- Baseline hazard estimation is step-function (discontinuous)

---

## 7. Model Comparison Summary

| Feature | Kaplan-Meier | Weibull (AFT) | Cox PH |
|---------|--------------|---------------|--------|
| **Type** | Non-parametric | Fully parametric | Semi-parametric |
| **Covariates** | Stratification only | Yes | Yes |
| **Distributional assumption** | None | Weibull | None (on baseline) |
| **Hazard assumption** | None | Monotonic | Proportional |
| **Coefficient interpretation** | — | Time ratio | Hazard ratio |
| **Extrapolation** | No | Yes | No |
| **Curve type** | Step function | Smooth | Step function |
| **Primary use** | Descriptive | Prediction, inference | Inference |

### When to Use Each Model

**Kaplan-Meier**: 
- Initial exploration and visualization
- Comparing survival between a few discrete groups
- When no covariate adjustment is needed
- Presenting results to non-technical audiences

**Weibull/Parametric Models**:
- When distributional assumptions are justified
- Need for extrapolation beyond observed data
- Estimating specific survival quantiles
- When interpretations in terms of time (not hazard) are preferred
- Sample size is limited (more efficient if assumptions hold)

**Cox Proportional Hazards**:
- Default choice when distributional assumptions are uncertain
- Multiple covariates require adjustment
- Hazard ratio interpretation is desired
- Baseline hazard shape is unknown
- Extensive diagnostics and inference are needed

---

## 8. Advanced Topics (Brief Overview)

### 8.1 Time-Varying Covariates

Covariates that change over time can be incorporated in the Cox model:

$$h(t \mid \mathbf{x}(t)) = h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x}(t))$$

Requires data in counting process format with (start, stop, event) for each interval.

### 8.2 Competing Risks

When multiple event types are possible (e.g., death from cancer vs. death from heart disease), standard methods may be biased. Solutions include:
- Cause-specific hazard models
- Fine-Gray subdistribution hazard model
- Cumulative incidence functions

### 8.3 Frailty Models

Account for unobserved heterogeneity (individual-level random effects):

$$h(t \mid \mathbf{x}, z) = z \cdot h_0(t) \exp(\boldsymbol{\beta}^T \mathbf{x})$$

where $z > 0$ is a random frailty term, often assumed Gamma or log-normal.

### 8.4 Regularized Cox Models

For high-dimensional data (genomics, many predictors), penalized regression is essential:
- LASSO Cox: $\ell(\boldsymbol{\beta}) - \lambda \|\boldsymbol{\beta}\|_1$
- Ridge Cox: $\ell(\boldsymbol{\beta}) - \lambda \|\boldsymbol{\beta}\|_2^2$
- Elastic Net: Combination of both

Implemented in the `glmnet` package in R.

---

## 9. R Implementation Reference

### 9.1 Creating Survival Objects

```r
library(survival)
library(survminer)

# Create a Surv object (right-censored data)
surv_obj <- Surv(time = data$time, event = data$status)

# For interval-censored data
surv_obj <- Surv(time = data$left, time2 = data$right, type = "interval2")
```

### 9.2 Kaplan-Meier Estimation

```r
# Overall survival curve
km_fit <- survfit(Surv(time, status) ~ 1, data = data)

# Stratified by a grouping variable
km_fit <- survfit(Surv(time, status) ~ group, data = data)

# Summary and median survival
summary(km_fit)
print(km_fit)

# Plotting
ggsurvplot(km_fit, 
           risk.table = TRUE,           # Add risk table
           surv.median.line = "hv",     # Add median line
           conf.int = TRUE,             # Confidence intervals
           pval = TRUE)                 # Log-rank p-value
```

### 9.3 Log-Rank Test

```r
# Compare survival between groups
survdiff(Surv(time, status) ~ group, data = data)
```

### 9.4 Weibull Model

```r
# Fit Weibull model
wb_fit <- survreg(Surv(time, status) ~ covariate1 + covariate2, 
                  data = data, dist = "weibull")

# Coefficients (AFT interpretation)
coef(wb_fit)
summary(wb_fit)

# Predict quantiles (e.g., median survival)
predict(wb_fit, type = "quantile", p = 0.5, newdata = new_data)

# Alternative distributions
exp_fit <- survreg(Surv(time, status) ~ x, data = data, dist = "exponential")
ln_fit <- survreg(Surv(time, status) ~ x, data = data, dist = "lognormal")
ll_fit <- survreg(Surv(time, status) ~ x, data = data, dist = "loglogistic")
```

### 9.5 Cox Proportional Hazards Model

```r
# Fit Cox model
cox_fit <- coxph(Surv(time, status) ~ covariate1 + covariate2, data = data)

# Model summary with hazard ratios
summary(cox_fit)

# Test proportional hazards assumption
cox.zph(cox_fit)
plot(cox.zph(cox_fit))

# Survival curves for specific covariate values
new_data <- data.frame(covariate1 = c(0, 1), covariate2 = c(50, 50))
surv_curves <- survfit(cox_fit, newdata = new_data)
ggsurvplot(surv_curves)
```

### 9.6 Model Diagnostics

```r
# Cox-Snell residuals (check overall fit)
# Should follow unit exponential if model is correct

# Martingale residuals (check functional form)
residuals(cox_fit, type = "martingale")

# Deviance residuals (identify outliers)
residuals(cox_fit, type = "deviance")

# Schoenfeld residuals (check PH assumption)
residuals(cox_fit, type = "schoenfeld")
```

---

## 10. Key Formulas Summary

### Fundamental Relationships

$$S(t) = 1 - F(t) = \int_t^\infty f(u) du$$

$$h(t) = \frac{f(t)}{S(t)} = -\frac{d}{dt}\log S(t)$$

$$S(t) = \exp\left(-\int_0^t h(u) du\right) = \exp(-H(t))$$

### Kaplan-Meier Estimator

$$\hat{S}(t) = \prod_{t_i \leq t}\left(1 - \frac{d_i}{n_i}\right)$$

### Weibull Distribution

$$S(t) = \exp\left[-\left(\frac{t}{\lambda}\right)^\gamma\right], \quad h(t) = \frac{\gamma}{\lambda}\left(\frac{t}{\lambda}\right)^{\gamma-1}$$

### Cox Model

$$h(t \mid \mathbf{x}) = h_0(t)\exp(\boldsymbol{\beta}^T\mathbf{x})$$

$$\text{Hazard Ratio} = \exp(\beta_j)$$

---

## References

1. Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer.
2. Hosmer, D. W., Lemeshow, S., & May, S. (2008). *Applied Survival Analysis* (2nd ed.). Wiley.
3. Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
4. Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187–220.
5. Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete Observations. *Journal of the American Statistical Association*, 53(282), 457–481.
