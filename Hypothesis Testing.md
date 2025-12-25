# Hypothesis Testing in Python: Comprehensive Notes

## Chapter 1: Introduction to Hypothesis Testing

### 1.1 What is Hypothesis Testing?

Hypothesis testing is a statistical framework that allows us to make decisions about population parameters based on sample data. In the real world, we almost never have access to an entire population—we can't survey every customer, measure every product, or track every stock trade. Instead, we collect a sample and use that sample to draw conclusions about the larger population.

The core challenge is distinguishing between two possibilities: did we observe something meaningful about the population, or did we just happen to get an unusual sample by random chance? Hypothesis testing provides a rigorous, quantitative way to make this distinction.

The framework works like a criminal trial: you start by assuming innocence (the null hypothesis), and only reject that assumption if the evidence is strong enough to prove guilt "beyond a reasonable doubt." Just as a jury doesn't need to prove the defendant is innocent—they simply evaluate whether the prosecution has made a convincing case—hypothesis testing doesn't prove anything definitively. It tells us whether the data is inconsistent enough with our default assumption that we should abandon it.

### 1.2 The Two Hypotheses

Every hypothesis test involves two competing claims about a population parameter (like a mean, proportion, or difference).

**Null Hypothesis (H₀):** The null hypothesis represents the default assumption, the status quo, or the "nothing special is happening" scenario. This is what you assume to be true until evidence suggests otherwise. The null hypothesis typically claims that there is no effect, no difference, or that a parameter equals some specific value. For example, "the average salary of data scientists equals $110,000" or "the proportion of late shipments equals 6%."

**Alternative Hypothesis (Hₐ):** The alternative hypothesis is the new claim that challenges the null hypothesis. This is typically what the researcher hopes to find evidence for—a new drug works better than the old one, a marketing campaign increased sales, or a manufacturing process produces fewer defects. The alternative hypothesis represents a departure from the status quo.

Here's a concrete example: suppose a company claims their shipments are late only 6% of the time. You suspect the rate is actually higher. You would set up:
- H₀: p = 0.06 (the late shipment rate equals 6%, as claimed)
- Hₐ: p > 0.06 (the late shipment rate is greater than 6%)

The logic of hypothesis testing is inherently conservative: we maintain H₀ unless we find strong evidence against it. We never "accept" H₀—we either reject it or fail to reject it. This is similar to how a defendant is found "not guilty" rather than "innocent."

### 1.3 Z-Scores: Standardizing Your Evidence

Since variables have arbitrary units and scales (dollars, percentages, kilograms, etc.), we need a way to standardize our observations so we can evaluate them on a common scale. A salary difference of $10,000 might be huge in one context and trivial in another—it depends on how much variability is typical. The z-score converts your observed statistic into a standardized measure that tells you how many standard errors away from the hypothesized value your observation lies.

**The Z-Score Formula:**

$$z = \frac{\text{sample statistic} - \text{hypothesized value}}{\text{standard error}}$$

The numerator measures the raw difference between what you observed in your sample and what you would expect under the null hypothesis. The denominator (standard error) measures how much natural variability you'd expect in your sample statistic due to random sampling. Dividing by the standard error puts the difference in context: a $10,000 difference is a big deal if typical sampling variability is $2,000, but not if typical variability is $50,000.

**Interpretation:** A z-score of 2.5 means your sample statistic is 2.5 standard errors above the hypothesized value. This is far from what we'd expect if H₀ were true. A z-score of 0.3 means your observation is only 0.3 standard errors away—very close to what we'd expect under H₀, easily explained by random chance. The larger the absolute z-score, the more "surprising" your result is under the null hypothesis, and the more evidence you have against H₀.

**How to Estimate Standard Error:**

The standard error measures how much your sample statistic would vary if you repeatedly drew new samples from the population. There are two main approaches to estimate it:

1. **Bootstrap method:** Generate thousands of resamples from your original data (sampling with replacement), calculate the statistic for each resample, and take the standard deviation of these statistics. This empirically constructs the sampling distribution.

2. **Analytical formulas:** Use mathematical formulas derived from probability theory. These are faster but require certain distributional assumptions to be valid.

```python
# Bootstrap approach to estimate standard error
std_error = np.std(bootstrap_distribution)

# Calculate z-score
z_score = (sample_statistic - hypothesized_value) / std_error
```

### 1.4 P-Values: Quantifying the Evidence

The p-value is the probability of observing a result at least as extreme as what you actually got, assuming the null hypothesis is true. It quantifies how "surprising" your data would be in a world where H₀ is correct.

Think of it this way: if the null hypothesis were true, and you repeated your study thousands of times, what fraction of those studies would produce results as extreme as (or more extreme than) yours? That fraction is the p-value.

**Key Properties and Common Misconceptions:**
- P-values are always between 0 and 1 (they are probabilities)
- Small p-values (e.g., 0.01) indicate your data is very surprising under H₀—strong evidence against the null hypothesis
- Large p-values (e.g., 0.45) indicate your data is not surprising under H₀—the result is easily explained by random chance
- **Critical misconception:** The p-value is NOT the probability that H₀ is true. It is P(data | H₀), not P(H₀ | data). This is a subtle but crucial distinction. A p-value of 0.03 does not mean there's a 3% chance the null hypothesis is true.

**Calculating P-Values from Z-Scores:**

The p-value calculation depends on the direction of your alternative hypothesis:

```python
from scipy.stats import norm

# Left-tailed test: Hₐ claims parameter < hypothesized value
# We want P(Z ≤ observed z-score)
p_value = norm.cdf(z_score)

# Right-tailed test: Hₐ claims parameter > hypothesized value
# We want P(Z ≥ observed z-score)
p_value = 1 - norm.cdf(z_score)

# Two-tailed test: Hₐ claims parameter ≠ hypothesized value
# We want P(|Z| ≥ |observed z-score|), checking both tails
p_value = 2 * (1 - norm.cdf(abs(z_score)))
```

### 1.5 One-Tailed vs. Two-Tailed Tests

The "tails" refer to the extreme ends of the probability distribution. Your alternative hypothesis determines which tail(s) contain the "extreme" values that would count as evidence against H₀.

**Two-tailed tests** are used when you're looking for any difference from the hypothesized value, regardless of direction. For example, testing whether a medication changes blood pressure (it could increase or decrease it). You would reject H₀ if your result is extremely high OR extremely low.

**One-tailed tests** are used when you have a specific directional prediction. For example, testing whether a new drug reduces blood pressure (you specifically expect a decrease, not an increase). You would only reject H₀ if your result is extreme in the predicted direction.

| Alternative Hypothesis Language | Test Type | What Counts as Evidence Against H₀ |
|--------------------------------|-----------|-----------------------------------|
| "different from," "not equal to," "changed" | Two-tailed | Extreme values in either direction |
| "less than," "fewer," "decreased," "lower" | Left-tailed | Only extremely low values |
| "greater than," "more," "increased," "exceeds" | Right-tailed | Only extremely high values |

**Important:** The choice of one-tailed vs. two-tailed must be made before looking at the data, based on your research question. Choosing after seeing the data is a form of cheating that invalidates your p-value.

### 1.6 Statistical Significance and Decision Making

**Significance Level (α):** The significance level is a pre-specified threshold that determines how much evidence you need before rejecting H₀. It represents the maximum probability of making a Type I error (rejecting H₀ when it's actually true) that you're willing to accept. Importantly, you must set α before analyzing your data—otherwise, there's a temptation to choose a significance level that gives you the result you want.

**Decision Rule:**
- If p-value ≤ α: Reject H₀. The data provides sufficient evidence against the null hypothesis. We say the result is "statistically significant."
- If p-value > α: Fail to reject H₀. The data does not provide sufficient evidence against the null hypothesis. Note: this does NOT mean H₀ is true—only that we lack evidence to reject it.

**Common Significance Levels and Their Contexts:**
- α = 0.10 (10%): Lenient threshold, sometimes used in exploratory research where false negatives are costly
- α = 0.05 (5%): The most common standard in social sciences, business, and many scientific fields
- α = 0.01 (1%): More conservative, used when false positives are particularly costly
- α = 0.0000003 (5-sigma): Used in particle physics to claim discoveries, reflecting the extraordinary evidence required for extraordinary claims

### 1.7 Types of Errors

When making decisions based on hypothesis tests, there are four possible outcomes, two of which are errors:

| | H₀ is Actually TRUE | H₀ is Actually FALSE |
|---|---------------------|----------------------|
| **Reject H₀** | Type I Error (False Positive) | Correct Decision ✓ |
| **Fail to Reject H₀** | Correct Decision ✓ | Type II Error (False Negative) |

**Type I Error (False Positive):** Rejecting H₀ when it is actually true. This is like convicting an innocent person. The probability of a Type I error equals α, your significance level. By choosing α = 0.05, you accept a 5% chance of this error.

**Type II Error (False Negative):** Failing to reject H₀ when it is actually false. This is like letting a guilty person go free. The probability of a Type II error is denoted β. Statistical power (1 - β) is the probability of correctly rejecting a false H₀.

In practice, there's a trade-off: decreasing α (to reduce false positives) typically increases β (more false negatives), and vice versa. The only way to reduce both simultaneously is to increase sample size.

### 1.8 Confidence Intervals

A confidence interval provides a range of plausible values for the population parameter, rather than just a point estimate. For a significance level of α, the corresponding confidence level is (1 - α) × 100%. So if you're using α = 0.05, you would construct a 95% confidence interval.

The interpretation is subtle: a 95% confidence interval does NOT mean there's a 95% probability the true parameter is in the interval. Rather, if you repeated your study many times and constructed a confidence interval each time, 95% of those intervals would contain the true parameter.

```python
# 95% confidence interval using bootstrap quantiles
lower = np.quantile(bootstrap_distribution, 0.025)  # 2.5th percentile
upper = np.quantile(bootstrap_distribution, 0.975)  # 97.5th percentile
```

**Connection to Hypothesis Testing:** There's a direct link between confidence intervals and two-tailed hypothesis tests. If the hypothesized value falls outside the (1-α) confidence interval, you would reject H₀ at significance level α. Conversely, if the hypothesized value is inside the interval, you would fail to reject H₀.

---

## Chapter 2: Two-Sample and ANOVA Tests

### 2.1 Two-Sample Problems

Many research questions involve comparing two groups rather than comparing one group to a fixed value. For example: Do people who started coding as children earn more than those who started as adults? Do late shipments weigh more than on-time shipments? Is the new drug more effective than the old one?

In these situations, we're comparing the means (or other statistics) of two separate populations. The null hypothesis typically states that the two population means are equal, while the alternative hypothesis states they differ in some way.

**Hypotheses for Comparing Two Means:**
- H₀: μ₁ - μ₂ = 0 (the two population means are equal; no difference between groups)
- Hₐ: μ₁ - μ₂ ≠ 0, > 0, or < 0 (depending on your research question)

### 2.2 The T-Statistic

When comparing two means, we use the t-statistic instead of the z-score. The reason is that we must estimate the population standard deviation from our sample data, which introduces additional uncertainty that the z-score doesn't account for. The t-distribution is wider than the normal distribution (has "fatter tails"), reflecting this extra uncertainty.

**T-Statistic Formula for Two Independent Samples:**

$$t = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Let's break this down:
- The numerator is the observed difference between the two sample means, minus the hypothesized difference (zero under H₀)
- The denominator is the standard error of the difference, which depends on both groups' variability and sample sizes
- $\bar{x}_1, \bar{x}_2$ are the sample means for each group
- $s_1, s_2$ are the sample standard deviations (measuring variability within each group)
- $n_1, n_2$ are the sample sizes

```python
# Calculate the t-statistic manually
numerator = xbar_1 - xbar_2
denominator = np.sqrt((s_1**2 / n_1) + (s_2**2 / n_2))
t_stat = numerator / denominator
```

### 2.3 The T-Distribution

The t-distribution is a probability distribution that looks similar to the normal distribution but has heavier tails. It was developed by William Sealy Gosset (publishing under the pseudonym "Student") specifically for situations where the population standard deviation must be estimated from sample data.

**Key Properties:**
- The t-distribution is parameterized by "degrees of freedom" (df)
- Lower degrees of freedom → fatter tails (more probability in the extremes, reflecting greater uncertainty)
- Higher degrees of freedom → the t-distribution approaches the normal distribution
- At df = ∞, the t-distribution exactly equals the standard normal distribution

**Degrees of Freedom for Two-Sample T-Test:**
$$df = n_1 + n_2 - 2$$

The degrees of freedom represent the amount of independent information in your data. You lose one degree of freedom for each group mean you estimate, hence subtracting 2.

```python
from scipy.stats import t

# Calculate degrees of freedom
degrees_of_freedom = n_1 + n_2 - 2

# Calculate p-value from t-statistic
# Left-tailed test (Hₐ: μ₁ < μ₂)
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Right-tailed test (Hₐ: μ₁ > μ₂)
p_value = 1 - t.cdf(t_stat, df=degrees_of_freedom)
```

### 2.4 Paired T-Tests

Sometimes the two samples you're comparing are not independent—there's a natural pairing between observations. For example: the same students tested before and after a training program, the same counties' voting patterns in 2008 and 2012, or the same patients' blood pressure with and without medication.

In paired designs, each observation in one group has a specific counterpart in the other group. This pairing is valuable information that should not be ignored. A paired t-test accounts for this dependency by analyzing the differences within each pair rather than treating the groups as independent.

**Why Pairing Matters:** By focusing on the difference within each pair, you remove between-subject variability. Each person (or county, or patient) serves as their own control. This typically makes the test more powerful—you're more likely to detect a real effect because you've eliminated a source of noise.

**The Key Insight:** Convert the paired data into a single variable of differences, then perform a one-sample t-test asking whether the mean difference equals zero.

**Hypotheses for Paired Test:**
- H₀: μ_diff = 0 (the mean of the differences is zero; no systematic change)
- Hₐ: μ_diff ≠ 0, > 0, or < 0

**Degrees of Freedom:** df = n_pairs - 1 (you have n pairs, but estimate one mean from them)

```python
import pingouin

# Method 1: Calculate differences explicitly, then test
sample_data['diff'] = sample_data['var_2012'] - sample_data['var_2008']
results = pingouin.ttest(x=sample_data['diff'], y=0, alternative='two-sided')

# Method 2: Let pingouin handle the pairing automatically
results = pingouin.ttest(x=sample_data['var_2012'], 
                          y=sample_data['var_2008'], 
                          paired=True, 
                          alternative='two-sided')
```

**Warning:** Using an unpaired test on paired data wastes valuable information. The unpaired test treats each observation as independent, ignoring the within-pair correlation. This typically increases the standard error and makes it harder to detect real effects (more Type II errors).

### 2.5 ANOVA (Analysis of Variance)

What if you want to compare more than two groups? For example, comparing average compensation across five job satisfaction levels, or comparing product quality across four manufacturing plants. Running multiple t-tests (comparing each pair) is problematic because it inflates the probability of false positives.

ANOVA (Analysis of Variance) is designed to compare means across three or more groups simultaneously in a single test.

**Hypotheses:**
- H₀: μ₁ = μ₂ = μ₃ = ... = μₖ (all group means are equal; no differences among any groups)
- Hₐ: At least one group mean is different from the others

**Key Limitation:** ANOVA tells you whether there's evidence that at least one group differs from the others, but it doesn't tell you which specific groups differ. If ANOVA rejects H₀, you know something is different, but you need follow-up tests to identify where the differences lie.

```python
import pingouin

# Perform one-way ANOVA
# Comparing converted_comp (numeric outcome) across job_sat (categorical groups)
results = pingouin.anova(data=stack_overflow, 
                          dv='converted_comp',      # dependent variable (what we're measuring)
                          between='job_sat')        # grouping variable
print(results)
```

### 2.6 Pairwise Tests and Multiple Comparisons

After finding a significant ANOVA result, the natural follow-up is: which specific groups are different? This requires pairwise comparisons—testing each pair of groups against each other.

**The Multiple Comparisons Problem:** With k groups, you have k(k-1)/2 possible pairwise comparisons. Each comparison carries a risk of false positives (Type I error). When you run many tests, these risks accumulate, making it increasingly likely that at least one comparison will be falsely significant.

| Number of Groups | Number of Pairwise Tests | P(at least one false positive) at α=0.05 |
|------------------|-------------------------|------------------------------------------|
| 3 | 3 | ~14% |
| 5 | 10 | ~40% |
| 10 | 45 | ~90% |

With 10 groups, even if there are NO true differences, you have a 90% chance of finding at least one "significant" result just by chance!

**The Solution: Bonferroni Correction**

The Bonferroni correction adjusts p-values upward (or equivalently, divides α by the number of comparisons) to control the family-wise error rate—the probability of making at least one Type I error across all comparisons. This is a conservative correction that makes it harder to reject H₀, reducing false positives at the cost of potentially missing some real effects.

```python
# Pairwise t-tests with Bonferroni correction
results = pingouin.pairwise_tests(data=stack_overflow, 
                                   dv='converted_comp', 
                                   between='job_sat', 
                                   padjust='bonf')  # Apply Bonferroni correction
```

In the output, use the `p-corr` (corrected p-values) column for your decisions, not `p-unc` (uncorrected).

---

## Chapter 3: Proportion Tests

### 3.1 One-Sample Proportion Test

Often we're interested in the proportion of a population that has some characteristic—the proportion of customers who churn, the proportion of shipments that arrive late, the proportion of voters who support a candidate. The one-sample proportion test compares an observed sample proportion to a hypothesized population proportion.

For example, a company claims that only 6% of their shipments are late. You collect data and find that 8% of shipments in your sample were late. Is this difference evidence that the true rate exceeds 6%, or could it just be sampling variability?

**Hypotheses:**
- H₀: p = p₀ (the population proportion equals the hypothesized value)
- Hₐ: p ≠ p₀, > p₀, or < p₀

**Z-Score Formula for Proportions:**

$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}}$$

Where:
- $\hat{p}$ = sample proportion (what you observed, e.g., 0.08)
- $p_0$ = hypothesized proportion from H₀ (e.g., 0.06)
- $n$ = sample size

The denominator is the standard error of a proportion, derived from the binomial distribution. Notice it uses $p_0$ (not $\hat{p}$) because we calculate the standard error assuming H₀ is true.

**Why Z Instead of T for Proportions?**

For means, we use the t-distribution because we estimate the population standard deviation from the sample, creating extra uncertainty. For proportions, the standard error formula only uses $p_0$ (from the hypothesis) in the denominator—we're not estimating it from the data. This avoids the "double-dipping" problem that requires the t-distribution, so the normal (z) distribution is appropriate.

```python
from scipy.stats import norm

# Calculate sample proportion
p_hat = (df['late'] == 'Yes').mean()  # Proportion of "Yes" values
p_0 = 0.06  # Hypothesized proportion
n = len(df)

# Calculate z-score
numerator = p_hat - p_0
denominator = np.sqrt(p_0 * (1 - p_0) / n)
z_score = numerator / denominator

# Calculate p-value (right-tailed: testing if proportion > 6%)
p_value = 1 - norm.cdf(z_score)
```

### 3.2 Two-Sample Proportion Test

The two-sample proportion test compares the proportions of a characteristic between two independent groups. For example: Is the proportion of late shipments different between expensive freight and reasonable freight? Is the customer churn rate different between two regions? Is the success rate of a new treatment different from the old treatment?

In each case, we have two groups, and we're comparing the proportion of "successes" (however defined) in each group.

**Hypotheses:**
- H₀: p₁ - p₂ = 0 (the population proportions are equal in both groups)
- Hₐ: p₁ - p₂ ≠ 0, > 0, or < 0

**The Pooled Proportion:**

Under H₀, we assume both groups have the same true proportion. The pooled proportion combines both samples to get our best estimate of this common proportion:

$$\hat{p} = \frac{n_1\hat{p}_1 + n_2\hat{p}_2}{n_1 + n_2} = \frac{\text{total successes in both groups}}{\text{total observations in both groups}}$$

**Standard Error for Two Proportions:**

$$SE = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}$$

**Z-Score:**

$$z = \frac{\hat{p}_1 - \hat{p}_2}{SE}$$

```python
from statsmodels.stats.proportion import proportions_ztest

# Arrays of successes (e.g., number of late shipments) in each group
success_counts = np.array([45, 16])  # 45 late in group 1, 16 late in group 2

# Total observations in each group
n_obs = np.array([545, 455])

# Run two-sample proportion z-test
z_stat, p_value = proportions_ztest(count=success_counts, 
                                     nobs=n_obs, 
                                     alternative='larger')  # Testing if p1 > p2
```

### 3.3 Chi-Square Test of Independence

The chi-square test of independence examines whether there is an association between two categorical variables. Unlike proportion tests that compare a specific "success" rate, the chi-square test looks at the entire distribution across categories.

For example: Is there an association between vendor shipping terms (EXW, CIP, DDP, FCA) and whether freight costs are expensive or reasonable? If the variables are independent, the distribution of freight costs should be the same regardless of which shipping term is used. If they're associated, certain shipping terms might be linked to higher or lower costs.

**Statistical Independence:** Two categorical variables are independent when knowing the value of one variable gives you no information about the likely value of the other. Mathematically, the proportion of "successes" in the response variable is the same across all categories of the explanatory variable.

**Hypotheses:**
- H₀: The two variables are independent (no association)
- Hₐ: The two variables are associated (there is a relationship)

**The Chi-Square Statistic:**

The test compares observed counts in each cell of the contingency table to the counts we would expect if the variables were truly independent:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where O = observed count in each cell, E = expected count under independence.

Large values of χ² indicate that observed counts deviate substantially from what independence would predict, suggesting an association exists.

**Key Properties:**
- χ² is always non-negative (it's a sum of squared terms)
- Larger χ² indicates stronger evidence against independence
- Chi-square tests are always right-tailed (only large values are evidence against H₀)
- The order of variables doesn't matter—testing whether X is associated with Y gives the same result as testing whether Y is associated with X

```python
import pingouin

# Chi-square test of independence
expected, observed, stats = pingouin.chi2_independence(
    data=late_shipments, 
    x='freight_cost_group',    # One categorical variable
    y='vendor_inco_term'       # Another categorical variable
)

# Extract results for the Pearson chi-square test
print(stats[stats['test'] == 'pearson'])
```

**Visualization: Proportional Stacked Bar Plot**

A useful way to visualize potential associations is with a proportional stacked bar plot. If variables are independent, all bars would have the same proportional split:

```python
# Calculate proportions within each category
props = df.groupby('var_1')['var_2'].value_counts(normalize=True)
wide_props = props.unstack()
wide_props.plot(kind='bar', stacked=True)
```

### 3.4 Chi-Square Goodness of Fit Test

While the chi-square test of independence compares two categorical variables, the chi-square goodness of fit test compares the observed distribution of a single categorical variable to a hypothesized distribution.

For example: You hypothesize that vendor shipping terms should follow a specific distribution (5% CIP, 10% DDP, 75% EXW, 10% FCA). Does your sample match this hypothesized distribution, or is the actual distribution significantly different?

**Hypotheses:**
- H₀: The sample follows the hypothesized distribution
- Hₐ: The sample does not follow the hypothesized distribution

```python
from scipy.stats import chisquare

# Observed counts from your sample
observed_counts = df['category'].value_counts().sort_index()

# Expected counts from hypothesized proportions
hypothesized_props = [0.05, 0.10, 0.75, 0.10]
expected_counts = np.array(hypothesized_props) * len(df)

# Perform goodness of fit test
chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
```

---

## Chapter 4: Non-Parametric Tests

### 4.1 Assumptions of Parametric Tests

The tests we've covered so far (z-tests, t-tests, ANOVA) are called "parametric" tests because they assume the data follows a specific parametric distribution (typically normal). For these tests to produce valid results, several assumptions must hold:

**1. Random Sampling:** The sample must be randomly drawn from the population of interest. This ensures the sample is representative and allows us to generalize conclusions to the population. No statistical test can verify this assumption—you must understand how the data was collected.

**2. Independence of Observations:** Each observation must be independent of the others. The value of one observation should not influence or predict another (except in paired designs, which explicitly model the dependency). Violations can occur when data has hierarchical structure (students within classrooms), time series structure (today's value depends on yesterday's), or other clustering.

**3. Sufficient Sample Size:** The sample must be large enough for the Central Limit Theorem to apply. The CLT states that the sampling distribution of the mean approaches normality as sample size increases, regardless of the population distribution. If your sample is too small, the sampling distribution may not be approximately normal, and p-values calculated from normal or t-distributions will be incorrect.

**Minimum Sample Size Guidelines:**

| Test Type | Minimum Requirement |
|-----------|---------------------|
| One-sample t-test | 30 observations |
| Two-sample t-test | 30 observations per group |
| Paired t-test | 30 pairs |
| One-sample proportion | At least 10 successes AND at least 10 failures |
| Two-sample proportion | At least 10 successes AND 10 failures in each group |
| Chi-square test | Expected count of at least 5 in each cell |

**Sanity Check:** Generate a bootstrap distribution of your statistic and visualize it with a histogram. If it doesn't look bell-shaped (approximately normal), one of the assumptions may be violated.

### 4.2 When to Use Non-Parametric Tests

Non-parametric tests make fewer assumptions about the data and are appropriate when:
- Sample sizes are too small for the Central Limit Theorem to apply
- Data is heavily skewed or has extreme outliers
- Data is ordinal (ranks or ratings) rather than truly continuous
- You want robust results that don't depend on distributional assumptions

**Trade-off:** When parametric assumptions do hold, parametric tests have more statistical power—they're better at detecting real effects when they exist. By converting data to ranks, non-parametric tests discard some information contained in the exact values.

### 4.3 The Core Idea: Working with Ranks

Non-parametric tests avoid distributional assumptions by converting raw data values into ranks (positions from smallest to largest), then analyzing the ranks instead of the original values.

**Why This Helps:**

Ranks are immune to outliers and skewness. Suppose you have salaries: $40K, $45K, $50K, $55K, $10M. The $10M outlier would massively distort a mean-based test. But when converted to ranks (1, 2, 3, 4, 5), the outlier simply becomes "the largest"—still rank 5, not astronomically different from the others. Ranks are always uniformly distributed from 1 to n, regardless of how skewed the original data was.

```python
from scipy.stats import rankdata

data = [1, 15, 4, 12, 3]
ranks = rankdata(data)  # Returns [1, 5, 3, 4, 2]
# 1 is smallest (rank 1), 15 is largest (rank 5)
```

### 4.4 Wilcoxon Signed-Rank Test

**Non-parametric alternative to:** Paired t-test

**Use when:** You have two related samples (before/after measurements, matched pairs) but the sample size is small or the differences are not normally distributed.

**How It Works:**
1. Calculate the difference between each pair
2. Take the absolute value of each difference
3. Rank the absolute differences from smallest to largest
4. Compute T⁺ (sum of ranks for pairs where the difference was positive) and T⁻ (sum of ranks for pairs where the difference was negative)
5. The test statistic W = min(T⁺, T⁻)

If there's no systematic difference between the paired measurements, positive and negative differences should be equally likely, and T⁺ and T⁻ should be similar. If one is much smaller than expected, that's evidence of a systematic difference.

```python
import pingouin

results = pingouin.wilcoxon(x=df['measure_1'], 
                            y=df['measure_2'], 
                            alternative='two-sided')
```

### 4.5 Wilcoxon-Mann-Whitney Test

**Non-parametric alternative to:** Independent two-sample t-test

**Use when:** You're comparing two independent groups but sample sizes are small, data is heavily skewed, or you have ordinal data (like satisfaction ratings).

**How It Works:**
1. Combine all observations from both groups into one dataset
2. Rank all observations from smallest to largest
3. Sum the ranks for each group separately
4. Test whether one group's ranks are systematically higher or lower than expected under the null hypothesis (that both groups come from the same distribution)

If one group tends to have larger values, its members will cluster at higher ranks, producing a rank sum that's unexpectedly high.

```python
import pingouin

# Data must be pivoted to wide format for pingouin
wide_df = df.pivot(columns='group', values='numeric_var')

results = pingouin.mwu(x=wide_df['Group_A'], 
                       y=wide_df['Group_B'], 
                       alternative='two-sided')
```

### 4.6 Kruskal-Wallis Test

**Non-parametric alternative to:** One-way ANOVA

**Use when:** You're comparing more than two independent groups but sample sizes are small or data is not normally distributed.

**How It Works:**
Similar to the Wilcoxon-Mann-Whitney test, but extended to handle three or more groups. It ranks all observations, then tests whether the different groups have similar average ranks (as expected under H₀) or whether some groups have systematically higher or lower ranks.

```python
import pingouin

results = pingouin.kruskal(data=df, 
                           dv='numeric_var',    # The variable being compared
                           between='group_var')  # The grouping variable
```

---

## Summary: Test Selection Guide

### Complete Test Comparison Table

| Test | Purpose | Data Type | # of Groups | Parametric Alternative | Assumptions | When to Use |
|------|---------|-----------|-------------|------------------------|-------------|-------------|
| **One-sample z-test** | Test if a population proportion equals a hypothesized value | Categorical (binary outcome like Yes/No) | 1 | N/A | n × p₀ ≥ 10 and n × (1-p₀) ≥ 10 | Testing claims like "6% of shipments are late" |
| **One-sample t-test** | Test if a population mean equals a hypothesized value | Numeric (continuous measurements) | 1 | N/A | n ≥ 30 or normally distributed population | Testing claims like "average salary is $110,000" |
| **Two-sample t-test** | Compare means between two independent groups | Numeric | 2 (independent) | N/A | n ≥ 30 per group | Comparing salaries of two job categories |
| **Paired t-test** | Compare means of two related/matched samples | Numeric | 2 (paired) | N/A | n ≥ 30 pairs | Before/after comparisons, matched studies |
| **ANOVA** | Compare means across three or more independent groups | Numeric | 3+ | N/A | n ≥ 30 per group | Comparing salaries across 5 job satisfaction levels |
| **Two-sample proportion test** | Compare proportions between two independent groups | Categorical | 2 | N/A | ≥ 10 successes and failures per group | Comparing late rates between freight cost categories |
| **Chi-square independence** | Test for association between two categorical variables | Both categorical | Any number of categories | N/A | Expected count ≥ 5 per cell | Testing if shipping terms relate to freight costs |
| **Chi-square goodness of fit** | Test if observed distribution matches hypothesized distribution | Single categorical variable | 1 variable | N/A | Expected count ≥ 5 per category | Testing if data matches expected proportions |
| **Wilcoxon signed-rank** | Compare two related samples without normality assumption | Numeric or ordinal | 2 (paired) | Paired t-test | None | Paired comparison with small n or non-normal data |
| **Wilcoxon-Mann-Whitney** | Compare two independent groups without normality assumption | Numeric or ordinal | 2 (independent) | Two-sample t-test | None | Group comparison with skewed data or outliers |
| **Kruskal-Wallis** | Compare 3+ independent groups without normality assumption | Numeric or ordinal | 3+ | ANOVA | None | Multi-group comparison with small n or non-normal data |

### Decision Flowchart

```
What type of variable are you analyzing?
│
├── NUMERIC (continuous measurements like salary, weight, time)
│   │
│   ├── How many groups are you comparing?
│   │   │
│   │   ├── 1 group vs. a hypothesized value
│   │   │   └── One-sample t-test (or Wilcoxon signed-rank if small n)
│   │   │
│   │   ├── 2 groups
│   │   │   ├── Are the groups independent or paired?
│   │   │   │   ├── Independent → Two-sample t-test (or Mann-Whitney)
│   │   │   │   └── Paired → Paired t-test (or Wilcoxon signed-rank)
│   │   │
│   │   └── 3+ groups → ANOVA (or Kruskal-Wallis)
│   │
│   └── Are assumptions met (n ≥ 30 per group, roughly normal)?
│       ├── Yes → Use parametric test (more powerful)
│       └── No → Use non-parametric alternative (more robust)
│
└── CATEGORICAL (categories like Yes/No, product types, ratings)
    │
    ├── Testing one proportion against a hypothesized value?
    │   └── One-sample z-test for proportions
    │
    ├── Comparing proportions between 2 groups?
    │   └── Two-sample proportion z-test
    │
    └── Testing association between two categorical variables?
        └── Chi-square test of independence
```

---

## Key Formulas Reference

### Z-Score (General Form)
$$z = \frac{\text{observed statistic} - \text{hypothesized value}}{\text{standard error}}$$

### Z-Score for Single Proportion
$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

### T-Statistic for Two Independent Samples
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

### Pooled Proportion (Two-Sample)
$$\hat{p} = \frac{n_1\hat{p}_1 + n_2\hat{p}_2}{n_1 + n_2}$$

### Chi-Square Statistic
$$\chi^2 = \sum \frac{(\text{Observed} - \text{Expected})^2}{\text{Expected}}$$

### Degrees of Freedom
- One-sample t-test: df = n - 1
- Two-sample t-test: df = n₁ + n₂ - 2
- Paired t-test: df = n_pairs - 1
- Chi-square independence: df = (rows - 1) × (columns - 1)

---

## Python Libraries Quick Reference

```python
# Essential imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t, chisquare
import pingouin
from statsmodels.stats.proportion import proportions_ztest

# P-values from test statistics
norm.cdf(z)          # Standard normal CDF (for proportion tests)
t.cdf(t_stat, df)    # T-distribution CDF (for mean tests)

# Pingouin tests (user-friendly output as DataFrames)
pingouin.ttest()           # T-tests (paired and unpaired)
pingouin.anova()           # One-way ANOVA
pingouin.pairwise_tests()  # Pairwise comparisons with corrections
pingouin.chi2_independence() # Chi-square independence test
pingouin.wilcoxon()        # Wilcoxon signed-rank (paired, non-parametric)
pingouin.mwu()             # Mann-Whitney U (independent, non-parametric)
pingouin.kruskal()         # Kruskal-Wallis (3+ groups, non-parametric)

# Scipy tests
chisquare()               # Chi-square goodness of fit
proportions_ztest()       # One and two-sample proportion tests
```

---

## Applications in Finance and Quantitative Finance

### Test-by-Test Finance Applications

| Test | Finance/QF Application |
|------|------------------------|
| **One-sample t-test** | Testing whether a portfolio's average return differs from a benchmark. For example, testing if a fund's alpha (excess return) is significantly different from zero, or if a trading strategy's mean return exceeds the risk-free rate. |
| **Two-sample t-test** | Comparing returns between two different strategies, time periods, or market conditions. For example, testing if value stocks outperform growth stocks, or if returns during high-volatility periods differ from low-volatility periods. |
| **Paired t-test** | Analyzing the same assets under different conditions. For example, comparing stock performance before and after a policy change, or testing if the same portfolio performs differently under two different rebalancing strategies. |
| **ANOVA** | Comparing performance across multiple categories simultaneously. For example, testing if risk-adjusted returns differ across sectors, credit rating categories, or multiple trading strategies. |
| **One-sample proportion test** | Testing if a trading signal's hit rate (proportion of profitable trades) exceeds a threshold. For example, testing if a momentum signal predicts the correct direction more than 50% of the time. |
| **Two-sample proportion test** | Comparing success rates between strategies or conditions. For example, testing if the proportion of profitable trades is higher during bull markets than bear markets, or comparing default rates between two loan categories. |
| **Chi-square independence** | Testing associations between categorical variables in financial data. For example, testing if there's an association between credit rating changes and economic conditions, or between sector membership and likelihood of bankruptcy. |
| **Chi-square goodness of fit** | Testing if return distributions match theoretical assumptions. For example, testing whether VaR model assumptions hold by checking if violation rates match expected frequencies. |
| **Wilcoxon signed-rank** | Comparing paired financial data when returns are highly skewed or sample sizes are small. Useful for hedge fund analysis where sample sizes are limited and outliers are common. |
| **Wilcoxon-Mann-Whitney** | Comparing distributions when normality assumptions are questionable. For example, comparing median returns between two strategies when returns are heavily skewed by extreme events. |
| **Kruskal-Wallis** | Comparing multiple portfolios or strategies when return distributions are non-normal. Useful for analyzing alternative investments with non-standard return patterns. |

---

## Common Pitfalls to Avoid

**1. Setting α after seeing the data:** Always decide your significance level before running the test. Choosing α after seeing results to achieve the outcome you want is called "p-hacking" and invalidates your conclusions.

**2. Confusing p-value interpretation:** The p-value is P(data this extreme | H₀ true), NOT P(H₀ is true | data). A p-value of 0.03 doesn't mean there's a 3% chance H₀ is true.

**3. Using unpaired tests on paired data:** This ignores the within-pair correlation, wastes information, reduces statistical power, and increases the chance of false negatives.

**4. Ignoring multiple comparisons:** Running many tests without correction inflates the false positive rate. With 20 tests at α = 0.05, you expect one false positive even when all null hypotheses are true.

**5. Treating non-significant as "no effect":** Failing to reject H₀ doesn't prove H₀ is true. You may simply lack statistical power to detect a real effect. Absence of evidence is not evidence of absence.

**6. Ignoring assumptions:** Parametric tests require sufficient sample sizes and approximate normality. Violating assumptions can make p-values unreliable. When in doubt, use non-parametric alternatives or verify assumptions with diagnostics.

**7. Forgetting practical significance:** A statistically significant result may be too small to matter practically. With large samples, even tiny effects become statistically significant. Always consider effect sizes alongside p-values: is the difference large enough to care about?
