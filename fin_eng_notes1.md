# Introduction to Financial Engineering — Quantitative Prerequisites

> **Course Notes** | Columbia University — Financial Engineering & Risk Management\
> Comprehensive lecture notes covering probability, stochastic processes, linear algebra, and optimisation.

---

## Table of Contents

1. [Probability Review](#1-probability-review)
   - [Cumulative Distribution Functions & Probability Mass Functions](#11-cumulative-distribution-functions--probability-mass-functions)
   - [Expectation and Variance](#12-expectation-and-variance)
   - [The Binomial Distribution](#13-the-binomial-distribution)
   - [The Poisson Distribution](#14-the-poisson-distribution)
   - [Bayes' Theorem](#15-bayes-theorem)
   - [Continuous Random Variables & the Normal Distribution](#16-continuous-random-variables--the-normal-distribution)
   - [The Log-Normal Distribution](#17-the-log-normal-distribution)
2. [Conditional Expectation & Variance](#2-conditional-expectation--variance)
   - [The Conditional Expectation Identity](#21-the-conditional-expectation-identity)
   - [The Conditional Variance Identity](#22-the-conditional-variance-identity)
   - [Application: Random Sums of Random Variables](#23-application-random-sums-of-random-variables)
   - [Indicator Functions — Chickens and Eggs](#24-indicator-functions--chickens-and-eggs)
3. [Multivariate Distributions](#3-multivariate-distributions)
   - [Joint and Marginal CDFs](#31-joint-and-marginal-cdfs)
   - [Conditional Distributions](#32-conditional-distributions)
   - [Independence and Its Implications](#33-independence-and-its-implications)
   - [Mean Vector and Covariance Matrix](#34-mean-vector-and-covariance-matrix)
4. [The Multivariate Normal Distribution](#4-the-multivariate-normal-distribution)
   - [Definition and PDF](#41-definition-and-pdf)
   - [Moment Generating Function](#42-moment-generating-function)
   - [Marginal and Conditional Distributions](#43-marginal-and-conditional-distributions)
5. [Martingales](#5-martingales)
   - [Definition and Intuition](#51-definition-and-intuition)
   - [Random Walk Example](#52-random-walk-example)
   - [The Martingale Betting Strategy](#53-the-martingale-betting-strategy)
   - [Pólya's Urn](#54-pólyas-urn)
6. [Brownian Motion](#6-brownian-motion)
   - [Definition and Properties](#61-definition-and-properties)
   - [Standard Brownian Motion](#62-standard-brownian-motion)
   - [Information Filtrations and the Independent Increments Property](#63-information-filtrations-and-the-independent-increments-property)
   - [A Calculation with Brownian Motion](#64-a-calculation-with-brownian-motion)
7. [Geometric Brownian Motion](#7-geometric-brownian-motion)
   - [Definition](#71-definition)
   - [Simulation and the Recursive Representation](#72-simulation-and-the-recursive-representation)
   - [Expected Value of GBM](#73-expected-value-of-gbm)
   - [Key Properties and Relevance to Stock Prices](#74-key-properties-and-relevance-to-stock-prices)
8. [Vectors](#8-vectors)
   - [Definitions: Row and Column Vectors](#81-definitions-row-and-column-vectors)
   - [Linear Combinations, Dependence, and Independence](#82-linear-combinations-dependence-and-independence)
   - [Basis and the Standard Basis](#83-basis-and-the-standard-basis)
   - [Norms (Length of a Vector)](#84-norms-length-of-a-vector)
   - [Inner Products and Angles](#85-inner-products-and-angles)
9. [Matrices](#9-matrices)
   - [Definitions and Notation](#91-definitions-and-notation)
   - [Transpose](#92-transpose)
   - [Matrix Multiplication](#93-matrix-multiplication)
   - [Linear Functions and Constraints](#94-linear-functions-and-constraints)
   - [Rank and Range](#95-rank-and-range)
   - [Inverse of a Matrix](#96-inverse-of-a-matrix)
10. [Linear Optimisation and Hedging](#10-linear-optimisation-and-hedging)
    - [The Hedging Problem Setup](#101-the-hedging-problem-setup)
    - [Portfolio Payoffs and the Role of Rank](#102-portfolio-payoffs-and-the-role-of-rank)
    - [Linear Programming and Duality](#103-linear-programming-and-duality)
    - [Lagrangian Relaxation (Linear Case)](#104-lagrangian-relaxation-linear-case)
11. [Non-Linear Optimisation](#11-non-linear-optimisation)
    - [Unconstrained Optimisation: Gradient and Hessian](#111-unconstrained-optimisation-gradient-and-hessian)
    - [Convex Functions](#112-convex-functions)
    - [Constrained Optimisation and Lagrange Multipliers](#113-constrained-optimisation-and-lagrange-multipliers)
    - [Application: Portfolio Selection (Mean–Variance)](#114-application-portfolio-selection-meanvariance)

---

## 1. Probability Review

### 1.1 Cumulative Distribution Functions & Probability Mass Functions

The **cumulative distribution function (CDF)** of a random variable $`X`$ is denoted $`F(x)`$ and defined as

```math
F(x) = P(X \leq x)
```

This function tells us the probability that the random variable $`X`$ takes on a value less than or equal to some threshold $`x`$. The CDF is a non-decreasing function that goes from 0 to 1. Every random variable — discrete or continuous — possesses a CDF.

For **discrete random variables**, we additionally define a **probability mass function (PMF)** $`p`$ that satisfies two properties:

1. $`p(x) \geq 0`$ for all $`x`$.
2. For all events $`A`$, the probability that $`X`$ falls in $`A`$ is given by summing the PMF over the outcomes in $`A`$:

```math
P(X \in A) = \sum_{x \in A} p(x)
```

The PMF gives the probability of each individual outcome, while the CDF gives the cumulative probability up to and including a given point.

### 1.2 Expectation and Variance

The **expected value** (or mean) of a discrete random variable $`X`$ is the probability-weighted average of all its possible values:

```math
E[X] = \sum_{i} x_i \, p(x_i)
```

**Example — A Fair Die.** Suppose we roll a fair six-sided die, so $`X \in \{1, 2, 3, 4, 5, 6\}`$ each with probability $`\tfrac{1}{6}`$. Then

```math
P(X \geq 4) = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{1}{2}
```

and the expected value is

```math
E[X] = \frac{1}{6}(1 + 2 + 3 + 4 + 5 + 6) = 3.5
```

The **variance** of $`X`$ measures how spread out its values are around the mean:

```math
\text{Var}(X) = E\!\left[(X - E[X])^2\right]
```

An equivalent and often more convenient computational form is

```math
\text{Var}(X) = E[X^2] - \left(E[X]\right)^2
```

This second form is obtained by expanding the square inside the first definition, distributing the expectation, and simplifying.

### 1.3 The Binomial Distribution

A random variable $`X`$ has a **binomial distribution** with parameters $`n`$ (number of trials) and $`p`$ (probability of success on each trial) — written $`X \sim \text{Bin}(n, p)`$ — if

```math
P(X = r) = \binom{n}{r} p^r (1 - p)^{n - r}, \quad r = 0, 1, \ldots, n
```

where the binomial coefficient is

```math
\binom{n}{r} = \frac{n!}{r!(n - r)!}
```

The binomial distribution arises naturally when we count the number of successes in $`n`$ **independent** trials, each with the same probability of success $`p`$. The classic example is counting heads in $`n`$ independent coin tosses.

**Mean and variance:**

```math
E[X] = np, \qquad \text{Var}(X) = np(1 - p)
```

**Application — Evaluating Fund Manager Skill.** Suppose a fund manager either outperforms or underperforms the market each year with probability $`p`$ and $`1 - p`$ respectively, independently across years. She has outperformed in 8 out of 10 years. If she had **no skill** (i.e. $`p = 0.5`$), we model the number of outperforming years as $`X \sim \text{Bin}(10, 0.5)`$ and ask: how likely is a record at least this good?

```math
P(X \geq 8) = \sum_{r=8}^{10} \binom{10}{r} \left(\frac{1}{2}\right)^{10}
```

If this probability is very small, we have evidence that the manager may possess genuine skill rather than just having been lucky. However, this analysis opens deeper questions — if there are $`M`$ fund managers and none have skill, how well should the *best* one be expected to do? This involves order statistics of the binomial distribution and is revisited later in the course.

### 1.4 The Poisson Distribution

A random variable $`X`$ has a **Poisson distribution** with parameter $`\lambda > 0`$ — written $`X \sim \text{Poisson}(\lambda)`$ — if

```math
P(X = r) = \frac{\lambda^r e^{-\lambda}}{r!}, \quad r = 0, 1, 2, \ldots
```

The Poisson distribution is commonly used to model the number of events occurring in a fixed interval of time or space (e.g. the number of defaults in a bond portfolio over a year, the number of trades arriving in an hour).

**Mean and variance are both equal to** $`\lambda`$:

```math
E[X] = \lambda, \qquad \text{Var}(X) = \lambda
```

**Proof that** $`E[X] = \lambda`$. Starting from the definition:

```math
E[X] = \sum_{r=0}^{\infty} r \cdot \frac{\lambda^r e^{-\lambda}}{r!}
```

The $`r = 0`$ term contributes zero, so we start from $`r = 1`$. Cancelling $`r`$ with the first factor in $`r!`$:

```math
E[X] = \sum_{r=1}^{\infty} \frac{\lambda^r e^{-\lambda}}{(r-1)!} = \lambda \sum_{r=1}^{\infty} \frac{\lambda^{r-1} e^{-\lambda}}{(r-1)!}
```

Substituting $`k = r - 1`$, the remaining sum becomes $`\sum_{k=0}^{\infty} \frac{\lambda^k e^{-\lambda}}{k!} = 1`$ (the total probability of a Poisson random variable). Therefore $`E[X] = \lambda`$.

### 1.5 Bayes' Theorem

Let $`A`$ and $`B`$ be events with $`P(B) > 0`$. The **conditional probability** of $`A`$ given $`B`$ is

```math
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
```

Since $`P(A \cap B) = P(B \mid A) \cdot P(A)`$, we can rewrite this as **Bayes' theorem**:

```math
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
```

If $`\{A_1, A_2, \ldots\}`$ forms a **partition** of the sample space (meaning the $`A_j`$'s are mutually exclusive and exhaustive — exactly one must occur), then the denominator can be expanded using the **law of total probability**:

```math
P(B) = \sum_{j} P(B \mid A_j) \, P(A_j)
```

giving the full form of Bayes' theorem:

```math
P(A_i \mid B) = \frac{P(B \mid A_i) \, P(A_i)}{\sum_{j} P(B \mid A_j) \, P(A_j)}
```

**Example — Two Dice.** Let $`Y_1`$ and $`Y_2`$ be outcomes of rolling two fair dice, and $`X = Y_1 + Y_2`$. We want to find $`P(Y_1 \geq 4 \mid X \geq 8)`$.

Using the definition of conditional probability:

```math
P(Y_1 \geq 4 \mid X \geq 8) = \frac{P(Y_1 \geq 4 \;\text{and}\; X \geq 8)}{P(X \geq 8)}
```

By enumerating the $`6 \times 6 = 36`$ equally likely outcomes, one finds 12 outcomes satisfy both conditions and 15 outcomes satisfy $`X \geq 8`$. Therefore:

```math
P(Y_1 \geq 4 \mid X \geq 8) = \frac{12/36}{15/36} = \frac{12}{15} = \frac{4}{5}
```

### 1.6 Continuous Random Variables & the Normal Distribution

A continuous random variable $`X`$ has a **probability density function (PDF)** $`f`$ satisfying:

1. $`f(x) \geq 0`$ for all $`x`$.
2. For all events $`A`$: $`P(X \in A) = \int_A f(y) \, dy`$.

The CDF and PDF are related by

```math
F(x) = \int_{-\infty}^{x} f(y) \, dy
```

An important intuition is that the probability of $`X`$ lying in a small interval around $`x`$ is approximately:

```math
P\left(x - \frac{\varepsilon}{2} \leq X \leq x + \frac{\varepsilon}{2}\right) \approx f(x) \cdot \varepsilon
```

This approximation improves as $`\varepsilon \to 0`$. Note that for continuous random variables, the density $`f(x)`$ is **not** a probability — it is a probability *density* whose integral over a region gives a probability.

**The Normal (Gaussian) Distribution.** We write $`X \sim N(\mu, \sigma^2)`$ if $`X`$ has the PDF

```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
```

The mean is $`\mu`$, the variance is $`\sigma^2`$, and the mode (peak of the density) also occurs at $`\mu`$. A crucial rule of thumb: approximately **95% of the probability** lies within $`\pm 2`$ standard deviations of the mean:

```math
P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.95
```

The normal distribution is the most important distribution in finance. It arises from the Central Limit Theorem and underpins many models, though it has well-known weaknesses — particularly in capturing the heavy tails observed in financial return data.

### 1.7 The Log-Normal Distribution

We say $`X`$ has a **log-normal distribution** with parameters $`\mu`$ and $`\sigma^2`$ if

```math
\ln(X) \sim N(\mu, \sigma^2)
```

Equivalently, $`X = e^Y`$ where $`Y \sim N(\mu, \sigma^2)`$.

The mean and variance of the log-normal are:

```math
E[X] = e^{\mu + \sigma^2/2}, \qquad \text{Var}(X) = e^{2\mu + \sigma^2}\left(e^{\sigma^2} - 1\right)
```

The log-normal distribution is strictly positive (since $`e^Y > 0`$ always), which makes it natural for modelling quantities that cannot be negative — such as stock prices. It plays a central role in the Black–Scholes model for option pricing.

---

## 2. Conditional Expectation & Variance

### 2.1 The Conditional Expectation Identity

Let $`X`$ and $`Y`$ be two random variables. The **conditional expectation identity** (also known as the **law of iterated expectations** or the **tower property**) states:

```math
E[X] = E\big[E[X \mid Y]\big]
```

The key insight is that $`E[X \mid Y]`$ is itself a random variable — it is a function of $`Y`$. We could write it as $`g(Y)`$ for some function $`g`$. The outer expectation then averages $`g(Y)`$ over all possible values of $`Y`$, recovering the unconditional mean of $`X`$.

This identity is extraordinarily useful: to compute $`E[X]`$, we can first condition on another random variable $`Y`$ (which may simplify the computation), and then average over $`Y`$.

### 2.2 The Conditional Variance Identity

The **conditional variance identity** (also called the **law of total variance** or **Eve's law**) states:

```math
\text{Var}(X) = \text{Var}\big(E[X \mid Y]\big) + E\big[\text{Var}(X \mid Y)\big]
```

In words: the total variance of $`X`$ decomposes into the *variance of the conditional mean* plus the *mean of the conditional variance*. The first term captures how much the average of $`X`$ varies across different values of $`Y`$. The second term captures the average residual variability of $`X`$ once $`Y`$ is known.

Again, both $`E[X \mid Y]`$ and $`\text{Var}(X \mid Y)`$ are functions of $`Y`$ — hence random variables — and we can write them as $`g(Y)`$ and $`h(Y)`$ respectively.

### 2.3 Application: Random Sums of Random Variables

Let $`W = X_1 + X_2 + \cdots + X_N`$, where the $`X_i`$'s are **i.i.d.** with mean $`\mu_X`$ and variance $`\sigma_X^2`$, and $`N`$ is itself a random variable **independent** of the $`X_i`$'s.

**Expected value of** $`W`$. Using the conditional expectation identity:

```math
E[W] = E\big[E[W \mid N]\big]
```

Given $`N = n`$, the sum $`W`$ is just $`n`$ i.i.d. random variables, so $`E[W \mid N] = N \mu_X`$. Taking the outer expectation:

```math
E[W] = E[N \mu_X] = \mu_X \, E[N]
```

**Variance of** $`W`$. Using the conditional variance identity:

```math
\text{Var}(W) = \text{Var}\big(E[W \mid N]\big) + E\big[\text{Var}(W \mid N)\big]
```

We computed $`E[W \mid N] = N\mu_X`$ and similarly $`\text{Var}(W \mid N) = N\sigma_X^2`$ (the variance of $`n`$ i.i.d. variables is $`n\sigma_X^2`$). Therefore:

```math
\text{Var}(W) = \text{Var}(N\mu_X) + E[N\sigma_X^2] = \mu_X^2 \,\text{Var}(N) + \sigma_X^2 \, E[N]
```

### 2.4 Indicator Functions — Chickens and Eggs

**Setup.** A hen lays $`N`$ eggs where $`N \sim \text{Poisson}(\lambda)`$. Each egg hatches independently with probability $`p`$. Let $`K`$ be the total number of chickens.

**Indicator functions.** We introduce the indicator $`\mathbf{1}_{H_i}`$ for the event that the $`i`$-th egg hatches:

```math
\mathbf{1}_{H_i} = \begin{cases} 1 & \text{if the } i\text{-th egg hatches} \\ 0 & \text{otherwise} \end{cases}
```

Then $`K = \sum_{i=1}^{N} \mathbf{1}_{H_i}`$. The expected value of each indicator is simply $`E[\mathbf{1}_{H_i}] = p`$.

**Conditional expectation.** Given $`N = n`$:

```math
E[K \mid N = n] = \sum_{i=1}^{n} E[\mathbf{1}_{H_i}] = np
```

**Unconditional expectation.** By the conditional expectation identity:

```math
E[K] = E\big[E[K \mid N]\big] = E[Np] = p \, E[N] = p\lambda
```

Indicator functions will reappear later in the course when modelling default events in bond portfolios: $`\mathbf{1}_{D_i}`$ will indicate whether the $`i`$-th bond in a basket defaults, and similar techniques will be used to compute the expected number of defaults.

---

## 3. Multivariate Distributions

### 3.1 Joint and Marginal CDFs

Let $`\mathbf{X} = (X_1, X_2, \ldots, X_n)`$ be a vector of random variables. The **joint CDF** is

```math
F_{\mathbf{X}}(\mathbf{x}) = P(X_1 \leq x_1, \; X_2 \leq x_2, \; \ldots, \; X_n \leq x_n)
```

The **marginal CDF** of a single component $`X_i`$ is obtained by setting all other arguments to $`\infty`$:

```math
F_{X_i}(x_i) = F_{\mathbf{X}}(\infty, \ldots, \infty, x_i, \infty, \ldots, \infty)
```

This extends naturally to joint marginals: the joint CDF of $`(X_i, X_j)`$ is recovered by placing $`\infty`$ in every argument except the $`i`$-th and $`j`$-th.

If a **joint PDF** $`f_{\mathbf{X}}`$ exists, then the joint CDF can be written as

```math
F_{\mathbf{X}}(\mathbf{x}) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_n} f_{\mathbf{X}}(u_1, \ldots, u_n) \, du_n \cdots du_1
```

### 3.2 Conditional Distributions

Partition the vector $`\mathbf{X}`$ into two blocks: $`\mathbf{X}_1 = (X_1, \ldots, X_k)`$ and $`\mathbf{X}_2 = (X_{k+1}, \ldots, X_n)`$. The **conditional PDF** of $`\mathbf{X}_2`$ given $`\mathbf{X}_1 = \mathbf{x}_1`$ is

```math
f_{\mathbf{X}_2 \mid \mathbf{X}_1}(\mathbf{x}_2 \mid \mathbf{x}_1) = \frac{f_{\mathbf{X}}(\mathbf{x}_1, \mathbf{x}_2)}{f_{\mathbf{X}_1}(\mathbf{x}_1)}
```

where the denominator is the marginal PDF of $`\mathbf{X}_1`$. The conditional CDF is then obtained by integrating the conditional PDF over the appropriate region.

### 3.3 Independence and Its Implications

The collection $`\mathbf{X} = (X_1, \ldots, X_n)`$ is **independent** if and only if the joint CDF factorises into the product of marginals:

```math
F_{\mathbf{X}}(\mathbf{x}) = \prod_{i=1}^{n} F_{X_i}(x_i)
```

Equivalently, if a joint PDF exists, independence means $`f_{\mathbf{X}}(\mathbf{x}) = \prod_{i=1}^{n} f_{X_i}(x_i)`$.

When $`\mathbf{X}_1`$ and $`\mathbf{X}_2`$ are independent, the conditional PDF of $`\mathbf{X}_2`$ given $`\mathbf{X}_1`$ reduces to just the marginal PDF of $`\mathbf{X}_2`$. In other words, knowing $`\mathbf{X}_1`$ tells you nothing about $`\mathbf{X}_2`$.

**Key implications of independence.** Let $`X`$ and $`Y`$ be independent. Then:

1. For any events $`A, B`$: $`P(X \in A, \; Y \in B) = P(X \in A) \cdot P(Y \in B)`$.
2. For any functions $`f, g`$: $`E[f(X) \cdot g(Y)] = E[f(X)] \cdot E[g(Y)]`$.

Property (2) implies (1) by taking $`f = \mathbf{1}_A`$ and $`g = \mathbf{1}_B`$.

More generally, if $`X_1, \ldots, X_n`$ are independent, then $`E\left[\prod_{i=1}^{n} f_i(X_i)\right] = \prod_{i=1}^{n} E[f_i(X_i)]`$.

**Conditional independence.** We say $`X`$ and $`Y`$ are **conditionally independent given** $`Z`$ if

```math
E[f(X) \cdot g(Y) \mid Z] = E[f(X) \mid Z] \cdot E[g(Y) \mid Z] \quad \text{for all } f, g
```

This concept is central to credit risk modelling. Let $`D_i`$ be the event that the $`i`$-th bond in a portfolio defaults. It is unreasonable to assume the $`D_i`$'s are independent — macroeconomic or industry-specific shocks cause correlated defaults. However, conditional on some latent factor $`Z`$ (representing, say, an industry health factor), the defaults may be conditionally independent:

```math
P(D_1, \ldots, D_n \mid Z) = \prod_{i=1}^{n} P(D_i \mid Z)
```

This decomposition is the foundation of the **Gaussian copula model** for pricing CDOs.

### 3.4 Mean Vector and Covariance Matrix

For a random vector $`\mathbf{X} = (X_1, \ldots, X_n)^\top`$:

**Mean vector:**

```math
\boldsymbol{\mu} = E[\mathbf{X}] = \begin{pmatrix} E[X_1] \\ \vdots \\ E[X_n] \end{pmatrix}
```

**Covariance matrix:**

```math
\Sigma = E\left[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top\right]
```

This is an $`n \times n`$ matrix whose $`(i, j)`$-th element is $`\text{Cov}(X_i, X_j)`$. The diagonal elements are the variances $`\text{Var}(X_i) \geq 0`$.

**Properties of the covariance matrix:**

The matrix $`\Sigma`$ is **symmetric**, meaning $`\Sigma_{ij} = \text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i) = \Sigma_{ji}`$. It is also **positive semi-definite**, meaning $`\mathbf{x}^\top \Sigma \mathbf{x} \geq 0`$ for all $`\mathbf{x} \in \mathbb{R}^n`$. The diagonal entries are non-negative since they are variances.

The **correlation matrix** $`\rho`$ has $`(i, j)`$-th element

```math
\rho_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sqrt{\text{Var}(X_i) \cdot \text{Var}(X_j)}}
```

It is also symmetric, positive semi-definite, and has ones along the diagonal.

**Linear transformations.** For a $`k \times n`$ matrix $`A`$ and a $`k \times 1`$ vector $`\mathbf{a}`$:

```math
E[A\mathbf{X} + \mathbf{a}] = A\boldsymbol{\mu} + \mathbf{a}, \qquad \text{Cov}(A\mathbf{X} + \mathbf{a}) = A \Sigma A^\top
```

A useful scalar special case:

```math
\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y) + 2ab\,\text{Cov}(X, Y)
```

**Important caveat:** If $`X`$ and $`Y`$ are independent then $`\text{Cov}(X, Y) = 0`$, **but the converse is not true in general**. Zero covariance does not imply independence — it only implies the absence of *linear* dependence.

---

## 4. The Multivariate Normal Distribution

### 4.1 Definition and PDF

An $`n`$-dimensional random vector $`\mathbf{X}`$ is **multivariate normal** — written $`\mathbf{X} \sim N_n(\boldsymbol{\mu}, \Sigma)`$ — if its PDF is

```math
f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} \, |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
```

The **standard multivariate normal** has $`\boldsymbol{\mu} = \mathbf{0}`$ and $`\Sigma = I_n`$ (the identity matrix). In this case the joint PDF factorises as

```math
f_{\mathbf{X}}(\mathbf{x}) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}} e^{-x_i^2 / 2}
```

which means the components $`X_1, \ldots, X_n`$ are **independent** standard normal random variables.

### 4.2 Moment Generating Function

The moment generating function (MGF) of $`\mathbf{X} \sim N_n(\boldsymbol{\mu}, \Sigma)`$ is

```math
\phi_{\mathbf{X}}(\mathbf{s}) = E\left[e^{\mathbf{s}^\top \mathbf{X}}\right] = \exp\left(\mathbf{s}^\top \boldsymbol{\mu} + \frac{1}{2} \mathbf{s}^\top \Sigma \mathbf{s}\right)
```

In the scalar case ($`X \sim N(\mu, \sigma^2)`$), this reduces to the familiar

```math
E[e^{sX}] = \exp\left(s\mu + \frac{1}{2}\sigma^2 s^2\right)
```

This MGF result will be used repeatedly — for instance, in computing expected values involving Geometric Brownian Motion.

### 4.3 Marginal and Conditional Distributions

Partition $`\mathbf{X}`$ into $`\mathbf{X}_1`$ ($`k`$ components) and $`\mathbf{X}_2`$ ($`n - k`$ components), with the mean and covariance partitioned accordingly:

```math
\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \qquad \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}
```

**Marginal distributions** of a multivariate normal are themselves normal:

```math
\mathbf{X}_1 \sim N_k(\boldsymbol{\mu}_1, \Sigma_{11}), \qquad \mathbf{X}_2 \sim N_{n-k}(\boldsymbol{\mu}_2, \Sigma_{22})
```

**Conditional distributions** (assuming $`\Sigma`$ is positive definite) are also normal:

```math
\mathbf{X}_2 \mid \mathbf{X}_1 = \mathbf{x}_1 \;\sim\; N_{n-k}\left(\boldsymbol{\mu}_{2 \cdot 1}, \; \Sigma_{2 \cdot 1}\right)
```

where

```math
\boldsymbol{\mu}_{2 \cdot 1} = \boldsymbol{\mu}_2 + \Sigma_{21} \Sigma_{11}^{-1}(\mathbf{x}_1 - \boldsymbol{\mu}_1)
```

```math
\Sigma_{2 \cdot 1} = \Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}
```

**Intuition.** Observing $`\mathbf{X}_1 = \mathbf{x}_1`$ shifts the conditional mean of $`\mathbf{X}_2`$ away from $`\boldsymbol{\mu}_2`$ by a correction term proportional to how much $`\mathbf{x}_1`$ deviates from $`\boldsymbol{\mu}_1`$, scaled by the regression coefficient $`\Sigma_{21}\Sigma_{11}^{-1}`$. Meanwhile, the conditional variance $`\Sigma_{2 \cdot 1}`$ is smaller than the marginal variance $`\Sigma_{22}`$, reflecting the reduction in uncertainty about $`\mathbf{X}_2`$ once $`\mathbf{X}_1`$ is known.

**Linear combinations** of multivariate normals remain normal: if $`\mathbf{X} \sim N_n(\boldsymbol{\mu}, \Sigma)`$ then

```math
A\mathbf{X} + \mathbf{a} \sim N_k(A\boldsymbol{\mu} + \mathbf{a}, \; A\Sigma A^\top)
```

---

## 5. Martingales

### 5.1 Definition and Intuition

A random process $`\{X_n\}`$ is a **martingale** with respect to an information filtration $`\{\mathcal{F}_n\}`$ and probability measure $`P`$ if:

1. **Integrability:** $`E[|X_n|] < \infty`$ for all $`n`$.
2. **Martingale property:** $`E[X_{n+m} \mid \mathcal{F}_n] = X_n`$ for all $`n, m \geq 0`$.

The **information filtration** $`\mathcal{F}_n`$ represents all information available at time $`n`$. Typically, $`\mathcal{F}_n`$ is generated by $`(X_1, X_2, \ldots, X_n)`$ — in other words, at time $`n`$ we have observed the process up to and including time $`n`$.

Condition (2) is the heart of the definition: the best prediction of the future value of the process, given everything we know now, is simply the current value. This captures the notion of a **fair game** — on average, the process neither drifts up nor drifts down.

Related concepts:

- **Submartingale:** Replace condition (2) with $`E[X_{n+m} \mid \mathcal{F}_n] \geq X_n`$ (the process tends to go up on average).
- **Supermartingale:** Replace condition (2) with $`E[X_{n+m} \mid \mathcal{F}_n] \leq X_n`$ (the process tends to go down on average).

A martingale is both a submartingale and a supermartingale.

### 5.2 Random Walk Example

Let $`S_n = \sum_{i=1}^{n} X_i`$ where the $`X_i`$'s are i.i.d. with mean $`\mu`$. Define

```math
M_n = S_n - n\mu
```

Then $`M_n`$ is a martingale. To verify:

```math
E[M_{n+m} \mid \mathcal{F}_n] = E\left[S_{n+m} - (n+m)\mu \mid \mathcal{F}_n\right]
```

Since the first $`n`$ terms of $`S_{n+m}`$ are known at time $`n`$:

```math
= \sum_{i=1}^{n} X_i + E\left[\sum_{i=n+1}^{n+m} X_i\right] - (n+m)\mu = S_n + m\mu - (n+m)\mu = S_n - n\mu = M_n
```

The i.i.d. assumption is essential: knowing the first $`n`$ values tells us nothing about future increments, each of which has expected value $`\mu`$. By subtracting the deterministic drift $`n\mu`$, we remove the systematic trend and obtain a fair game.

### 5.3 The Martingale Betting Strategy

Consider the classic **doubling strategy** (also called the Martingale betting strategy in gambling). Let $`X_1, X_2, \ldots`$ be i.i.d. with $`P(X_i = 1) = P(X_i = -1) = 0.5`$ (a fair coin-flipping game). The strategy: start with a \$1 bet, and keep doubling until you win, then stop.

The size of the bet on the $`n`$-th play is $`2^{n-1}`$ (since \$1, \$2, \$4, \$8, etc.). Let $`W_n`$ denote total winnings after $`n`$ coin tosses (with $`W_0 = 0`$).

**$`W_n`$ can only take two values:** Either $`W_n = 1`$ (if we have already won at some point) or $`W_n = -(2^n - 1)`$ (if we have lost every toss so far).

**Why?** If we win for the first time on the $`n`$-th bet, our total winnings are:

```math
W_n = -\left(1 + 2 + 4 + \cdots + 2^{n-2}\right) + 2^{n-1} = -(2^{n-1} - 1) + 2^{n-1} = 1
```

The losses form a geometric series summing to $`2^{n-1} - 1`$, and the final win of $`2^{n-1}`$ exactly exceeds this by 1. If we have not yet won after $`n`$ bets, $`W_n = -(2^n - 1)`$.

**$`W_n`$ is a martingale.** To show $`E[W_{n+1} \mid W_n] = W_n`$, consider two cases:

**Case 1:** $`W_n = 1`$ (already won). The game has stopped, so $`W_{n+1} = 1`$ with certainty, and $`E[W_{n+1} \mid W_n = 1] = 1 = W_n`$.

**Case 2:** $`W_n = -(2^n - 1)`$ (haven't won yet). We bet $`2^n`$ on the next toss. With probability $`\tfrac{1}{2}`$ we win, giving $`W_{n+1} = 1`$, and with probability $`\tfrac{1}{2}`$ we lose, giving $`W_{n+1} = -(2^{n+1} - 1)`$. Then:

```math
E[W_{n+1} \mid W_n] = \frac{1}{2}(1) + \frac{1}{2}(-(2^{n+1} - 1)) = \frac{1}{2} - \frac{2^{n+1} - 1}{2} = -(2^n - 1) = W_n
```

In both cases the martingale property holds. This example is important because it generalises: any betting strategy where the bet size depends only on past information preserves the martingale property — a result deeply connected to trading strategies in finance.

### 5.4 Pólya's Urn

Consider an urn initially containing one red ball and one green ball (two balls total). At each step, draw a ball at random, then return it along with one additional ball of the same colour.

After $`n`$ draws, the urn contains $`n + 2`$ balls. Let $`X_n`$ denote the number of red balls after $`n`$ draws. If $`X_n = k`$, then:

```math
X_{n+1} = \begin{cases} k + 1 & \text{with probability } \dfrac{k}{n+2} \quad \text{(drew red)} \\[6pt] k & \text{with probability } \dfrac{n + 2 - k}{n + 2} \quad \text{(drew green)} \end{cases}
```

**Claim:** $`M_n = \tfrac{X_n}{n + 2}`$ (the *proportion* of red balls) is a martingale. This can be verified by computing $`E[M_{n+1} \mid X_n = k]`$ and showing it equals $`\tfrac{k}{n+2} = M_n`$.

---

## 6. Brownian Motion

### 6.1 Definition and Properties

A **Brownian motion** (also called a Wiener process) with parameters $`\mu`$ (drift) and $`\sigma`$ (volatility) is a continuous-time stochastic process $`\{X_t : t \geq 0\}`$ satisfying:

1. **Independent increments:** For fixed times $`t_1 < t_2 < \cdots < t_n`$, the increments $`X_{t_2} - X_{t_1}, \; X_{t_3} - X_{t_2}, \; \ldots, \; X_{t_n} - X_{t_{n-1}}`$ are mutually independent.
2. **Normally distributed increments:** For any $`s > 0`$, the increment $`X_{t+s} - X_t \sim N(\mu s, \; \sigma^2 s)`$.
3. **Continuous paths:** $`X_t`$ is a continuous function of $`t`$ (the path never jumps).

We write $`X_t \sim BM(\mu, \sigma)`$. The parameter $`\mu`$ is the **drift** (average rate of change per unit time) and $`\sigma`$ is the **volatility** (controlling the magnitude of random fluctuations).

Brownian motion was introduced in a financial context by **Bachelier** (1900), who used it to model stock prices on the Paris exchange, and independently by **Einstein** (1905) in the context of diffusion processes. **Wiener** (1920s) rigorously established its existence as a well-defined mathematical object.

### 6.2 Standard Brownian Motion

When $`\mu = 0`$ and $`\sigma = 1`$, we have a **standard Brownian motion**, denoted $`W_t`$, with $`W_0 = 0`$.

If $`X_t`$ is a $`BM(\mu, \sigma)`$ starting at $`X_0 = x`$, then it can be written as

```math
X_t = x + \mu t + \sigma W_t
```

From this representation:

```math
E[X_t] = x + \mu t, \qquad \text{Var}(X_t) = \sigma^2 t
```

The variance grows **linearly** with time — this is a hallmark of diffusion processes. Over long horizons, uncertainty about the process's value grows without bound.

### 6.3 Information Filtrations and the Independent Increments Property

Let $`\mathcal{F}_t`$ denote the information available at time $`t`$ (i.e. the entire history of the process up to time $`t`$). The **independent increments property** implies a crucial fact:

> Any function of the increment $`W_{t+s} - W_t`$ is independent of $`\mathcal{F}_t`$.

In other words, knowing the entire past history of the Brownian motion provides **no information** about future increments. Even conditioning on $`\mathcal{F}_t`$, the increment $`W_{t+s} - W_t`$ remains $`N(0, s)`$.

### 6.4 A Calculation with Brownian Motion

As an illustration, let us compute $`E[W_{t+s} \cdot W_s]`$ for a standard Brownian motion:

```math
E[W_{t+s} \cdot W_s] = E\left[(W_{t+s} - W_s + W_s) \cdot W_s\right]
```

```math
= E\left[(W_{t+s} - W_s) \cdot W_s\right] + E[W_s^2]
```

For the first term, we use the conditional expectation identity: condition on $`\mathcal{F}_s`$ so that $`W_s`$ becomes a known constant and $`(W_{t+s} - W_s)`$ is independent of $`\mathcal{F}_s`$ with mean 0. Therefore the first term equals zero.

For the second term, since $`E[W_s] = 0`$:

```math
E[W_s^2] = \text{Var}(W_s) + (E[W_s])^2 = s + 0 = s
```

Therefore:

```math
E[W_{t+s} \cdot W_s] = s
```

This result extends to the general formula $`E[W_t \cdot W_s] = \min(t, s)`$ for standard Brownian motion.

---

## 7. Geometric Brownian Motion

### 7.1 Definition

A stochastic process $`X_t`$ is a **Geometric Brownian Motion (GBM)** with parameters $`\mu`$ (drift) and $`\sigma`$ (volatility) if

```math
X_t = X_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]
```

where $`W_t`$ is a standard Brownian motion. We write $`X_t \sim GBM(\mu, \sigma)`$.

Since $`W_t \sim N(0, t)`$, the exponent $`\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma W_t`$ is normally distributed, so $`X_t`$ has a **log-normal distribution**. The $`-\tfrac{\sigma^2}{2}`$ term is an Itô correction that ensures the expected growth rate is exactly $`\mu`$ (as we shall verify below).

### 7.2 Simulation and the Recursive Representation

A key property of GBM is its recursive (multiplicative) structure. By substituting $`t + s`$ for $`t`$ and rearranging:

```math
X_{t+s} = X_t \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)s + \sigma(W_{t+s} - W_t)\right]
```

The increment $`W_{t+s} - W_t \sim N(0, s)`$ is independent of $`X_t`$ (by the independent increments property of Brownian motion). This representation is extremely useful for **simulation**: to generate a sample path of GBM at times $`0, \Delta, 2\Delta, 3\Delta, \ldots`$, we simply iterate:

```math
X_{(k+1)\Delta} = X_{k\Delta} \cdot \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta + \sigma \sqrt{\Delta} \; Z_{k+1}\right]
```

where $`Z_1, Z_2, \ldots`$ are i.i.d. standard normal random variables. This can be implemented straightforwardly in Python, MATLAB, or even Excel.

### 7.3 Expected Value of GBM

From the recursive representation, conditioned on time-$`t`$ information:

```math
E[X_{t+s} \mid \mathcal{F}_t] = X_t \cdot E\left[\exp\left(\left(\mu - \frac{\sigma^2}{2}\right)s + \sigma(W_{t+s} - W_t)\right)\right]
```

Since $`W_{t+s} - W_t \sim N(0, s)`$ and is independent of $`\mathcal{F}_t`$, we compute the expectation using the MGF of the normal distribution. If $`Z \sim N(a, b^2)`$, then $`E[e^{sZ}] = e^{as + b^2 s^2 / 2}`$.

Here the exponent contains $`\sigma \cdot (W_{t+s} - W_t) \sim N(0, \sigma^2 s)`$, so:

```math
E\left[e^{\sigma(W_{t+s} - W_t)}\right] = e^{\sigma^2 s / 2}
```

Multiplying by $`e^{(\mu - \sigma^2/2)s}`$:

```math
E[X_{t+s} \mid \mathcal{F}_t] = X_t \cdot e^{(\mu - \sigma^2/2)s} \cdot e^{\sigma^2 s/2} = X_t \cdot e^{\mu s}
```

Therefore, the **expected growth rate of GBM is** $`\mu`$, confirming the role of the drift parameter.

### 7.4 Key Properties and Relevance to Stock Prices

**Positive values.** If $`X_t > 0`$, then $`X_{t+s} > 0`$ for all $`s > 0`$ (since the exponential is always positive). This respects the **limited liability** of stock prices — they cannot go negative.

**Returns are independent of price level.** The distribution of $`X_{t+s}/X_t`$ depends only on $`s`$, $`\mu`$, and $`\sigma`$ — not on $`X_t`$. This is a desirable property: we generally do not expect stock returns to depend on the current price.

**Log-returns are normally distributed:**

```math
\ln\left(\frac{X_{t+s}}{X_t}\right) \sim N\left(\left(\mu - \frac{\sigma^2}{2}\right)s, \; \sigma^2 s\right)
```

**Independent return ratios.** The ratios $`X_{t_2}/X_{t_1}, \; X_{t_3}/X_{t_2}, \ldots`$ are mutually independent (following from the independent increments of Brownian motion).

**Continuous paths.** $`X_t`$ is continuous in $`t`$ (it never jumps), inherited from the continuity of Brownian motion.

These properties make GBM a reasonable (if imperfect) model for stock prices, and it is the underlying process in the **Black–Scholes** option pricing formula.

---

## 8. Vectors

### 8.1 Definitions: Row and Column Vectors

A **vector** is an ordered collection of $`n`$ real numbers. It can be arranged as a **row vector** $`(v_1, v_2, \ldots, v_n)`$ or a **column vector**. We write $`\mathbf{v} \in \mathbb{R}^n`$ to indicate a vector with $`n`$ real-valued components.

**Convention:** In this course, all vectors are **column vectors** by default unless stated otherwise.

### 8.2 Linear Combinations, Dependence, and Independence

Given vectors $`\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k`$ and scalars $`\alpha_1, \alpha_2, \ldots, \alpha_k`$, the vector

```math
\mathbf{w} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k
```

is called a **linear combination** of the $`\mathbf{v}_i`$'s. The set of all such linear combinations is the **linear span** of $`\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}`$.

**Linear independence.** The vectors $`\mathbf{v}_1, \ldots, \mathbf{v}_k`$ are **linearly independent** if no vector in the set can be written as a linear combination of the others. Equivalently, the only solution to $`\alpha_1 \mathbf{v}_1 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}`$ is $`\alpha_1 = \cdots = \alpha_k = 0`$.

**Example in** $`\mathbb{R}^2`$**.** Two vectors $`\mathbf{v}`$ and $`\mathbf{w}`$ that point in different directions are linearly independent — neither lies on the line spanned by the other. Any third vector $`\mathbf{x} \in \mathbb{R}^2`$ can be written as a linear combination of $`\mathbf{v}`$ and $`\mathbf{w}`$, so it is linearly dependent on them.

### 8.3 Basis and the Standard Basis

A **basis** for $`\mathbb{R}^n`$ is a set of $`n`$ linearly independent vectors that spans the entire space. Every vector in $`\mathbb{R}^n`$ can be uniquely expressed as a linear combination of the basis vectors.

The **standard basis** for $`\mathbb{R}^n`$ consists of vectors $`\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n`$ where $`\mathbf{e}_i`$ has a 1 in the $`i`$-th position and 0 elsewhere. Any vector $`\mathbf{w} = (w_1, \ldots, w_n)^\top`$ decomposes as

```math
\mathbf{w} = w_1 \mathbf{e}_1 + w_2 \mathbf{e}_2 + \cdots + w_n \mathbf{e}_n
```

The standard basis is particularly convenient because the coefficients in this decomposition are simply the components of $`\mathbf{w}`$ itself.

### 8.4 Norms (Length of a Vector)

The **$`\ell^2`$ norm** (Euclidean norm) of a vector $`\mathbf{v} \in \mathbb{R}^n`$ is

```math
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}
```

A norm is any function $`\|\cdot\|`$ satisfying three properties: non-negativity ($`\|\mathbf{v}\| \geq 0`$ with equality iff $`\mathbf{v} = \mathbf{0}`$), absolute homogeneity ($`\|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\|`$), and the **triangle inequality** ($`\|\mathbf{v} + \mathbf{w}\| \leq \|\mathbf{v}\| + \|\mathbf{w}\|`$).

Another important norm is the **$`\ell^1`$ norm** (Manhattan distance):

```math
\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|
```

In $`\mathbb{R}^2`$, for a vector $`(4, 3)`$: the $`\ell^2`$ distance is $`\sqrt{16 + 9} = 5`$ (the "as the crow flies" distance), while the $`\ell^1`$ distance is $`4 + 3 = 7`$ (walking along a grid, as in Manhattan's city blocks).

### 8.5 Inner Products and Angles

The **inner product** (or dot product) of $`\mathbf{v}, \mathbf{w} \in \mathbb{R}^n`$ is

```math
\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} v_i w_i = \mathbf{v}^\top \mathbf{w}
```

The inner product is related to the angle $`\theta`$ between the two vectors by

```math
\cos\theta = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\|_2 \, \|\mathbf{w}\|_2}
```

When $`\mathbf{v} \cdot \mathbf{w} = 0`$, the vectors are **orthogonal** (perpendicular), corresponding to $`\theta = 90°`$. The $`\ell^2`$ norm can be expressed in terms of the inner product: $`\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^\top \mathbf{v}}`$.

---

## 9. Matrices

### 9.1 Definitions and Notation

A **matrix** is a rectangular array of real numbers with $`m`$ rows and $`n`$ columns, written as $`A \in \mathbb{R}^{m \times n}`$. The element in the $`i`$-th row and $`j`$-th column is denoted $`A_{ij}`$, where the first subscript always refers to the row.

**Special cases:** A row vector is a $`1 \times n`$ matrix. A column vector is an $`m \times 1`$ matrix.

The **identity matrix** $`I_n`$ is the $`n \times n`$ matrix with ones on the diagonal and zeros elsewhere. It satisfies $`AI = IA = A`$ for any conformably sized matrix $`A`$ — it is the matrix analogue of the number 1.

### 9.2 Transpose

The **transpose** of a matrix $`A \in \mathbb{R}^{m \times n}`$ is the matrix $`A^\top \in \mathbb{R}^{n \times m}`$ obtained by swapping rows and columns: $`(A^\top)_{ij} = A_{ji}`$.

For a column vector $`\mathbf{v}`$, the transpose $`\mathbf{v}^\top`$ is a row vector. The inner product $`\mathbf{v} \cdot \mathbf{w}`$ can then be written as the matrix product $`\mathbf{v}^\top \mathbf{w}`$.

### 9.3 Matrix Multiplication

For $`A \in \mathbb{R}^{m \times d}`$ and $`B \in \mathbb{R}^{d \times p}`$, the product $`C = AB`$ is a matrix in $`\mathbb{R}^{m \times p}`$ with elements

```math
C_{ij} = \sum_{\ell=1}^{d} A_{i\ell} \, B_{\ell j}
```

The rule is: take the $`i`$-th **row** of $`A`$ and the $`j`$-th **column** of $`B`$, multiply them component-wise, and sum. The **inner dimensions** ($`d`$) must match; the result has the **outer dimensions** ($`m \times p`$).

**Example.** For $`A \in \mathbb{R}^{2 \times 3}`$ and $`\mathbf{b} \in \mathbb{R}^{3 \times 1}`$:

```math
\begin{pmatrix} 2 & 3 & 7 \\ 1 & 6 & 5 \end{pmatrix} \begin{pmatrix} 2 \\ 6 \\ 4 \end{pmatrix} = \begin{pmatrix} 2(2) + 3(6) + 7(4) \\ 1(2) + 6(6) + 5(4) \end{pmatrix} = \begin{pmatrix} 50 \\ 58 \end{pmatrix}
```

The $`\ell^2`$ norm can now be written compactly: $`\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^\top \mathbf{v}}`$.

### 9.4 Linear Functions and Constraints

A function $`f : \mathbb{R}^n \to \mathbb{R}^m`$ is **linear** if

```math
f(\alpha \mathbf{x} + \beta \mathbf{y}) = \alpha f(\mathbf{x}) + \beta f(\mathbf{y}) \quad \text{for all } \mathbf{x}, \mathbf{y} \in \mathbb{R}^n \text{ and all } \alpha, \beta \in \mathbb{R}
```

A fundamental result in linear algebra states that $`f`$ is linear **if and only if** there exists a matrix $`A`$ such that $`f(\mathbf{x}) = A\mathbf{x}`$. Every linear function is just a matrix multiplication.

**Linear constraints** come in two forms: **equalities** ($`A\mathbf{x} = \mathbf{b}`$) and **inequalities** ($`A\mathbf{x} \leq \mathbf{b}`$, interpreted component-wise). These constraints define geometric regions (hyperplanes, half-spaces, polyhedra) that are central to optimisation problems in finance.

### 9.5 Rank and Range

The **column rank** of a matrix $`A`$ is the maximum number of linearly independent columns. The **row rank** is the maximum number of linearly independent rows. A theorem guarantees these are always equal, so we simply speak of the **rank** of $`A`$.

The **range** (or column space) of $`A \in \mathbb{R}^{m \times d}`$ is the set of all vectors $`\mathbf{y} \in \mathbb{R}^m`$ that can be written as $`\mathbf{y} = A\boldsymbol{\theta}`$ for some $`\boldsymbol{\theta} \in \mathbb{R}^d`$:

```math
\text{Range}(A) = \{A\boldsymbol{\theta} : \boldsymbol{\theta} \in \mathbb{R}^d\}
```

The rank tells us the **dimension** of the range. A rank-1 matrix maps all of $`\mathbb{R}^d`$ onto a one-dimensional line; a rank-$`k`$ matrix maps onto a $`k`$-dimensional subspace. Higher rank means the matrix can produce a richer variety of outputs — a concept that will be critical when discussing complete versus incomplete markets.

### 9.6 Inverse of a Matrix

A square matrix $`A \in \mathbb{R}^{n \times n}`$ with rank $`n`$ (full rank) is **invertible**: there exists a unique matrix $`A^{-1}`$ such that

```math
A^{-1}A = AA^{-1} = I_n
```

This is the matrix analogue of division: just as $`\tfrac{1}{a} \cdot a = 1`$ for $`a \neq 0`$, we have $`A^{-1}A = I`$ for invertible $`A`$. A matrix is invertible if and only if all its columns (and rows) are linearly independent.

---

## 10. Linear Optimisation and Hedging

### 10.1 The Hedging Problem Setup

Consider a market with $`d`$ assets. At time $`t = 0`$, the price vector is $`\mathbf{p} \in \mathbb{R}^d`$. At time $`t = 1`$, the market can be in one of $`m`$ possible states. The payoff matrix $`S \in \mathbb{R}^{m \times d}`$ has entry $`S_{ij}`$ representing the price of asset $`j`$ in state $`i`$. Each **row** of $`S`$ gives the prices of all assets in a particular state; each **column** gives the prices of a particular asset across all states.

We have an **obligation** $`\mathbf{x} \in \mathbb{R}^m`$ — a vector of payments we owe in each state. We choose a **portfolio** $`\boldsymbol{\theta} \in \mathbb{R}^d`$ (where $`\theta_j`$ is the number of shares of asset $`j`$; negative values indicate short-selling).

**Cost at time 0:** $`\mathbf{p}^\top \boldsymbol{\theta} = \sum_{j=1}^{d} p_j \theta_j`$.

**Payoff at time 1 in state** $`i`$: $`y_i = \sum_{j=1}^{d} S_{ij} \theta_j`$, or in vector form: $`\mathbf{y} = S\boldsymbol{\theta}`$.

### 10.2 Portfolio Payoffs and the Role of Rank

The payoff vector $`\mathbf{y} = S\boldsymbol{\theta}`$ can be viewed as a **linear combination of the columns** of $`S`$:

```math
\mathbf{y} = \theta_1 \mathbf{S}_1 + \theta_2 \mathbf{S}_2 + \cdots + \theta_d \mathbf{S}_d
```

Thus, the set of achievable payoffs is exactly the **range of** $`S`$. If $`\text{rank}(S) = m`$, then every payoff vector in $`\mathbb{R}^m`$ is achievable — this is a **complete market**. If $`\text{rank}(S) < m`$, some payoffs cannot be replicated — an **incomplete market**.

We say the portfolio $`\boldsymbol{\theta}`$ **hedges** the obligation $`\mathbf{x}`$ if $`\mathbf{y} \geq \mathbf{x}`$ component-wise (the payoff in every state covers the obligation).

The hedging optimisation problem is:

```math
\min_{\boldsymbol{\theta}} \; \mathbf{p}^\top \boldsymbol{\theta} \quad \text{subject to} \quad S\boldsymbol{\theta} \geq \mathbf{x}
```

This is a **linear program** — both the objective and constraints are linear in the decision variable $`\boldsymbol{\theta}`$.

### 10.3 Linear Programming and Duality

A **linear program (LP)** has a linear objective function and linear equality/inequality constraints. A fundamental result is that every LP has a **dual** LP, and the two are intimately connected.

**Primal:**

```math
\min_{\mathbf{x}} \; \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \geq \mathbf{b}
```

**Dual:**

```math
\max_{\mathbf{u}} \; \mathbf{b}^\top \mathbf{u} \quad \text{s.t.} \quad A^\top \mathbf{u} = \mathbf{c}, \; \mathbf{u} \geq \mathbf{0}
```

**Weak duality** states that the optimal primal value $`P`$ is always at least as large as the optimal dual value $`D`$: that is, $`P \geq D`$. This gives a useful chain of inequalities: for any primal-feasible $`\mathbf{x}`$ and dual-feasible $`\mathbf{u}`$:

```math
\mathbf{c}^\top \mathbf{x} \geq P \geq D \geq \mathbf{b}^\top \mathbf{u}
```

If we can find $`\mathbf{x}`$ and $`\mathbf{u}`$ with $`\mathbf{c}^\top \mathbf{x} \approx \mathbf{b}^\top \mathbf{u}`$, then both must be near-optimal.

**Strong duality:** When either $`P`$ or $`D`$ is finite, we have $`P = D`$ (the duality gap is zero). Additionally, taking the dual of the dual recovers the primal.

For the equality-constrained variant — $`\min \mathbf{c}^\top \mathbf{x}`$ s.t. $`A\mathbf{x} = \mathbf{b}`$ — the dual is $`\max \mathbf{b}^\top \mathbf{u}`$ s.t. $`A^\top \mathbf{u} = \mathbf{c}`$ (with $`\mathbf{u}`$ unrestricted in sign).

### 10.4 Lagrangian Relaxation (Linear Case)

The dual LP is derived via **Lagrangian relaxation**:

1. **Dualise:** Multiply the constraint $`A\mathbf{x} - \mathbf{b} \geq \mathbf{0}`$ by a non-negative multiplier $`\mathbf{u} \geq \mathbf{0}`$.
2. **Add to the objective:** Form $`\mathbf{c}^\top \mathbf{x} - \mathbf{u}^\top(A\mathbf{x} - \mathbf{b}) \leq \mathbf{c}^\top \mathbf{x}`$ (since we subtract a non-negative quantity).
3. **Relax:** Drop the constraint (enlarging the feasible set, which can only lower the minimum).
4. **Minimise over** $`\mathbf{x}`$: The unconstrained minimum of $`(\mathbf{c} - A^\top \mathbf{u})^\top \mathbf{x}`$ is $`-\infty`$ unless $`\mathbf{c} = A^\top \mathbf{u}`$, in which case the minimum is 0 and the bound becomes $`P \geq \mathbf{b}^\top \mathbf{u}`$.
5. **Maximise over** $`\mathbf{u}`$: We get the dual — $`\max \mathbf{b}^\top \mathbf{u}`$ subject to $`A^\top \mathbf{u} = \mathbf{c}`$, $`\mathbf{u} \geq \mathbf{0}`$.

---

## 11. Non-Linear Optimisation

### 11.1 Unconstrained Optimisation: Gradient and Hessian

Consider minimising $`f(\mathbf{x})`$ where $`\mathbf{x} \in \mathbb{R}^n`$ and $`f`$ is a smooth (twice-differentiable) function.

A **local minimum** is a point that is optimal within some neighbourhood (unlike a **global minimum**, which is optimal everywhere). Since derivatives can only "see" locally, derivative-based methods find local minima.

**First-order necessary condition (gradient).** At a local minimum $`\mathbf{x}^*`$, the gradient must vanish:

```math
\nabla f(\mathbf{x}^*) = \begin{pmatrix} \dfrac{\partial f}{\partial x_1} \\[6pt] \vdots \\[6pt] \dfrac{\partial f}{\partial x_n} \end{pmatrix}\bigg|_{\mathbf{x}^*} = \mathbf{0}
```

**Second-order condition (Hessian).** The Hessian matrix of second partial derivatives is

```math
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
```

At a local minimum, $`H`$ must be **positive semi-definite** (all eigenvalues $`\geq 0`$). If $`H`$ is **positive definite** (all eigenvalues $`> 0`$), the point is definitively a local minimum.

**Example.** Minimise $`f(x_1, x_2) = x_1^2 + 3x_1 x_2 + x_2^3`$.

Setting $`\nabla f = \mathbf{0}`$ gives $`2x_1 + 3x_2 = 0`$ and $`3x_1 + 3x_2^2 = 0`$. Solving yields two critical points: $`(0, 0)`$ and $`(-\tfrac{9}{4}, \tfrac{3}{2})`$.

The Hessian is:

```math
H = \begin{pmatrix} 2 & 3 \\ 3 & 6x_2 \end{pmatrix}
```

At $`(0, 0)`$: $`H = \begin{pmatrix} 2 & 3 \\ 3 & 0 \end{pmatrix}`$, which has a negative eigenvalue — **not a local minimum** (it is a saddle point).

At $`(-\tfrac{9}{4}, \tfrac{3}{2})`$: $`H = \begin{pmatrix} 2 & 3 \\ 3 & 9 \end{pmatrix}`$, which is positive definite — **confirmed local minimum**.

### 11.2 Convex Functions

A function $`f`$ is **convex** if for any two points $`\mathbf{x}, \mathbf{y}`$ and any $`\lambda \in [0, 1]`$:

```math
f(\lambda \mathbf{x} + (1 - \lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1 - \lambda) f(\mathbf{y})
```

Geometrically, the line segment connecting any two points on the graph lies **above** the graph. For convex functions, **every local minimum is a global minimum** — there are no misleading local optima. To find the global minimum, simply set $`\nabla f = \mathbf{0}`$ and solve (no need to check the Hessian).

### 11.3 Constrained Optimisation and Lagrange Multipliers

When constraints are present, we use **Lagrange multipliers** to convert a constrained problem into an unconstrained one.

**Example — Utility Maximisation.** Maximise $`2\ln(1 + x_1) + 4\ln(1 + x_2)`$ subject to $`x_1 + x_2 = 12`$.

Form the **Lagrangian** by multiplying the constraint by a multiplier $`\nu`$ and subtracting from the objective:

```math
\mathcal{L}(x_1, x_2, \nu) = 2\ln(1 + x_1) + 4\ln(1 + x_2) - \nu(x_1 + x_2 - 12)
```

Setting partial derivatives to zero:

```math
\frac{\partial \mathcal{L}}{\partial x_1} = \frac{2}{1 + x_1} - \nu = 0 \implies x_1 = \frac{2}{\nu} - 1
```

```math
\frac{\partial \mathcal{L}}{\partial x_2} = \frac{4}{1 + x_2} - \nu = 0 \implies x_2 = \frac{4}{\nu} - 1
```

Substituting into the constraint $`x_1 + x_2 = 12`$: this gives $`\tfrac{6}{\nu} - 2 = 12`$, so $`\nu = \tfrac{3}{7}`$.

Back-substituting: $`x_1 = \tfrac{11}{3}`$ and $`x_2 = \tfrac{25}{3}`$. The solution invests more in the second opportunity (higher return coefficient) but diminishing returns from the logarithm prevent investing everything there.

### 11.4 Application: Portfolio Selection (Mean–Variance)

The classic **Markowitz mean–variance portfolio selection** problem:

```math
\max_{\mathbf{x}} \; \boldsymbol{\mu}^\top \mathbf{x} - \lambda \, \mathbf{x}^\top V \mathbf{x} \quad \text{subject to} \quad \mathbf{1}^\top \mathbf{x} = 1
```

where $`\mathbf{x} \in \mathbb{R}^n`$ is the portfolio weight vector, $`\boldsymbol{\mu}`$ is the vector of expected returns, $`V`$ is the covariance matrix of returns, $`\lambda > 0`$ is the risk aversion parameter, and $`\mathbf{1}^\top \mathbf{x} = 1`$ ensures the weights sum to 1 (fully invested portfolio).

**Lagrangian approach.** Introduce multiplier $`\nu`$ for the constraint:

```math
\mathcal{L} = \boldsymbol{\mu}^\top \mathbf{x} - \lambda \, \mathbf{x}^\top V \mathbf{x} - \nu(\mathbf{1}^\top \mathbf{x} - 1)
```

Taking the gradient with respect to $`\mathbf{x}`$ and setting it to zero:

```math
\nabla_{\mathbf{x}} \mathcal{L} = \boldsymbol{\mu} - 2\lambda V \mathbf{x} - \nu \mathbf{1} = \mathbf{0}
```

Solving for $`\mathbf{x}`$:

```math
\mathbf{x}^* = \frac{1}{2\lambda} V^{-1}(\boldsymbol{\mu} - \nu \mathbf{1})
```

To find $`\nu`$, apply the constraint $`\mathbf{1}^\top \mathbf{x}^* = 1`$:

```math
\mathbf{1}^\top V^{-1}(\boldsymbol{\mu} - \nu \mathbf{1}) = 2\lambda
```

```math
\nu = \frac{\mathbf{1}^\top V^{-1} \boldsymbol{\mu} - 2\lambda}{\mathbf{1}^\top V^{-1} \mathbf{1}}
```

Substituting $`\nu`$ back gives the closed-form optimal portfolio weights. By varying $`\lambda`$ (the risk aversion parameter), we trace out the **efficient frontier** — the set of portfolios offering the maximum expected return for each level of risk. This analysis leads to the result that any efficient portfolio can be expressed as a combination of a small number of **mutual funds**, and ultimately to the **Capital Asset Pricing Model (CAPM)**.

---

## Summary of Key Relationships

| Concept | Key Formula | Application in Finance |
|---|---|---|
| Binomial Distribution | $`P(X=r) = \binom{n}{r}p^r(1-p)^{n-r}`$ | Fund manager performance evaluation |
| Poisson Distribution | $`P(X=r) = \frac{\lambda^r e^{-\lambda}}{r!}`$ | Modelling default counts |
| Bayes' Theorem | $`P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}`$ | Updating beliefs with new information |
| Conditional Expectation | $`E[X] = E[E[X \mid Y]]`$ | Pricing derivatives, computing expected defaults |
| Conditional Variance | $`\text{Var}(X) = \text{Var}(E[X \mid Y]) + E[\text{Var}(X \mid Y)]`$ | Decomposing portfolio risk |
| Martingale Property | $`E[X_{n+m} \mid \mathcal{F}_n] = X_n`$ | Fair pricing, trading strategies |
| Brownian Motion | $`W_{t+s} - W_t \sim N(0, s)`$ | Foundation of continuous-time finance |
| Geometric Brownian Motion | $`X_t = X_0 e^{(\mu-\sigma^2/2)t + \sigma W_t}`$ | Black–Scholes stock price model |
| Linear Duality | $`P = D`$ (strong duality) | Hedging, risk management |
| Mean–Variance Portfolio | $`\mathbf{x}^* = \frac{1}{2\lambda}V^{-1}(\boldsymbol{\mu} - \nu\mathbf{1})`$ | Efficient frontier, CAPM |

---

*These notes are based on the prerequisite modules of Columbia University's Financial Engineering & Risk Management course. They provide the mathematical foundations for topics covered later in the course, including derivatives pricing, credit risk modelling, and portfolio optimisation.*
