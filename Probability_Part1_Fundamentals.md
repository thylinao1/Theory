# Part I: The Fundamentals of Probability

> **Comprehensive Course Notes**
> Based on MIT 6.041 *Introduction to Probability* (Tsitsiklis & Jaillet) and Dennis Sun's *Introduction to Probability*
> Part 1 of 3: The Fundamentals | Parts 2 & 3: Inference, Limit Theorems & Random Processes

---

## Table of Contents

1. [Probability Models and Axioms](#1-probability-models-and-axioms)
2. [Conditioning and Bayes' Rule](#2-conditioning-and-bayes-rule)
3. [Independence](#3-independence)
4. [Counting](#4-counting)
5. [Discrete Random Variables](#5-discrete-random-variables)
6. [Variance, Conditioning on an Event, and Multiple Random Variables](#6-variance-conditioning-on-an-event-and-multiple-random-variables)
7. [Conditioning on a Random Variable and Independence of Random Variables](#7-conditioning-on-a-random-variable-and-independence-of-random-variables)
8. [Continuous Random Variables and Probability Density Functions](#8-continuous-random-variables-and-probability-density-functions)
9. [Conditioning on an Event and Multiple Continuous Random Variables](#9-conditioning-on-an-event-and-multiple-continuous-random-variables)
10. [Conditioning on a Random Variable, Independence, and Bayes' Rule for Continuous Variables](#10-conditioning-on-a-random-variable-independence-and-bayes-rule-for-continuous-variables)
11. [Derived Distributions](#11-derived-distributions)
12. [Sums of Independent Random Variables and Covariance](#12-sums-of-independent-random-variables-and-covariance)
13. [Conditional Expectation and Variance Revisited](#13-conditional-expectation-and-variance-revisited)

---

## 1. Probability Models and Axioms

### 1.1 What Is a Probability Model?

A probability model is a mathematical description of an uncertain situation. Every probability model consists of two ingredients: a **sample space** $`\Omega`$ (the set of all possible outcomes of an experiment) and a **probability law** $`P`$ (a function that assigns probabilities to events, i.e., subsets of $`\Omega`$). The purpose of building such a model is to reason systematically about uncertainty, whether you are designing a communication system, pricing an insurance policy, or training a machine learning algorithm.

### 1.2 Sample Space

The sample space $`\Omega`$ is the collection of every conceivable outcome. It must be **mutually exclusive** (no two outcomes can occur simultaneously) and **collectively exhaustive** (one outcome must occur).

**Discrete/finite example.** Two rolls of a four-sided die: $`\Omega = \lbrace (i,j) : i,j \in \lbrace 1,2,3,4 \rbrace \rbrace`$, giving 16 equally likely outcomes.

**Continuous example.** Throwing a dart at a unit square: $`\Omega = \lbrace (x,y) : 0 \le x,y \le 1 \rbrace`$, an uncountable set.

**Discrete but infinite example.** Tossing a coin until the first head: $`\Omega = \lbrace 1, 2, 3, \ldots \rbrace`$, where outcome $`k`$ means the first head appeared on toss $`k`$.

### 1.3 Probability Axioms

Given a sample space $`\Omega`$, a probability law $`P`$ must satisfy three axioms.

**Axiom 1 (Nonnegativity).** For every event $`A`$:

$$P(A) \ge 0$$

**Axiom 2 (Normalization).** The probability of the entire sample space is one:

$$P(\Omega) = 1$$

**Axiom 3 (Additivity).** If $`A`$ and $`B`$ are disjoint events ($`A \cap B = \emptyset`$), then:

$$P(A \cup B) = P(A) + P(B)$$

For continuous or countably infinite sample spaces, we strengthen this to **countable additivity**: if $`A_1, A_2, \ldots`$ are pairwise disjoint, then $`P(A_1 \cup A_2 \cup \cdots) = \sum_{i=1}^{\infty} P(A_i)`$.

### 1.4 Consequences of the Axioms

From just these three axioms, many useful properties follow.

The probability of the complement of $`A`$ satisfies $`P(A^c) = 1 - P(A)`$, because $`A`$ and $`A^c`$ partition $`\Omega`$.

For any two events (not necessarily disjoint), the **inclusion-exclusion principle** gives:

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

We can extend this reasoning. If $`A \subset B`$, then $`P(A) \le P(B)`$. Also, the **union bound** states that for any events $`A_1, \ldots, A_n`$:

$$P(A_1 \cup \cdots \cup A_n) \le P(A_1) + \cdots + P(A_n)$$

This bound is extremely useful in practice when computing the exact union probability is hard but individual probabilities are easy.

### 1.5 Discrete Uniform Law

When the sample space is finite with $`n`$ equally likely outcomes, the probability of any event $`A`$ is simply:

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{number of outcomes in } A}{n}$$

This is the foundation for all counting-based probability problems.

### 1.6 Probability Calculations with Continuous Models

When $`\Omega`$ is a subset of $`\mathbb{R}^n`$ and outcomes are "equally likely," probability is computed as a ratio of areas (or volumes):

$$P(A) = \frac{\text{area of } A}{\text{area of } \Omega}$$

For example, if the sample space is the unit square, the probability that $`x + y \le 1/2`$ equals the area of the triangle below the line $`x + y = 1/2`$, which is $`1/8`$.

### 1.7 Interpretations of Probability

There are two major ways to interpret probability. The **frequentist** interpretation defines probability as the long-run relative frequency of an event: if you repeat the experiment many times, the fraction of times $`A`$ occurs converges to $`P(A)`$. The **Bayesian (subjective)** interpretation treats probability as a degree of belief about an uncertain event, even if the experiment cannot be repeated. Both interpretations use the same mathematical axioms.

---

## 2. Conditioning and Bayes' Rule

### 2.1 Conditional Probability

Conditional probability is the mechanism for incorporating new information into a probability model. If we learn that event $`B`$ has occurred, we should revise the probability of any event $`A`$ accordingly.

**Definition.** Given $`P(B) > 0`$:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

The intuition is straightforward: once we know $`B`$ happened, the new "universe" of possible outcomes shrinks from $`\Omega`$ to $`B`$. The probability of $`A`$ is now the fraction of $`B`$ that also belongs to $`A`$.

**Conditional probabilities behave like ordinary probabilities.** They satisfy all three axioms. In particular: $`P(A \mid B) \ge 0`$, $`P(B \mid B) = 1`$, and if $`A \cap C = \emptyset`$ then $`P(A \cup C \mid B) = P(A \mid B) + P(C \mid B)`$.

### 2.2 The Multiplication Rule

The definition of conditional probability can be rearranged to give the **multiplication rule**, which is useful for computing the probability of an intersection:

$$P(A \cap B) = P(B) \cdot P(A \mid B) = P(A) \cdot P(B \mid A)$$

This extends to three or more events:

$$P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)$$

Use this when a problem naturally decomposes into sequential stages (draw a card, then draw another, and so on).

### 2.3 Total Probability Theorem

When the sample space is partitioned into disjoint events $`A_1, A_2, \ldots, A_n`$ (meaning $`A_i \cap A_j = \emptyset`$ for $`i \ne j`$ and $`A_1 \cup \cdots \cup A_n = \Omega`$), we can compute the probability of any event $`B`$ by conditioning on each piece of the partition:

$$P(B) = \sum_{i=1}^{n} P(A_i) \cdot P(B \mid A_i)$$

This theorem is indispensable when a direct calculation of $`P(B)`$ is difficult but the conditional probabilities $`P(B \mid A_i)`$ are easy to find. The idea is to "divide and conquer" by breaking the problem into simpler scenarios.

### 2.4 Bayes' Rule

Bayes' rule allows us to "reverse" conditional probabilities. Given a partition $`A_1, \ldots, A_n`$ of the sample space:

$$P(A_i \mid B) = \frac{P(A_i) \cdot P(B \mid A_i)}{\sum_{j=1}^{n} P(A_j) \cdot P(B \mid A_j)}$$

The denominator is just $`P(B)`$ from the total probability theorem.

**When to use Bayes' rule.** Whenever you want to infer the "cause" from an observed "effect." You start with **prior beliefs** $`P(A_i)`$ about which scenario is in play, you have a **model** of the world $`P(B \mid A_i)`$ describing how likely the observation is under each scenario, and then Bayes' rule gives you the **posterior beliefs** $`P(A_i \mid B)`$ after seeing the evidence.

**Example (radar detection).** Let $`A`$ = "airplane is present" with $`P(A) = 0.05`$. The radar registers something ($`B`$) with $`P(B \mid A) = 0.99`$ and $`P(B \mid A^c) = 0.10`$. Then:

$$P(A \mid B) = \frac{0.05 \times 0.99}{0.05 \times 0.99 + 0.95 \times 0.10} = \frac{0.0495}{0.1445} \approx 0.3426$$

Even with an excellent radar, the posterior probability is only about 34% because the prior probability of an airplane is so low. This is a classic illustration of the **base rate fallacy**.

---

## 3. Independence

### 3.1 Independence of Two Events

Two events $`A`$ and $`B`$ are **independent** if knowing that one occurred gives no information about whether the other occurred. Formally:

$$P(A \cap B) = P(A) \cdot P(B)$$

An equivalent statement (when $`P(B) > 0`$) is $`P(A \mid B) = P(A)`$.

Independence is a modeling assumption, not something you can deduce from the axioms alone. When we model coin tosses as independent, we are asserting that the outcome of one toss does not influence the next.

### 3.2 Conditional Independence

Events $`A`$ and $`B`$ are **conditionally independent given** $`C`$ if:

$$P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

A critical subtlety: independence does not imply conditional independence, and conditional independence does not imply (unconditional) independence. Conditioning can create or destroy independence.

**Example.** Two fair coin tosses. Let $`A`$ = "first toss is heads," $`B`$ = "second toss is heads," $`C`$ = "both tosses gave the same result." Then $`A`$ and $`B`$ are independent (unconditionally), but given $`C`$, knowing the first toss immediately tells you the second, so $`A`$ and $`B`$ are not conditionally independent given $`C`$.

### 3.3 Independence of a Collection of Events

Events $`A_1, A_2, \ldots, A_n`$ are **(mutually) independent** if for every subset $`S \subseteq \lbrace 1, 2, \ldots, n \rbrace`$:

$$P\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)$$

Pairwise independence alone is not sufficient. You need the product formula to hold for every combination of two, three, four, ..., up to $`n`$ events simultaneously.

### 3.4 Reliability

Independence is the key ingredient in reliability analysis. Consider a system of components, each of which works independently with some probability $`p_i`$.

**Series system** (all must work): $`P(\text{system works}) = \prod_{i} p_i`$.

**Parallel system** (at least one must work): $`P(\text{system works}) = 1 - \prod_{i}(1 - p_i)`$.

More complex systems can be analyzed by decomposing them into series and parallel sub-networks.

---

## 4. Counting

Counting is the backbone of computing probabilities under the discrete uniform law, where every outcome is equally likely and $`P(A) = |A| / |\Omega|`$.

### 4.1 Basic Counting Principle (Multiplication Rule)

If an experiment consists of $`r`$ sequential stages, with $`n_i`$ choices at stage $`i`$, the total number of possible outcomes is:

$$n_1 \cdot n_2 \cdots n_r$$

**Example.** Number of license plates with 2 letters followed by 3 digits: $`26 \times 26 \times 10 \times 10 \times 10 = 676{,}000`$. If repetition is prohibited: $`26 \times 25 \times 10 \times 9 \times 8 = 468{,}000`$.

### 4.2 Permutations

The number of ways to arrange $`n`$ distinct elements in a specific order is:

$$n! = n \times (n-1) \times \cdots \times 2 \times 1$$

The number of ways to choose and order $`k`$ items from $`n`$ distinct items (a $`k`$-permutation) is:

$$\frac{n!}{(n-k)!} = n(n-1)\cdots(n-k+1)$$

### 4.3 Combinations (Binomial Coefficient)

The number of ways to choose $`k`$ items from $`n`$ distinct items **without regard to order** is:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

This counts the number of $`k`$-element subsets of an $`n`$-element set. The key identity connecting permutations and combinations is: (number of $`k`$-permutations) = $`\binom{n}{k} \times k!`$, because you first choose which $`k`$ items, then order them.

**Number of subsets** of an $`n`$-element set: $`2^n`$ (each element is either in or out).

### 4.4 Binomial Probabilities

Consider $`n`$ independent coin tosses, each with $`P(\text{Heads}) = p`$. The probability of exactly $`k`$ heads is:

$$P(k \text{ heads}) = \binom{n}{k} p^k (1-p)^{n-k}$$

This arises because any specific sequence with $`k`$ heads has probability $`p^k(1-p)^{n-k}`$, and there are $`\binom{n}{k}`$ such sequences.

### 4.5 Partitions (Multinomial Coefficient)

Given $`n`$ distinct items, the number of ways to partition them into $`r`$ groups of sizes $`n_1, n_2, \ldots, n_r`$ (where $`n_1 + \cdots + n_r = n`$) is:

$$\frac{n!}{n_1! \, n_2! \cdots n_r!}$$

This is the **multinomial coefficient**. It generalizes the binomial coefficient: choosing a subset of size $`k`$ from $`n`$ is a partition into two groups of sizes $`k`$ and $`n-k`$.

**Example (card dealing).** A 52-card deck is dealt fairly to four players (13 cards each). The probability that each player gets exactly one ace:

$$P = \frac{4! \times \frac{48!}{12!\,12!\,12!\,12!}}{\frac{52!}{13!\,13!\,13!\,13!}} = \frac{4 \cdot 3 \cdot 2 \cdot 1 \cdot 13^4}{52 \cdot 51 \cdot 50 \cdot 49} \times \text{(combinatorial simplification)} \approx 0.105$$

---

## 5. Discrete Random Variables

### 5.1 What Is a Random Variable?

A **random variable** $`X`$ is a function from the sample space $`\Omega`$ to the real numbers. It assigns a numerical value to each outcome of the experiment. Random variables allow us to shift our focus from events to numerical quantities, which opens up tools like expectation and variance.

A random variable is **discrete** if it takes values in a finite or countably infinite set.

### 5.2 Probability Mass Function (PMF)

The **probability mass function** (PMF) of a discrete random variable $`X`$ is:

$$p_X(x) = P(X = x)$$

It satisfies two properties: $`p_X(x) \ge 0`$ for all $`x`$, and $`\sum_x p_X(x) = 1`$.

To compute the PMF, collect all outcomes $`\omega`$ such that $`X(\omega) = x`$ and add up their probabilities.

### 5.3 Common Discrete Distributions

**Bernoulli** with parameter $`p \in [0,1]`$. This is the simplest random variable: $`X`$ takes value 1 with probability $`p`$ and value 0 with probability $`1 - p`$. It models a single trial with two outcomes (success/failure, heads/tails).

$$p_X(k) = p^k(1-p)^{1-k}, \quad k \in \lbrace 0, 1 \rbrace$$

**Binomial** with parameters $`n`$ and $`p`$. This counts the number of successes in $`n`$ independent Bernoulli trials:

$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

**Geometric** with parameter $`p`$. This counts the number of independent trials until the first success:

$$p_X(k) = (1-p)^{k-1}p, \quad k = 1, 2, 3, \ldots$$

The geometric distribution is the only discrete distribution with the **memorylessness property**: $`P(X > m + n \mid X > m) = P(X > n)`$.

**Poisson** with parameter $`\lambda > 0`$. This models the number of occurrences in a fixed interval when events happen independently at a constant average rate:

$$p_X(k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

The Poisson distribution arises as a limit of the binomial when $`n`$ is large, $`p`$ is small, and $`\lambda = np`$ is moderate.

### 5.4 Expectation (Mean)

The **expected value** (or mean) of a discrete random variable $`X`$ is:

$$E[X] = \sum_x x \cdot p_X(x)$$

The expectation is the "center of mass" of the PMF. If you repeat the experiment many times and average the observed values of $`X`$, the average converges to $`E[X]`$ (law of large numbers).

**Properties of expectation:**
- If $`X \ge 0`$, then $`E[X] \ge 0`$.
- If $`a \le X \le b`$, then $`a \le E[X] \le b`$.

### 5.5 The Expected Value Rule (LOTUS)

The **law of the unconscious statistician** (LOTUS) lets you compute $`E[g(X)]`$ directly from the PMF of $`X`$, without needing the PMF of $`g(X)`$:

$$E[g(X)] = \sum_x g(x) \cdot p_X(x)$$

This is extremely useful. For instance, to compute $`E[X^2]`$, you simply weight $`x^2`$ by the PMF of $`X`$.

### 5.6 Linearity of Expectation

For any constants $`a`$ and $`b`$:

$$E[aX + b] = aE[X] + b$$

More generally, for any random variables $`X_1, \ldots, X_n`$ (even if they are dependent):

$$E[X_1 + X_2 + \cdots + X_n] = E[X_1] + E[X_2] + \cdots + E[X_n]$$

Linearity is arguably the single most powerful property of expectation. It holds unconditionally, with no independence assumption.

### 5.7 Expectations of Common Distributions

| Distribution | $`E[X]`$ |
|:---|:---|
| Bernoulli($`p`$) | $`p`$ |
| Binomial($`n, p`$) | $`np`$ |
| Geometric($`p`$) | $`1/p`$ |
| Poisson($`\lambda`$) | $`\lambda`$ |

The mean of the binomial follows elegantly from linearity: write $`X = X_1 + \cdots + X_n`$ where each $`X_i`$ is Bernoulli($`p`$), so $`E[X] = nE[X_1] = np`$.

---

## 6. Variance, Conditioning on an Event, and Multiple Random Variables

### 6.1 Variance

The **variance** measures how spread out a distribution is around its mean $`\mu = E[X]`$:

$$\text{var}(X) = E\left[(X - \mu)^2\right]$$

A useful computational shortcut is:

$$\text{var}(X) = E[X^2] - (E[X])^2$$

The **standard deviation** is $`\sigma_X = \sqrt{\text{var}(X)}`$ and has the same units as $`X`$.

### 6.2 Properties of Variance

For constants $`a`$ and $`b`$:

$$\text{var}(aX + b) = a^2 \,\text{var}(X)$$

Adding a constant shifts the distribution but does not change its spread. Multiplying by a constant scales the variance by the square of that constant.

### 6.3 Variances of Common Distributions

| Distribution | $`\text{var}(X)`$ |
|:---|:---|
| Bernoulli($`p`$) | $`p(1-p)`$ |
| Binomial($`n, p`$) | $`np(1-p)`$ |
| Geometric($`p`$) | $`(1-p)/p^2`$ |
| Poisson($`\lambda`$) | $`\lambda`$ |

For the Bernoulli: $`E[X^2] = 0^2(1-p) + 1^2 p = p`$, so $`\text{var}(X) = p - p^2 = p(1-p)`$. The variance of the binomial follows from writing $`X`$ as a sum of independent Bernoullis and using the additivity of variance for independent random variables.

### 6.4 Conditional PMF and Expectation Given an Event

Just as we defined conditional probability, we can define a **conditional PMF** given that event $`A`$ has occurred:

$$p_{X \mid A}(x) = P(X = x \mid A)$$

This is a valid PMF (it sums to 1) and we can compute the **conditional expectation**:

$$E[X \mid A] = \sum_x x \cdot p_{X \mid A}(x)$$

The expected value rule also applies: $`E[g(X) \mid A] = \sum_x g(x) \cdot p_{X \mid A}(x)`$.

### 6.5 Total Expectation Theorem

If events $`A_1, \ldots, A_n`$ partition the sample space, then:

$$E[X] = \sum_{i=1}^{n} P(A_i) \cdot E[X \mid A_i]$$

This is the expectation analogue of the total probability theorem and is very useful for computing expectations by "divide and conquer."

**Example (mean of the geometric).** Let $`X \sim \text{Geometric}(p)`$. Condition on the first trial. If it is a success (probability $`p`$), then $`X = 1`$. If it is a failure (probability $`1 - p`$), then you still need a geometric number of additional trials, so $`X = 1 + X'`$ where $`X'`$ has the same distribution as $`X`$. Therefore:

$$E[X] = p \cdot 1 + (1-p)(1 + E[X])$$

Solving: $`E[X] = 1/p`$.

### 6.6 Multiple Random Variables and Joint PMFs

When we have two discrete random variables $`X`$ and $`Y`$ defined on the same experiment, their **joint PMF** is:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

The individual (marginal) PMFs are obtained by summing out the other variable:

$$p_X(x) = \sum_y p_{X,Y}(x, y), \qquad p_Y(y) = \sum_x p_{X,Y}(x, y)$$

### 6.7 Functions of Multiple Random Variables

If $`Z = g(X,Y)`$, then:

$$E[Z] = E[g(X,Y)] = \sum_x \sum_y g(x,y) \cdot p_{X,Y}(x,y)$$

**Linearity** extends to functions of multiple random variables:

$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

This holds regardless of whether $`X`$ and $`Y`$ are independent.

### 6.8 The Hat Problem

A classic application of linearity. Suppose $`n`$ people each throw their hat into a box, and then each person draws a hat at random. Let $`X`$ be the number of people who get their own hat. Write $`X = X_1 + \cdots + X_n`$, where $`X_i = 1`$ if person $`i`$ gets their own hat. Then $`E[X_i] = 1/n`$ for each $`i`$, so:

$$E[X] = n \cdot \frac{1}{n} = 1$$

Regardless of how large $`n`$ is, the expected number of matches is always 1. Computing the variance (which equals 1 as well) requires more work and involves the covariance terms.

---

## 7. Conditioning on a Random Variable and Independence of Random Variables

### 7.1 Conditional PMFs Given a Random Variable

When we condition on the event $`\lbrace Y = y \rbrace`$, we obtain the **conditional PMF of** $`X`$ **given** $`Y = y`$:

$$p_{X \mid Y}(x \mid y) = P(X = x \mid Y = y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$$

This is defined whenever $`p_Y(y) > 0`$. For each fixed value of $`y`$, the conditional PMF is a legitimate PMF in $`x`$ (it sums to 1 over all $`x`$).

### 7.2 Conditional Expectation Given a Random Variable

The **conditional expectation of** $`X`$ **given** $`Y = y`$ is:

$$E[X \mid Y = y] = \sum_x x \cdot p_{X \mid Y}(x \mid y)$$

### 7.3 Total Probability and Expectation Theorems (Random Variable Version)

Using the possible values of $`Y`$ as a partition:

$$p_X(x) = \sum_y p_Y(y) \cdot p_{X \mid Y}(x \mid y)$$

$$E[X] = \sum_y p_Y(y) \cdot E[X \mid Y = y]$$

### 7.4 Independence of Random Variables

Discrete random variables $`X`$ and $`Y`$ are **independent** if:

$$p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y$$

Equivalently, $`p_{X \mid Y}(x \mid y) = p_X(x)`$ for all $`x`$ and all $`y`$ with $`p_Y(y) > 0`$.

### 7.5 Independence and Expectations

If $`X`$ and $`Y`$ are independent, then:

$$E[XY] = E[X] \cdot E[Y]$$

More generally, $`E[g(X) \cdot h(Y)] = E[g(X)] \cdot E[h(Y)]`$ for any functions $`g`$ and $`h`$. The converse is not true in general: $`E[XY] = E[X]E[Y]`$ does not imply independence.

### 7.6 Independence and Variances

If $`X`$ and $`Y`$ are independent, then:

$$\text{var}(X + Y) = \text{var}(X) + \text{var}(Y)$$

This extends to any number of mutually independent random variables:

$$\text{var}(X_1 + \cdots + X_n) = \text{var}(X_1) + \cdots + \text{var}(X_n)$$

This is the key reason why the variance of the binomial equals $`np(1-p)`$: write the binomial as a sum of $`n`$ independent Bernoulli random variables, each with variance $`p(1-p)`$.

---

## 8. Continuous Random Variables and Probability Density Functions

### 8.1 From PMFs to PDFs

A random variable $`X`$ is **continuous** if there exists a nonnegative function $`f_X(x)`$, called the **probability density function** (PDF), such that for any interval $`[a, b]`$:

$$P(a \le X \le b) = \int_a^b f_X(x) \, dx$$

The PDF satisfies $`f_X(x) \ge 0`$ and $`\int_{-\infty}^{\infty} f_X(x) \, dx = 1`$.

**Key difference from the discrete case:** the PDF $`f_X(x)`$ is not a probability. It is a probability *density*. For a small interval of width $`\delta`$:

$$P(x \le X \le x + \delta) \approx f_X(x) \cdot \delta$$

An immediate consequence: $`P(X = a) = 0`$ for any single point $`a`$. Continuous random variables assign zero probability to individual points.

### 8.2 Expectation of a Continuous Random Variable

$$E[X] = \int_{-\infty}^{\infty} x \, f_X(x) \, dx$$

The expected value rule (LOTUS) for a function $`g(X)`$:

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f_X(x) \, dx$$

Linearity of expectation holds exactly as in the discrete case: $`E[aX + b] = aE[X] + b`$.

### 8.3 Variance

$$\text{var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 f_X(x) \, dx$$

The shortcut formula $`\text{var}(X) = E[X^2] - (E[X])^2`$ and the scaling property $`\text{var}(aX + b) = a^2 \text{var}(X)`$ carry over unchanged.

### 8.4 Common Continuous Distributions

**Continuous Uniform on** $`[a, b]`$.

$$f_X(x) = \frac{1}{b - a}, \quad a \le x \le b$$

$$E[X] = \frac{a + b}{2}, \qquad \text{var}(X) = \frac{(b - a)^2}{12}$$

The uniform distribution models complete ignorance: all values in $`[a, b]`$ are equally likely.

**Exponential with parameter** $`\lambda > 0`$.

$$f_X(x) = \lambda e^{-\lambda x}, \quad x \ge 0$$

$$E[X] = \frac{1}{\lambda}, \qquad \text{var}(X) = \frac{1}{\lambda^2}$$

The exponential distribution models waiting times in Poisson processes. It is the continuous analogue of the geometric distribution and possesses the **memorylessness** property: $`P(X > s + t \mid X > s) = P(X > t)`$.

**Normal (Gaussian)** $`N(\mu, \sigma^2)`$.

$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

$$E[X] = \mu, \qquad \text{var}(X) = \sigma^2$$

The standard normal $`N(0, 1)`$ has $`\mu = 0`$ and $`\sigma^2 = 1`$.

The normal distribution is arguably the most important distribution in probability and statistics because of the **Central Limit Theorem**: the sum of many independent random variables (regardless of their individual distributions) tends toward a normal distribution. It also arises naturally as a model for measurement noise, which is the accumulation of many small, independent error terms.

### 8.5 Standardization

Any normal random variable can be converted to a standard normal. If $`X \sim N(\mu, \sigma^2)`$, define:

$$Z = \frac{X - \mu}{\sigma}$$

Then $`Z \sim N(0, 1)`$. This allows us to look up probabilities using the standard normal CDF $`\Phi(z) = P(Z \le z)`$ from tables or software:

$$P(X \le x) = \Phi\left(\frac{x - \mu}{\sigma}\right)$$

**Useful symmetry:** $`\Phi(-z) = 1 - \Phi(z)`$.

### 8.6 Linear Functions of Normal Random Variables

If $`X \sim N(\mu, \sigma^2)`$ and $`Y = aX + b`$, then:

$$Y \sim N(a\mu + b, \, a^2 \sigma^2)$$

A linear function of a normal random variable is still normal. This is a special property of the normal family.

### 8.7 Cumulative Distribution Function (CDF)

The **CDF** of a random variable $`X`$ is:

$$F_X(x) = P(X \le x)$$

For continuous random variables: $`F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt`$, and the PDF is the derivative of the CDF: $`f_X(x) = \frac{dF_X}{dx}(x)`$.

For discrete random variables: $`F_X(x) = \sum_{k \le x} p_X(k)`$, which is a staircase function.

**General CDF properties:**
- $`F_X`$ is non-decreasing.
- $`F_X(x) \to 0`$ as $`x \to -\infty`$ and $`F_X(x) \to 1`$ as $`x \to +\infty`$.
- $`P(a < X \le b) = F_X(b) - F_X(a)`$.

---

## 9. Conditioning on an Event and Multiple Continuous Random Variables

### 9.1 Conditional PDF Given an Event

If $`A`$ is an event with $`P(A) > 0`$, the conditional PDF of $`X`$ given $`A`$ is defined so that:

$$P(X \in B \mid A) = \int_B f_{X \mid A}(x) \, dx$$

For the special case $`A = \lbrace X \in [a,b] \rbrace`$:

$$f_{X \mid X \in A}(x) = \begin{cases} \dfrac{f_X(x)}{P(A)}, & \text{if } x \in A \\[6pt] 0, & \text{if } x \notin A \end{cases}$$

The conditional expectation given an event is $`E[X \mid A] = \int x \, f_{X \mid A}(x) \, dx`$.

### 9.2 Memorylessness of the Exponential

If $`X \sim \text{Exponential}(\lambda)`$, then for any $`s, t \ge 0`$:

$$P(X > s + t \mid X > s) = P(X > t) = e^{-\lambda t}$$

The exponential distribution "forgets" how long you have already waited. This makes it the natural choice for modeling phenomena like radioactive decay and inter-arrival times in a Poisson process. Among all continuous distributions, the exponential is the **only** one with this property.

### 9.3 Total Probability and Total Expectation (Continuous Version)

If $`A_1, \ldots, A_n`$ partition $`\Omega`$:

$$f_X(x) = \sum_{i} P(A_i) \cdot f_{X \mid A_i}(x)$$

$$E[X] = \sum_{i} P(A_i) \cdot E[X \mid A_i]$$

### 9.4 Mixed Distributions

Some random variables are neither purely discrete nor purely continuous. They assign positive probability to some individual points while also spreading probability over intervals. For example, suppose $`X = 0`$ with probability 1/4 and $`X \sim \text{Uniform}[1,2]`$ with probability 3/4. Such a variable has a **mixed distribution** described by a combination of point masses and a density.

The CDF of a mixed distribution has both jumps (at point masses) and continuous segments (where the PDF is positive).

### 9.5 Jointly Continuous Random Variables

Two random variables $`X`$ and $`Y`$ are **jointly continuous** if there exists a nonnegative function $`f_{X,Y}(x,y)`$ (the **joint PDF**) such that for any region $`B \subseteq \mathbb{R}^2`$:

$$P((X,Y) \in B) = \iint_B f_{X,Y}(x,y) \, dx \, dy$$

The joint PDF satisfies $`\int \int f_{X,Y}(x,y) \, dx \, dy = 1`$.

### 9.6 Marginal PDFs

The individual (marginal) PDFs are obtained by integrating out the other variable:

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy, \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx$$

### 9.7 Expected Value Rule and Linearity (Joint)

If $`Z = g(X,Y)`$:

$$E[Z] = \int \int g(x,y) \, f_{X,Y}(x,y) \, dx \, dy$$

Linearity of expectation holds: $`E[aX + bY + c] = aE[X] + bE[Y] + c`$, regardless of the dependence structure between $`X`$ and $`Y`$.

### 9.8 The Joint CDF

$$F_{X,Y}(x,y) = P(X \le x, Y \le y) = \int_{-\infty}^{x} \int_{-\infty}^{y} f_{X,Y}(s,t) \, dt \, ds$$

The joint PDF is recovered by differentiation: $`f_{X,Y}(x,y) = \frac{\partial^2 F_{X,Y}}{\partial x \, \partial y}(x,y)`$.

---

## 10. Conditioning on a Random Variable, Independence, and Bayes' Rule for Continuous Variables

### 10.1 Conditional PDFs Given Another Random Variable

The **conditional PDF of** $`X`$ **given** $`Y = y`$ is:

$$f_{X \mid Y}(x \mid y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

This is defined whenever $`f_Y(y) > 0`$. For each fixed $`y`$, it is a valid PDF in $`x`$ (integrates to 1).

**Remark.** The notation $`P(X = x \mid Y = y)`$ does not make literal sense for continuous variables (both events have probability zero), but the conditional PDF is rigorously defined via the limiting argument $`f_{X \mid Y}(x \mid y) \cdot \delta \approx P(x \le X \le x + \delta \mid y \le Y \le y + \delta)`$.

### 10.2 Conditional Expectation

$$E[X \mid Y = y] = \int_{-\infty}^{\infty} x \, f_{X \mid Y}(x \mid y) \, dx$$

### 10.3 Total Probability and Expectation Theorems (Continuous Version)

Using $`Y`$ as the conditioning variable (integrating over all values of $`y`$):

$$f_X(x) = \int_{-\infty}^{\infty} f_Y(y) \cdot f_{X \mid Y}(x \mid y) \, dy$$

$$E[X] = \int_{-\infty}^{\infty} f_Y(y) \cdot E[X \mid Y = y] \, dy$$

### 10.4 Independence of Continuous Random Variables

Continuous random variables $`X`$ and $`Y`$ are **independent** if:

$$f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y) \quad \text{for all } x, y$$

Equivalently, $`f_{X \mid Y}(x \mid y) = f_X(x)`$ for all $`x, y`$.

If $`X`$ and $`Y`$ are independent, then $`E[g(X) \cdot h(Y)] = E[g(X)] \cdot E[h(Y)]`$ and $`\text{var}(X + Y) = \text{var}(X) + \text{var}(Y)`$.

### 10.5 Independent Normal Random Variables

If $`X \sim N(\mu_X, \sigma_X^2)`$ and $`Y \sim N(\mu_Y, \sigma_Y^2)`$ are independent, then the joint PDF factors into the product of the two marginal PDFs. This means that the contours of the joint PDF are circles (if $`\sigma_X = \sigma_Y`$) or axis-aligned ellipses (if $`\sigma_X \ne \sigma_Y`$).

### 10.6 Bayes' Rule for Continuous Random Variables

**Inference about a continuous unknown based on a discrete observation.** If $`X`$ is continuous and $`K`$ is discrete:

$$f_{X \mid K}(x \mid k) = \frac{p_{K \mid X}(k \mid x) \cdot f_X(x)}{p_K(k)}$$

where $`p_K(k) = \int p_{K \mid X}(k \mid x) \cdot f_X(x) \, dx`$.

**Inference about a continuous unknown based on a continuous observation.** If both $`X`$ and $`Y`$ are continuous:

$$f_{X \mid Y}(x \mid y) = \frac{f_{Y \mid X}(y \mid x) \cdot f_X(x)}{f_Y(y)}$$

where $`f_Y(y) = \int f_{Y \mid X}(y \mid x) \cdot f_X(x) \, dx`$.

In both cases, the structure is the same: posterior $`\propto`$ likelihood $`\times`$ prior. This is the continuous version of the Bayes' rule we saw in Section 2.

---

## 11. Derived Distributions

### 11.1 The Problem

If $`X`$ is a random variable with known distribution and $`Y = g(X)`$, what is the distribution of $`Y`$? This question arises constantly: if we know the distribution of a signal, what is the distribution of the signal after it passes through a nonlinear system?

### 11.2 The Discrete Case

If $`X`$ is discrete and $`Y = g(X)`$:

$$p_Y(y) = P(g(X) = y) = \sum_{x:\, g(x) = y} p_X(x)$$

Collect all values of $`x`$ that map to the same $`y`$ and add up their probabilities.

### 11.3 Linear Functions of Continuous Random Variables

If $`Y = aX + b`$ where $`a \ne 0`$:

$$f_Y(y) = \frac{1}{|a|} f_X\left(\frac{y - b}{a}\right)$$

The $`|a|`$ in the denominator comes from the Jacobian of the transformation. If $`a > 0`$ the PDF shifts and scales; if $`a < 0`$ it also flips.

**Special case:** if $`X \sim N(\mu, \sigma^2)`$ and $`Y = aX + b`$, then $`Y \sim N(a\mu + b, a^2\sigma^2)`$.

### 11.4 General Monotonic Functions

If $`Y = g(X)`$ where $`g`$ is strictly monotonic and differentiable, and $`h = g^{-1}`$ is the inverse function:

$$f_Y(y) = f_X(h(y)) \cdot \left|\frac{dh}{dy}(y)\right|$$

The absolute value of the derivative of the inverse accounts for the stretching or compression of probability when you change variables.

### 11.5 The Two-Step CDF Method

For an arbitrary (possibly nonmonotonic) function $`g`$, the general approach is:

**Step 1.** Find the CDF of $`Y`$:

$$F_Y(y) = P(Y \le y) = P(g(X) \le y)$$

Express $`\lbrace g(X) \le y \rbrace`$ in terms of $`X`$ and evaluate the probability.

**Step 2.** Differentiate:

$$f_Y(y) = \frac{dF_Y}{dy}(y)$$

**Example:** $`Y = X^2`$ where $`X \sim \text{Uniform}[0, 1]`$. For $`0 \le y \le 1`$:

$$F_Y(y) = P(X^2 \le y) = P(X \le \sqrt{y}) = \sqrt{y}$$

$$f_Y(y) = \frac{1}{2\sqrt{y}}, \quad 0 < y \le 1$$

### 11.6 Nonmonotonic Functions

If $`Y = X^2`$ and $`X`$ ranges over both positive and negative values, the CDF approach gives:

$$F_Y(y) = P(X^2 \le y) = P(-\sqrt{y} \le X \le \sqrt{y}) = F_X(\sqrt{y}) - F_X(-\sqrt{y})$$

Differentiating:

$$f_Y(y) = \frac{1}{2\sqrt{y}} \left[f_X(\sqrt{y}) + f_X(-\sqrt{y})\right], \quad y > 0$$

### 11.7 Functions of Multiple Random Variables

If $`Z = g(X, Y)`$ where $`X`$ and $`Y`$ have a known joint distribution, the CDF method still works:

$$F_Z(z) = P(g(X,Y) \le z) = \iint_{\lbrace (x,y): g(x,y) \le z \rbrace} f_{X,Y}(x,y) \, dx \, dy$$

Then differentiate to get $`f_Z(z)`$.

---

## 12. Sums of Independent Random Variables and Covariance

### 12.1 The Distribution of a Sum: Convolution

Let $`Z = X + Y`$ where $`X`$ and $`Y`$ are independent. The distribution of $`Z`$ is given by the **convolution** of the distributions of $`X`$ and $`Y`$.

**Discrete case:**

$$p_Z(z) = \sum_x p_X(x) \cdot p_Y(z - x)$$

**Continuous case:**

$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) \cdot f_Y(z - x) \, dx$$

**Convolution mechanics (discrete).** To compute $`p_Z(z)`$ for a specific $`z`$: flip the PMF of $`Y`$ horizontally, shift it by $`z`$, multiply term-by-term with the PMF of $`X`$, and sum the products.

### 12.2 Sum of Independent Normals

If $`X \sim N(\mu_X, \sigma_X^2)`$ and $`Y \sim N(\mu_Y, \sigma_Y^2)`$ are independent, then:

$$X + Y \sim N(\mu_X + \mu_Y, \, \sigma_X^2 + \sigma_Y^2)$$

This extends to any finite sum: the sum of finitely many independent normal random variables is itself normal, with mean equal to the sum of the means and variance equal to the sum of the variances. This closure property makes the normal distribution especially tractable.

### 12.3 Covariance

The **covariance** of two random variables $`X`$ and $`Y`$ measures how they vary together:

$$\text{cov}(X, Y) = E\left[(X - E[X])(Y - E[Y])\right] = E[XY] - E[X] \cdot E[Y]$$

If $`X`$ and $`Y`$ tend to be above (or below) their means simultaneously, the covariance is positive. If one tends to be above its mean when the other is below, the covariance is negative.

**Key properties of covariance:**
- $`\text{cov}(X, X) = \text{var}(X)`$
- $`\text{cov}(X, Y) = \text{cov}(Y, X)`$ (symmetry)
- $`\text{cov}(aX + b, Y) = a \cdot \text{cov}(X, Y)`$ (linearity in first argument)
- $`\text{cov}(X, Y + Z) = \text{cov}(X, Y) + \text{cov}(X, Z)`$ (additivity)
- If $`X`$ and $`Y`$ are independent, then $`\text{cov}(X, Y) = 0`$

The converse of the last property is **not true**: zero covariance does not imply independence. Covariance only captures linear association.

### 12.4 Variance of a Sum

For any two random variables:

$$\text{var}(X + Y) = \text{var}(X) + \text{var}(Y) + 2\,\text{cov}(X, Y)$$

For $`n`$ random variables:

$$\text{var}(X_1 + \cdots + X_n) = \sum_{i=1}^{n} \text{var}(X_i) + \sum_{i \ne j} \text{cov}(X_i, X_j)$$

If the $`X_i`$ are pairwise uncorrelated (which is implied by independence), all covariance terms vanish and variance is additive.

### 12.5 The Correlation Coefficient

The **correlation coefficient** is a dimensionless version of covariance:

$$\rho(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

**Key properties:**
- $`-1 \le \rho(X, Y) \le 1`$
- $`|\rho| = 1`$ if and only if $`X`$ and $`Y`$ are linearly related: $`X - E[X] = c(Y - E[Y])`$ for some constant $`c`$.
- $`\rho = 0`$ means "uncorrelated" (no linear association), which is weaker than independence.
- $`\rho(aX + b, Y) = \text{sign}(a) \cdot \rho(X, Y)`$ (scale-invariant, sign depends on direction of scaling).

**Correlation does not imply causation.** Two variables may be correlated because they share a common underlying factor, not because one causes the other. For instance, math aptitude $`X`$ and musical ability $`Y`$ might be correlated because both are influenced by a hidden factor $`Z`$ (general cognitive ability), even though neither directly causes the other.

### 12.6 Practical Impact of Correlations

**Example (portfolio variance).** A real-estate company invests \$10M in each of 10 states. Let $`X_i`$ be the return from state $`i`$, with $`E[X_i] = 1`$ and $`\sigma_{X_i} = 1.3`$ (in millions). The total return is $`S = X_1 + \cdots + X_{10}`$, with $`E[S] = 10`$.

If the $`X_i`$ are uncorrelated: $`\text{var}(S) = 10 \times 1.3^2 = 16.9`$, so $`\sigma_S \approx 4.1`$.

If each pair has correlation $`\rho = 0.9`$: each $`\text{cov}(X_i, X_j) = 0.9 \times 1.3^2 = 1.521`$, and there are $`10 \times 9 = 90`$ such pairs, so $`\text{var}(S) = 16.9 + 90 \times 1.521 = 153.8`$, giving $`\sigma_S \approx 12.4`$. Positive correlations dramatically increase portfolio risk.

---

## 13. Conditional Expectation and Variance Revisited

### 13.1 Conditional Expectation as a Random Variable

In earlier sections we defined $`E[X \mid Y = y]`$ as a number for each fixed value $`y`$. If we define $`g(y) = E[X \mid Y = y]`$, then $`g(Y)`$ is a random variable (because $`Y`$ is random). We write:

$$E[X \mid Y] = g(Y)$$

This is a random variable that takes the value $`E[X \mid Y = y]`$ whenever $`Y`$ happens to equal $`y`$. It has its own distribution, mean, and variance.

### 13.2 Law of Iterated Expectations (Tower Property)

$$E\left[E[X \mid Y]\right] = E[X]$$

This is one of the most elegant and powerful results in probability. It says: if you first compute the conditional expectation of $`X`$ given $`Y`$ (producing a random variable), and then take the unconditional expectation of that, you get back the overall expectation of $`X`$.

**Intuition.** Think of $`Y`$ as defining "scenarios." In each scenario, you compute the expected value of $`X`$. The overall expected value of $`X`$ is the weighted average of these scenario-specific expectations, where the weights are the probabilities of the scenarios.

**Example (stick-breaking).** A stick of length $`\ell`$ is broken at a uniformly chosen point $`Y`$. The left piece (of length $`Y`$) is then broken at a uniformly chosen point $`X`$. Then $`E[X \mid Y = y] = y/2`$, so $`E[X \mid Y] = Y/2`$, and:

$$E[X] = E[Y/2] = \frac{1}{2} E[Y] = \frac{1}{2} \cdot \frac{\ell}{2} = \frac{\ell}{4}$$

**Application (forecast revisions).** If forecasts are computed as conditional expectations, then the law of iterated expectations guarantees that the average of revised forecasts (averaging over possible new information) equals the original forecast. Forecasts computed via conditional expectations are **unbiased** in this sense.

### 13.3 Conditional Variance as a Random Variable

Analogously, define:

$$\text{var}(X \mid Y) = E\left[(X - E[X \mid Y])^2 \mid Y\right]$$

This is a random variable: for each value $`y`$, it equals the conditional variance $`\text{var}(X \mid Y = y)`$.

### 13.4 Law of Total Variance

$$\text{var}(X) = E\left[\text{var}(X \mid Y)\right] + \text{var}\left(E[X \mid Y]\right)$$

This decomposes the total variance of $`X`$ into two interpretable pieces:

- $`E[\text{var}(X \mid Y)]`$: the **average variability within** groups defined by $`Y`$.
- $`\text{var}(E[X \mid Y])`$: the **variability between** group means.

**Example (section means and variances).** A class has two sections: section 1 has 10 students, section 2 has 20 students. You pick a student at random and observe their score $`X`$. Let $`Y`$ denote the section. Section 1 has mean score 90 and variance 10; section 2 has mean score 60 and variance 20.

$$E[\text{var}(X \mid Y)] = \frac{1}{3} \cdot 10 + \frac{2}{3} \cdot 20 = \frac{50}{3} \quad \text{(within-section variability)}$$

$$\text{var}(E[X \mid Y]) = \frac{1}{3}(90 - 70)^2 + \frac{2}{3}(60 - 70)^2 = \frac{800}{3} \quad \text{(between-section variability)}$$

$$\text{var}(X) = \frac{50}{3} + \frac{800}{3} = \frac{850}{3}$$

### 13.5 Sum of a Random Number of Independent Random Variables

Suppose $`N`$ is a nonnegative integer-valued random variable, and $`X_1, X_2, \ldots`$ are i.i.d. random variables, independent of $`N`$. Define:

$$Y = X_1 + X_2 + \cdots + X_N$$

**Mean:**

$$E[Y] = E[N] \cdot E[X]$$

This follows from the law of iterated expectations: $`E[Y \mid N = n] = nE[X]`$, so $`E[Y \mid N] = N \cdot E[X]`$, and taking expectations gives $`E[Y] = E[N] \cdot E[X]`$.

**Variance:**

$$\text{var}(Y) = E[N] \cdot \text{var}(X) + (E[X])^2 \cdot \text{var}(N)$$

This follows from the law of total variance: $`\text{var}(Y \mid N) = N \cdot \text{var}(X)`$, and $`E[Y \mid N] = N \cdot E[X]`$. Then:

$$E[\text{var}(Y \mid N)] = E[N] \cdot \text{var}(X)$$

$$\text{var}(E[Y \mid N]) = \text{var}(N \cdot E[X]) = (E[X])^2 \cdot \text{var}(N)$$

**Example.** You visit $`N`$ stores, where $`N`$ has some distribution, and spend $`X_i`$ at store $`i`$. The total spending $`Y = X_1 + \cdots + X_N`$ has mean $`E[N] \cdot E[X_i]`$ and variance $`E[N] \cdot \text{var}(X_i) + (E[X_i])^2 \cdot \text{var}(N)`$.

---

## Quick Reference: Key Distributions Summary

| Distribution | PMF / PDF | Mean | Variance | Key Property |
|:---|:---|:---|:---|:---|
| Bernoulli($`p`$) | $`p^k(1-p)^{1-k}`$ | $`p`$ | $`p(1-p)`$ | Single trial |
| Binomial($`n,p`$) | $`\binom{n}{k}p^k(1-p)^{n-k}`$ | $`np`$ | $`np(1-p)`$ | Sum of Bernoullis |
| Geometric($`p`$) | $`(1-p)^{k-1}p`$ | $`1/p`$ | $`(1-p)/p^2`$ | Memoryless (discrete) |
| Poisson($`\lambda`$) | $`e^{-\lambda}\lambda^k / k!`$ | $`\lambda`$ | $`\lambda`$ | Limit of Binomial |
| Uniform[$`a,b`$] | $`1/(b-a)`$ | $`(a+b)/2`$ | $`(b-a)^2/12`$ | Equal likelihood |
| Exponential($`\lambda`$) | $`\lambda e^{-\lambda x}`$ | $`1/\lambda`$ | $`1/\lambda^2`$ | Memoryless (continuous) |
| Normal($`\mu, \sigma^2`$) | $`\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}`$ | $`\mu`$ | $`\sigma^2`$ | CLT, closed under sums |

---

## Quick Reference: Key Formulas

**Bayes' Rule:**\
$$P(A_i \mid B) = \frac{P(A_i)P(B \mid A_i)}{\sum_j P(A_j)P(B \mid A_j)}$$

**Total Probability Theorem:**\
$$P(B) = \sum_i P(A_i)P(B \mid A_i)$$

**Total Expectation Theorem:**\
$$E[X] = \sum_i P(A_i) \cdot E[X \mid A_i]$$

**Law of Iterated Expectations:**\
$$E[E[X \mid Y]] = E[X]$$

**Law of Total Variance:**\
$$\text{var}(X) = E[\text{var}(X \mid Y)] + \text{var}(E[X \mid Y])$$

**Convolution (sum of independent r.v.'s):**\
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, dx$$

**Derived Distribution (monotonic** $`g`$**):**\
$$f_Y(y) = f_X(h(y)) \left|\frac{dh}{dy}\right| \quad \text{where } h = g^{-1}$$

**Covariance:**\
$$\text{cov}(X,Y) = E[XY] - E[X]E[Y]$$

**Variance of a Sum:**\
$$\text{var}\left(\sum_i X_i\right) = \sum_i \text{var}(X_i) + \sum_{i \ne j} \text{cov}(X_i, X_j)$$

**Correlation Coefficient:**\
$$\rho(X,Y) = \frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y}, \quad -1 \le \rho \le 1$$

**Random Sum:**\
$$E[X_1 + \cdots + X_N] = E[N]E[X], \qquad \text{var}(X_1 + \cdots + X_N) = E[N]\,\text{var}(X) + (E[X])^2\,\text{var}(N)$$

---

*End of Part 1: The Fundamentals. Parts 2 (Inference and Limit Theorems) and 3 (Random Processes) to follow.*
