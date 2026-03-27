# Part III: Random Processes

> **Comprehensive Course Notes**
> Based on MIT 6.041 *Introduction to Probability* (Tsitsiklis & Jaillet) and Dennis Sun's *Introduction to Probability*
> Part 3 of 3 | [Part 1: The Fundamentals](./Probability_Part1_Fundamentals.md) | [Part 2: Inference and Limit Theorems](./Probability_Part2_Inference_and_Limit_Theorems.md)

---

## Table of Contents

1. [The Bernoulli Process](#1-the-bernoulli-process)
2. [The Poisson Process](#2-the-poisson-process)
3. [More on the Poisson Process: Merging, Splitting, and Random Incidence](#3-more-on-the-poisson-process-merging-splitting-and-random-incidence)
4. [Markov Chains: Definitions and State Classification](#4-markov-chains-definitions-and-state-classification)
5. [Markov Chains: Steady-State Behavior and Birth-Death Processes](#5-markov-chains-steady-state-behavior-and-birth-death-processes)
6. [Markov Chains: Absorption, First Passage, and Applications](#6-markov-chains-absorption-first-passage-and-applications)
7. [General Random Processes and Signal Processing Foundations](#7-general-random-processes-and-signal-processing-foundations)

---

## 1. The Bernoulli Process

### 1.1 Definition and Setup

A **Bernoulli process** is a sequence of independent Bernoulli trials $`X_1, X_2, X_3, \ldots`$ where each trial has the same probability of success:

$$P(X_i = 1) = p, \qquad P(X_i = 0) = 1 - p$$

The two defining assumptions are **independence** (the outcome of any trial does not affect the others) and **time-homogeneity** (the success probability $`p`$ is the same at every trial). This is the simplest nontrivial stochastic process and serves as a building block for more complex models. It naturally models sequences of lottery outcomes, arrivals at a server in each time slot, or any repeated experiment with binary outcomes.

### 1.2 Stochastic Processes: Two Views

A stochastic process can be understood from two perspectives. The **first view** treats it as a sequence of random variables $`X_1, X_2, \ldots`$ with a joint distribution. For the Bernoulli process, the joint PMF factors completely: $`p_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = p_{X_1}(x_1) \cdots p_{X_n}(x_n)`$ for all $`n`$, which is just a restatement of independence.

The **second view** treats the sample space $`\Omega`$ as the set of all infinite sequences of 0s and 1s. Each outcome $`\omega \in \Omega`$ is an entire realization of the process. For instance, $`P(X_i = 1 \text{ for all } i) = 0`$ when $`p < 1`$, because this probability is at most $`p^n`$ for every $`n`$.

### 1.3 Number of Successes in $`n`$ Trials

The total number of successes in $`n`$ time slots is $`S = X_1 + \cdots + X_n`$, which follows a Binomial distribution:

$$P(S = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad k = 0, 1, \ldots, n$$

$$E[S] = np, \qquad \text{var}(S) = np(1-p)$$

### 1.4 Time Until the First Success

The time of the first success $`T_1 = \min\lbrace i : X_i = 1 \rbrace`$ follows a Geometric distribution:

$$P(T_1 = k) = (1-p)^{k-1} p, \qquad k = 1, 2, 3, \ldots$$

$$E[T_1] = \frac{1}{p}, \qquad \text{var}(T_1) = \frac{1-p}{p^2}$$

### 1.5 Fresh-Start and Memorylessness Properties

The Bernoulli process has a powerful **fresh-start property**. If we start watching at any fixed time $`n`$, the process $`X_{n+1}, X_{n+2}, \ldots`$ is itself a Bernoulli process with the same parameter $`p`$, and it is independent of $`X_1, \ldots, X_n`$.

Even more remarkably, this holds after a **random time** $`N`$, as long as $`N`$ is **causally determined**, meaning that whether $`N = n`$ depends only on $`X_1, \ldots, X_n`$ (not on future values). For example, $`N`$ = "time of the 3rd success" is causally determined, so the process restarts fresh after the 3rd success. But $`N`$ = "the time just before the first occurrence of 1,1,1" is not causally determined (you need to peek ahead), so the fresh-start property does not apply.

### 1.6 Interarrival Times and the Pascal Distribution

Let $`Y_k`$ denote the time of the $`k`$th success and $`T_k = Y_k - Y_{k-1}`$ (with $`Y_0 = 0`$) the $`k`$th interarrival time. By the fresh-start property, the interarrival times $`T_1, T_2, T_3, \ldots`$ are i.i.d. Geometric($`p`$).

The time of the $`k`$th success is $`Y_k = T_1 + \cdots + T_k`$, a sum of $`k`$ i.i.d. geometrics. Its PMF is the **Pascal (negative binomial) distribution**:

$$p_{Y_k}(t) = \binom{t-1}{k-1} p^k (1-p)^{t-k}, \qquad t = k, k+1, \ldots$$

$$E[Y_k] = \frac{k}{p}, \qquad \text{var}(Y_k) = \frac{k(1-p)}{p^2}$$

The logic: for $`Y_k = t`$, we need exactly $`k - 1`$ successes in the first $`t - 1`$ trials and then a success at time $`t`$.

### 1.7 Merging Bernoulli Processes

If $`\lbrace X_t \rbrace \sim \text{Bernoulli}(p)`$ and $`\lbrace Y_t \rbrace \sim \text{Bernoulli}(q)`$ are independent, the merged process $`Z_t = \max(X_t, Y_t)`$ (an arrival occurs if either process has one) is Bernoulli with parameter $`p + q - pq`$. Given that an arrival occurred in the merged process at time $`t`$, the probability it came from the first process is $`p/(p + q - pq)`$.

### 1.8 Splitting a Bernoulli Process

Given a Bernoulli($`p`$) process, each arrival is independently routed to stream 1 with probability $`q`$ and stream 2 with probability $`1 - q`$. Stream 1 is Bernoulli($`pq`$) and stream 2 is Bernoulli($`p(1-q)`$). However, the two resulting streams are **not independent** (unlike the Poisson case), because knowing that both streams had no arrival at time $`t`$ gives information about the original process.

### 1.9 Poisson Approximation to the Binomial

When $`n`$ is large, $`p`$ is small, and $`\lambda = np`$ is moderate, the Binomial($`n, p`$) PMF is well approximated by the Poisson($`\lambda`$) PMF:

$$\binom{n}{k} p^k (1-p)^{n-k} \approx \frac{\lambda^k e^{-\lambda}}{k!}$$

This follows from the fact that $`\lim_{n \to \infty}(1 - \lambda/n)^n = e^{-\lambda}`$. The approximation is the bridge connecting the Bernoulli process (discrete time) to the Poisson process (continuous time).

---

## 2. The Poisson Process

### 2.1 Definition

The **Poisson process** is the continuous-time analogue of the Bernoulli process. It models arrivals occurring at random times along a continuous time axis. The defining properties are:

**Independence.** The numbers of arrivals in disjoint time intervals are independent.

**Small-interval probabilities.** For a very small interval of duration $`\delta`$:

$$P(k, \delta) \approx \begin{cases} 1 - \lambda\delta & \text{if } k = 0 \\ \lambda\delta & \text{if } k = 1 \\ 0 & \text{if } k > 1 \end{cases}$$

More precisely, $`P(k, \delta) = \lambda\delta + O(\delta^2)`$ for $`k = 1`$ and $`P(k, \delta) = O(\delta^2)`$ for $`k \ge 2`$.

The parameter $`\lambda > 0`$ is the **arrival rate** (expected number of arrivals per unit time). The process arises naturally as the limit of a Bernoulli process: divide $`[0, \tau]`$ into $`n = \tau/\delta`$ small slots, each with success probability $`p = \lambda\delta`$, and let $`\delta \to 0`$.

**Applications** include radioactive decay, photon arrivals, financial market shocks, phone call placements, and service requests.

### 2.2 Number of Arrivals in an Interval

Let $`N_\tau`$ denote the number of arrivals in an interval of duration $`\tau`$. Then $`N_\tau`$ follows a Poisson distribution:

$$P(k, \tau) = P(N_\tau = k) = \frac{(\lambda\tau)^k e^{-\lambda\tau}}{k!}, \qquad k = 0, 1, 2, \ldots$$

$$E[N_\tau] = \lambda\tau, \qquad \text{var}(N_\tau) = \lambda\tau$$

The derivation connects directly to the Poisson approximation: $`N_\tau \approx \text{Binomial}(n, p)`$ with $`n = \tau/\delta`$ and $`p = \lambda\delta`$, so $`np = \lambda\tau`$, and as $`\delta \to 0`$ the binomial converges to Poisson($`\lambda\tau`$).

### 2.3 The Time Until the First Arrival

The time $`T_1`$ until the first arrival has CDF:

$$P(T_1 \le t) = 1 - P(T_1 > t) = 1 - P(0, t) = 1 - e^{-\lambda t}$$

Differentiating, $`f_{T_1}(t) = \lambda e^{-\lambda t}`$ for $`t \ge 0`$. This is the **Exponential($`\lambda`$)** distribution, with $`E[T_1] = 1/\lambda`$ and $`\text{var}(T_1) = 1/\lambda^2`$.

### 2.4 Memorylessness and the Fresh-Start Property

The Poisson process inherits a fresh-start property analogous to the Bernoulli process. If we start watching at any fixed time $`t`$, the future process is a Poisson process with rate $`\lambda`$, independent of the history up to time $`t`$. The time until the next arrival from time $`t`$ is Exponential($`\lambda`$), regardless of when the last arrival occurred. This is the **memorylessness** of the exponential distribution: $`P(T_1 > s + t \mid T_1 > s) = P(T_1 > t)`$.

The fresh-start property also holds after a random time $`N`$ that is a stopping time (causally determined), such as $`N = T_1`$ (the time of the first arrival).

### 2.5 Interarrival Times and the Erlang Distribution

By the fresh-start property, the interarrival times $`T_k = Y_k - Y_{k-1}`$ are i.i.d. Exponential($`\lambda`$). The time of the $`k`$th arrival $`Y_k = T_1 + \cdots + T_k`$ follows the **Erlang distribution** of order $`k`$:

$$f_{Y_k}(y) = \frac{\lambda^k y^{k-1} e^{-\lambda y}}{(k-1)!}, \qquad y \ge 0$$

$$E[Y_k] = \frac{k}{\lambda}, \qquad \text{var}(Y_k) = \frac{k}{\lambda^2}$$

The Erlang distribution is the continuous-time counterpart of the Pascal distribution.

### 2.6 Bernoulli-Poisson Correspondence

The following table summarizes the parallel structure of the two processes:

| Property | Poisson | Bernoulli |
|:---|:---|:---|
| Time of arrival | Continuous | Discrete |
| Arrival rate | $`\lambda`$ per unit time | $`p`$ per trial |
| Number of arrivals | Poisson | Binomial |
| Interarrival time | Exponential | Geometric |
| Time to $`k`$th arrival | Erlang | Pascal (Neg. Binomial) |

---

## 3. More on the Poisson Process: Merging, Splitting, and Random Incidence

### 3.1 Sum of Independent Poisson Random Variables

If $`M \sim \text{Poisson}(\mu)`$ and $`N \sim \text{Poisson}(\nu)`$ are independent, then:

$$M + N \sim \text{Poisson}(\mu + \nu)$$

This follows naturally from the Poisson process: consider consecutive intervals of lengths $`\mu/\lambda`$ and $`\nu/\lambda`$ in a Poisson process of rate $`\lambda = 1`$. The counts $`M`$ and $`N`$ are independent (disjoint intervals) and Poisson, and their sum equals the count in the combined interval.

### 3.2 Merging Independent Poisson Processes

If two independent Poisson processes with rates $`\lambda_1`$ and $`\lambda_2`$ are combined (superimposed), the merged process is Poisson with rate $`\lambda_1 + \lambda_2`$. This is verified by checking the small-interval probabilities: in a slot of duration $`\delta`$, the probability of exactly one arrival in the merged process is $`(\lambda_1 + \lambda_2)\delta + O(\delta^2)`$, and two or more arrivals have probability $`O(\delta^2)`$.

Given that an arrival occurs in the merged process at some time, the probability it came from process 1 is:

$$P(\text{from process 1} \mid \text{arrival}) = \frac{\lambda_1}{\lambda_1 + \lambda_2}$$

The "source" decisions for different arrivals are independent. So if we observe $`n`$ arrivals in the merged process, the number from process 1 is Binomial($`n, \lambda_1/(\lambda_1 + \lambda_2)`$).

### 3.3 Splitting a Poisson Process

Given a Poisson process of rate $`\lambda`$, each arrival is independently routed to stream 1 (with probability $`q`$) or stream 2 (with probability $`1 - q`$). The resulting streams are Poisson with rates $`\lambda q`$ and $`\lambda(1 - q)`$ respectively.

A remarkable fact: unlike the Bernoulli case, the two split Poisson streams are **independent** of each other. This is because in any small interval, the probability of an arrival in stream 1 is $`\lambda q \delta`$ and in stream 2 is $`\lambda(1-q)\delta`$, and the probability of both is $`O(\delta^2)`$, so they are effectively independent at the infinitesimal level.

### 3.4 Competing Exponentials

Consider $`n`$ independent exponential random variables $`X_1, \ldots, X_n`$ with rates $`\lambda_1, \ldots, \lambda_n`$. The minimum $`\min(X_1, \ldots, X_n)`$ is Exponential with rate $`\lambda_1 + \cdots + \lambda_n`$.

**Example (lightbulb burnout).** Three independent lightbulbs, each with Exponential($`\lambda`$) lifetime. The expected time until the first burns out is $`1/(3\lambda)`$, because $`\min(X, Y, Z) \sim \text{Exponential}(3\lambda)`$. The expected time until all burn out is $`1/(3\lambda) + 1/(2\lambda) + 1/\lambda = 11/(6\lambda)`$: after the first burnout, two remain (merged rate $`2\lambda`$), and after the second, one remains (rate $`\lambda`$).

### 3.5 Random Incidence (Inspection Paradox)

Suppose a Poisson process has been running for a long time. You arrive at some time $`t^*`$ and measure the length of the interarrival interval you land in. Let $`U`$ be the time of the last arrival before $`t^*`$ and $`V`$ the time of the next arrival after $`t^*`$.

By memorylessness, $`V - t^*`$ is Exponential($`\lambda`$), independent of the past. The elapsed time $`t^* - U`$ is also Exponential($`\lambda`$) by time-reversal symmetry. Therefore:

$$E[V - U] = E[V - t^*] + E[t^* - U] = \frac{1}{\lambda} + \frac{1}{\lambda} = \frac{2}{\lambda}$$

The interarrival interval you observe has expected length $`2/\lambda`$, which is twice the expected interarrival time $`1/\lambda`$. This is the **inspection paradox** (or random incidence): you are more likely to land in a longer interval simply because longer intervals occupy more of the time axis.

This paradox is not special to the Poisson process. For any renewal process with i.i.d. interarrival times, a random observer is biased toward longer intervals. For example, if interarrival times are equally likely to be 5 or 10 minutes ($`E[T_k] = 7.5`$), the probability of landing in a 10-minute interval is $`2/3`$ (not $`1/2`$), so the expected observed interval is $`(1/3)(5) + (2/3)(10) \approx 8.3 > 7.5`$.

The same logic applies to average family size (sampling a random person vs. a random family), average bus occupancy, and average class size.

---

## 4. Markov Chains: Definitions and State Classification

### 4.1 What Is a Markov Chain?

A **discrete-time finite-state Markov chain** is a sequence of random variables $`X_0, X_1, X_2, \ldots`$ taking values in a finite set $`\lbrace 1, 2, \ldots, m \rbrace`$ (the **state space**), with the property that the future depends on the past only through the present state.

The **transition probabilities** are:

$$p_{ij} = P(X_{n+1} = j \mid X_n = i)$$

The **Markov property** states that conditioning on the entire history gives the same result as conditioning on only the current state:

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i) = p_{ij}$$

The chain is **time-homogeneous**: the transition probabilities $`p_{ij}`$ do not depend on $`n`$. For each state $`i`$, the transition probabilities out of $`i`$ sum to 1: $`\sum_j p_{ij} = 1`$.

To specify a Markov chain model, you need to identify the states, the transitions (which states can be reached from which), and the transition probabilities.

**Example (checkout counter).** Customers arrive as Bernoulli($`p`$), service times are Geometric($`q`$), and $`X_n`$ is the number of customers at time $`n`$. The state transitions depend only on the current queue length, making this a Markov chain.

### 4.2 $`n`$-Step Transition Probabilities

The **$`n`$-step transition probability** from state $`i`$ to state $`j`$ is:

$$r_{ij}(n) = P(X_n = j \mid X_0 = i)$$

The base cases are $`r_{ij}(0) = 1`$ if $`i = j`$ and 0 otherwise, and $`r_{ij}(1) = p_{ij}`$.

The **key recursion** (Chapman-Kolmogorov equation) is:

$$r_{ij}(n) = \sum_{k=1}^{m} r_{ik}(n-1) \cdot p_{kj}$$

This says: to get from $`i`$ to $`j`$ in $`n`$ steps, go from $`i`$ to some intermediate state $`k`$ in $`n-1`$ steps, then from $`k`$ to $`j`$ in one step.

With a random initial state: $`P(X_n = j) = \sum_{i=1}^{m} P(X_0 = i) \cdot r_{ij}(n)`$.

### 4.3 Calculating Trajectory Probabilities

The probability of a specific trajectory uses the multiplication rule:

$$P(X_1 = j, X_2 = k, X_3 = l \mid X_0 = i) = p_{ij} \cdot p_{jk} \cdot p_{kl}$$

This follows directly from the Markov property: each transition depends only on the current state.

### 4.4 Recurrent and Transient States

State $`i`$ is **recurrent** if, starting from $`i`$, no matter where the chain goes, there is always a path back to $`i`$. If state $`i`$ is not recurrent, it is called **transient**. A transient state will eventually be left forever; the chain will settle into a recurrent class.

A **recurrent class** is a maximal collection of recurrent states that communicate only with each other. Once the chain enters a recurrent class, it stays there forever. A chain can have multiple recurrent classes and multiple transient states.

### 4.5 Periodicity

The states in a recurrent class are **periodic** with period $`d > 1`$ if they can be grouped into $`d`$ groups such that all transitions from one group lead to the next group (cyclically). For example, in a chain that alternates deterministically between two groups of states, $`d = 2`$.

A self-transition ($`p_{ii} > 0`$) at any state in the class guarantees the class is **aperiodic** ($`d = 1`$). Aperiodicity is required for the convergence theorem below.

---

## 5. Markov Chains: Steady-State Behavior and Birth-Death Processes

### 5.1 The Convergence Theorem

**Theorem.** If a Markov chain has a single recurrent class that is aperiodic (plus possibly some transient states), then the $`n`$-step transition probabilities converge to steady-state values that do not depend on the initial state:

$$\lim_{n \to \infty} r_{ij}(n) = \pi_j \quad \text{for all } i$$

The steady-state probabilities $`\pi_j`$ are the unique solution to the **balance equations** together with the normalization constraint:

$$\pi_j = \sum_{k} \pi_k \, p_{kj}, \qquad j = 1, \ldots, m$$

$$\sum_{j=1}^{m} \pi_j = 1$$

The balance equations say: the long-run frequency of being in state $`j`$ equals the sum of long-run frequencies of being in each state $`k`$ times the probability of transitioning from $`k`$ to $`j`$. In other words, "frequency of transitions into $`j`$" equals "frequency of being in $`j`$."

**When the conditions fail.** If the chain is periodic, $`r_{ij}(n)`$ oscillates and does not converge. If there are multiple recurrent classes, the limiting distribution depends on the initial state (which class the chain ends up in).

### 5.2 Visit Frequency Interpretation

The steady-state probability $`\pi_j`$ has a concrete frequency interpretation: in the long run, the fraction of time the chain spends in state $`j`$ converges to $`\pi_j`$. Similarly, the long-run frequency of transitions from $`i`$ to $`j`$ is $`\pi_i \cdot p_{ij}`$.

### 5.3 Example: Two-State Chain

Consider states $`\lbrace 1, 2 \rbrace`$ with $`p_{12} = 0.5`$, $`p_{11} = 0.5`$, $`p_{21} = 0.2`$, $`p_{22} = 0.8`$. This is a single aperiodic recurrent class (self-transitions exist). The balance equations are:

$$\pi_1 = \pi_1 \cdot 0.5 + \pi_2 \cdot 0.2, \qquad \pi_1 + \pi_2 = 1$$

From the first equation: $`\pi_1 \cdot 0.5 = \pi_2 \cdot 0.2`$, so $`\pi_1 = (2/5)\pi_2`$. Substituting into the normalization: $`\pi_2(2/5 + 1) = 1`$, giving $`\pi_2 = 5/7`$ and $`\pi_1 = 2/7`$.

### 5.4 Using Steady-State Probabilities for Long-Run Calculations

Once the chain has "settled in" (after many steps), the state at time $`n`$ is approximately distributed according to $`\pi`$. For example, with the two-state chain above starting at state 1:

$$P(X_1 = 1, X_{100} = 1 \mid X_0 = 1) \approx p_{11} \cdot \pi_1 = 0.5 \times \frac{2}{7}$$

$$P(X_{100} = 1, X_{200} = 1 \mid X_0 = 1) \approx \pi_1 \times \pi_1 = \left(\frac{2}{7}\right)^2$$

The convergence to steady state is typically exponential: $`r_{ij}(n) - \pi_j = O(c^n)`$ for some $`0 < c < 1`$.

### 5.5 Birth-Death Processes

A **birth-death process** is a Markov chain on states $`\lbrace 0, 1, \ldots, m \rbrace`$ where transitions only go to neighboring states: from state $`i`$, the chain moves to $`i + 1`$ with probability $`p_i`$ ("birth"), to $`i - 1`$ with probability $`q_i`$ ("death"), or stays at $`i`$ with probability $`1 - p_i - q_i`$.

The balance equations simplify to **local balance** (also called detailed balance):

$$\pi_i \, p_i = \pi_{i+1} \, q_{i+1}$$

This says: the frequency of transitions from $`i`$ to $`i+1`$ equals the frequency of transitions from $`i+1`$ to $`i`$. Solving recursively:

$$\pi_{i+1} = \pi_i \cdot \frac{p_i}{q_{i+1}}, \qquad i = 0, 1, \ldots$$

All $`\pi_i`$ are expressed in terms of $`\pi_0`$, and then $`\pi_0`$ is determined by the normalization $`\sum_j \pi_j = 1`$.

**Special case:** $`p_i = p`$ and $`q_i = q`$ for all $`i`$. Let $`\rho = p/q`$. Then $`\pi_i = \pi_0 \rho^i`$.

If $`p = q`$ (symmetric random walk on $`\lbrace 0, \ldots, m \rbrace`$): $`\pi_i = \pi_0`$ for all $`i`$, so $`\pi_i = 1/(m+1)`$ (uniform distribution).

If $`p < q`$ and $`m \to \infty`$ (infinite state space, more deaths than births): the geometric series converges, giving $`\pi_0 = 1 - \rho`$, $`\pi_i = (1 - \rho)\rho^i`$, and the steady-state mean is $`E[X_n] = \rho/(1 - \rho)`$.

### 5.6 Application: Phone System Design (Erlang's Model)

Calls arrive as a Poisson process with rate $`\lambda`$, each call has exponential duration with parameter $`\mu`$, and there are $`B`$ lines. The number of active calls is a birth-death process on $`\lbrace 0, 1, \ldots, B \rbrace`$ with birth rate $`\lambda`$ and death rate $`i\mu`$ from state $`i`$ (because $`i`$ calls are each independently ending). In the discrete-time approximation:

$$\lambda \pi_{i-1} = i\mu \pi_i \implies \pi_i = \pi_0 \frac{\lambda^i}{\mu^i \, i!}$$

The probability a customer finds the system busy is $`\pi_B`$. To achieve $`\pi_B \le 1\%`$, one chooses $`B`$ accordingly. This is the **Erlang B formula**, foundational in telecommunications engineering.

---

## 6. Markov Chains: Absorption, First Passage, and Applications

### 6.1 Absorption Probabilities

An **absorbing state** is a recurrent state $`k`$ with $`p_{kk} = 1`$ (once entered, it is never left). The key question: what is the probability $`a_i`$ that the chain eventually reaches absorbing state $`s`$, given that it started in state $`i`$?

**Boundary conditions:** $`a_s = 1`$ (if starting at $`s`$, already absorbed), and $`a_k = 0`$ for any other absorbing state $`k \ne s`$.

**For all other (transient) states:** conditioning on the first step:

$$a_i = \sum_{j=1}^{m} p_{ij} \, a_j$$

This system of linear equations has a unique solution. The logic: from state $`i`$, you take one step to some state $`j`$ (with probability $`p_{ij}`$), and from $`j`$ the absorption probability is $`a_j`$.

### 6.2 Expected Time to Absorption

Let $`\mu_i`$ be the expected number of steps to reach absorbing state $`s`$, starting from state $`i`$.

**Boundary condition:** $`\mu_s = 0`$.

**For all other states:**

$$\mu_i = 1 + \sum_{j} p_{ij} \, \mu_j$$

The "1" accounts for the current step, and then we average over where that step leads. This system also has a unique solution.

### 6.3 Mean First Passage and Recurrence Times

For a chain with a single recurrent class, fix a recurrent state $`s`$.

The **mean first passage time** from $`i`$ to $`s`$ is $`t_i = E[\min\lbrace n \ge 0 : X_n = s \rbrace \mid X_0 = i]`$. It satisfies:

$$t_s = 0, \qquad t_i = 1 + \sum_j p_{ij} \, t_j \quad \text{for all } i \ne s$$

The **mean recurrence time** of state $`s`$ is the expected time to return to $`s`$ starting from $`s`$:

$$t_s^* = E[\min\lbrace n \ge 1 : X_n = s \rbrace \mid X_0 = s] = 1 + \sum_j p_{sj} \, t_j$$

A beautiful connection: $`t_s^* = 1/\pi_s`$. States visited more frequently in steady state have shorter recurrence times.

### 6.4 The Gambler's Ruin Problem

A gambler starts with $`i`$ dollars and bets \$1 each round in a fair game ($`p = 0.5`$), stopping upon reaching 0 (ruin) or $`n`$ (victory). The states are $`\lbrace 0, 1, \ldots, n \rbrace`$ with absorbing states 0 and $`n`$.

**Absorption probability.** In the fair case ($`p = 0.5`$):

$$a_i = P(\text{reach } n \mid \text{start at } i) = \frac{i}{n}$$

This follows from the equation $`a_i = 0.5 \, a_{i+1} + 0.5 \, a_{i-1}`$ with $`a_0 = 0`$, $`a_n = 1`$.

The expected wealth at the end is $`0 \cdot (1 - a_i) + n \cdot a_i = n \cdot i/n = i`$. The game is a **martingale**: the expected final wealth equals the starting wealth.

**Expected duration.** In the fair case:

$$\mu_i = i(n - i)$$

Starting at $`i = 1`$ with $`n = 100`$: the expected game length is $`1 \times 99 = 99`$ steps. Starting at $`i = 50`$: $`50 \times 50 = 2500`$ steps.

**Unfair game** ($`p \ne 0.5`$). Let $`r = (1-p)/p`$. Then:

$$a_i = \frac{1 - r^i}{1 - r^n}$$

If $`p < 0.5`$ (unfavorable odds), $`r > 1`$ and $`a_i`$ decays rapidly with $`n`$. With $`p = 0.49`$, $`i = 50`$, and $`n = 100`$: $`a_{50} \approx 0.12`$.

---

## 7. General Random Processes and Signal Processing Foundations

### 7.1 Random Walks

A **general random walk** is defined by $`X[0] = 0`$ and $`X[n] = X[n-1] + Z[n]`$ for $`n \ge 1`$, where $`Z[n]`$ is a white noise sequence (i.i.d. with zero mean). The random walk is the cumulative sum of the noise terms: $`X[n] = Z[1] + Z[2] + \cdots + Z[n]`$. It has $`E[X[n]] = 0`$, and $`\text{var}(X[n]) = n \cdot \text{var}(Z)`$, growing linearly with time.

### 7.2 Brownian Motion

**Brownian motion** $`\lbrace B(t) : t \ge 0 \rbrace`$ is the continuous-time limit of a scaled random walk. With diffusion parameter $`d`$, the increment $`B(t_1) - B(t_0)`$ is normally distributed:

$$B(t_1) - B(t_0) \sim N\left(0, \sqrt{d(t_1 - t_0)}\right)$$

Increments over disjoint intervals are independent. Brownian motion is the foundational process in continuous-time stochastic modeling, appearing in physics (diffusion), finance (stock prices), and many other fields.

### 7.3 Mean and Variance Functions

For any random process $`\lbrace X(t) \rbrace`$ (continuous or discrete time):

The **mean function** is $`\mu_X(t) = E[X(t)]`$, describing how the average value evolves.

The **variance function** is $`V(t) = \text{var}(X(t))`$, describing how the spread evolves.

The **autocovariance function** captures the dependence between the process at two different times:

$$C_X(s, t) = \text{cov}(X(s), X(t)) = E[X(s) \cdot X(t)] - \mu_X(s) \cdot \mu_X(t)$$

Note that $`C_X(t, t) = V(t)`$.

### 7.4 Autocorrelation Function

The **autocorrelation function** is:

$$R_X(s, t) = E[X(s) \cdot X(t)]$$

It relates to the autocovariance by $`R_X(s, t) = C_X(s, t) + \mu_X(s) \cdot \mu_X(t)`$.

For a stationary process (see below), $`R_X`$ depends only on the time difference $`\tau = s - t`$: $`R_X(\tau) = E[X(t) \cdot X(t + \tau)]`$. The expected power of the process is $`E[X(t)^2] = R_X(0)`$.

### 7.5 Stationarity

A process is **strictly stationary** if the joint distribution of $`(X(t_1), \ldots, X(t_n))`$ is the same as that of $`(X(t_1 + \tau), \ldots, X(t_n + \tau))`$ for all shifts $`\tau`$. All statistical properties are time-invariant.

A process is **wide-sense stationary (WSS)** if: the mean is constant ($`\mu_X(t) = \mu_X`$ for all $`t`$), and the autocovariance depends only on the lag ($`C_X(s, t) = C_X(s - t)`$).

WSS is a weaker condition than strict stationarity. It is sufficient for many signal processing applications because it guarantees that second-order statistics (mean, variance, covariance) do not change with time.

### 7.6 Linear Time-Invariant (LTI) Filters

An **LTI system** $`\mathcal{L}`$ satisfies two properties: **linearity** ($`\mathcal{L}[a_1 x_1 + a_2 x_2] = a_1 \mathcal{L}[x_1] + a_2 \mathcal{L}[x_2]`$) and **time-invariance** ($`\mathcal{L}[x(t - \tau)](t) = \mathcal{L}[x](t - \tau)`$).

An LTI system is completely characterized by its **impulse response** $`h(t)`$ (or $`h[n]`$ in discrete time), defined as $`h = \mathcal{L}[\delta]`$, the output when the input is an impulse.

The output for a general input is given by **convolution**:

$$y(t) = (h * x)(t) = \int_{-\infty}^{\infty} h(\tau) \cdot x(t - \tau) \, d\tau$$

In discrete time: $`Y[n] = \sum_{k=-\infty}^{\infty} h[k] \cdot X[n-k]`$.

### 7.7 LTI Systems with Stationary Inputs

When a WSS process $`X`$ is passed through an LTI filter with impulse response $`h`$, the output $`Y`$ is also WSS, with:

$$\mu_Y = \mu_X \cdot \int_{-\infty}^{\infty} h(t) \, dt$$

$$C_Y(\tau) = (h(-t) * h * C_X)(\tau)$$

### 7.8 Autoregressive Processes

A **first-order autoregressive (AR(1))** process is defined by:

$$X[n] = a_1 \, X[n-1] + Z[n]$$

where $`Z[n]`$ is white noise with variance $`\sigma^2`$ and $`|a_1| < 1`$ for stability. The impulse response is $`h[n] = a_1^n \, u[n]`$ (where $`u[n]`$ is the unit step). The autocovariance is:

$$C_X[k] = \sigma^2 \frac{a_1^{|k|}}{1 - a_1^2}$$

The autocovariance decays exponentially with lag, meaning the process has short-range memory.

### 7.9 Power Spectral Density (PSD)

The **power spectral density** $`S_X(f)`$ is the Fourier transform of the autocorrelation function:

$$R_X(\tau) \xleftrightarrow{\mathcal{F}} S_X(f)$$

The PSD describes how the power of the process is distributed across frequencies. Key properties: $`S_X(f) \ge 0`$ for all $`f`$, and the total power satisfies:

$$\int_{-\infty}^{\infty} S_X(f) \, df = R_X(0) = E[X(t)^2]$$

When a WSS process passes through an LTI filter:

$$S_Y(f) = |H(f)|^2 \cdot S_X(f)$$

where $`H(f)`$ is the frequency response (Fourier transform of $`h(t)`$). This fundamental relation says the filter shapes the power spectrum of the input by multiplying it by the squared magnitude of the frequency response.

---

## Quick Reference: Key Formulas

**Bernoulli process -- $`k`$th arrival time (Pascal):**\
$$p_{Y_k}(t) = \binom{t-1}{k-1} p^k(1-p)^{t-k}, \quad E[Y_k] = k/p$$

**Poisson PMF (arrivals in interval $`\tau`$):**\
$$P(k, \tau) = \frac{(\lambda\tau)^k e^{-\lambda\tau}}{k!}, \quad E[N_\tau] = \lambda\tau, \quad \text{var}(N_\tau) = \lambda\tau$$

**Erlang PDF ($`k`$th arrival in Poisson process):**\
$$f_{Y_k}(y) = \frac{\lambda^k y^{k-1} e^{-\lambda y}}{(k-1)!}, \quad E[Y_k] = k/\lambda$$

**Merging Poisson processes:**\
$$\text{Poisson}(\lambda_1) + \text{Poisson}(\lambda_2) = \text{Poisson}(\lambda_1 + \lambda_2)$$

**Competing exponentials:**\
$$\min(X_1, \ldots, X_n) \sim \text{Exp}(\lambda_1 + \cdots + \lambda_n)$$

**Markov chain balance equations:**\
$$\pi_j = \sum_k \pi_k \, p_{kj}, \quad \sum_j \pi_j = 1$$

**Birth-death local balance:**\
$$\pi_i \, p_i = \pi_{i+1} \, q_{i+1}$$

**Absorption probabilities:**\
$$a_i = \sum_j p_{ij} \, a_j \quad \text{(with boundary conditions at absorbing states)}$$

**Expected time to absorption:**\
$$\mu_i = 1 + \sum_j p_{ij} \, \mu_j$$

**Mean recurrence time:**\
$$t_s^* = 1/\pi_s$$

**Gambler's ruin (fair game):**\
$$a_i = i/n, \qquad \mu_i = i(n - i)$$

**LTI filter output spectrum:**\
$$S_Y(f) = |H(f)|^2 \cdot S_X(f)$$

---

*End of Part 3: Random Processes. [Part 1: The Fundamentals](./Probability_Part1_Fundamentals.md) | [Part 2: Inference and Limit Theorems](./Probability_Part2_Inference_and_Limit_Theorems.md)*
