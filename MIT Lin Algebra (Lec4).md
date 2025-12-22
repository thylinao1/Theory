# MIT 18.642 — Stochastic Matrices & Financial Modeling Notes

## 1. Stochastic Matrices

*A matrix governing probability transitions; columns represent how probability mass flows from one state to all possible next states.*

**Definition:** Matrix A = (aᵢⱼ) is stochastic if:
- aᵢⱼ ≥ 0 for all i, j (non-negative entries)
- Σᵢ aᵢⱼ = 1 (each column sums to one)

**Interpretation:** Column j gives conditional probabilities P(next state = i | current state = j)

---

## 2. Markov Chain Model

*A memoryless stochastic process where the future state depends only on the current state, not the history; foundation for modeling sequential random phenomena.*

**Components:**
- States: s = 1, 2, ..., m
- State at time t: Sₜ
- Transition probabilities: aᵢⱼ = P(Sₜ₊₁ = i | Sₜ = j)
- Probability vector: π(t) ∈ ℝᵐ where πⱼ(t) = P(Sₜ = j)

**Dynamics:** `π(t+1) = A·π(t)` — law of total probability in matrix form

---

## 3. Markov Chain Dynamics & Stationary Distribution

*The long-run behavior of a Markov chain; stationary distribution represents equilibrium probabilities that no longer change under transitions.*

**Iteration:** π(t) = Aᵗ·π(0)

**Stationary Distribution π*:**
- Satisfies: `A·π* = π*` (eigenvector of A with eigenvalue 1)
- Represents long-run state probabilities

**Existence & Uniqueness:** Guaranteed if chain is:
- **Irreducible:** Can reach any state from any other state
- **Aperiodic:** No fixed cycling pattern

---

## 4. Single-Period Market Model

*Simplest model of uncertainty in finance; one decision point today, one resolution of uncertainty tomorrow across discrete scenarios.*

| Symbol | Meaning |
|--------|---------|
| T | Future time (single period) |
| m | Number of possible states ω₁, ..., ωₘ |
| n | Number of traded assets |
| A₀ʲ | Price of asset j today |
| Aₜʲ | Payoff vector of asset j at time T |

---

## 5. Portfolios and Contingent Claims

*A portfolio combines assets; a contingent claim is any payoff that depends on which state occurs—the building block of derivatives.*

**Portfolio q ∈ ℝⁿ:**
- Cost today: V₀ = Σⱼ qⱼ·A₀ʲ
- Payoff at T: Vₜ = Σⱼ qⱼ·Aₜʲ

**Contingent Claim:** Any payoff vector Cₜ ∈ ℝᵐ (one value per state)

---

## 6. No-Arbitrage Principle

*The fundamental assumption that markets don't offer risk-free profits; if violated, traders would exploit it until prices adjust.*

**Arbitrage exists if:**
- V₀ ≤ 0 (zero or negative cost)
- Vₜ(ωᵢ) ≥ 0 for all i (non-negative payoff in all states)
- Vₜ(ωᵢ) > 0 for at least one i (strictly positive in some state)

**No-arbitrage** rules out "free lunches" and constrains asset prices.

---

## 7. Pricing Measure (Risk-Neutral Measure)

*A mathematical probability measure Q* under which all assets are priced as discounted expected payoffs; not real-world probabilities, but prices of state-contingent claims.*

**Existence:** No arbitrage ⟹ ∃ pricing measure Q* with qᵢ* > 0

**Asset Pricing Formula:**
```
A₀ʲ = α · E_Q*[Aₜʲ] = α · Σᵢ Aₜʲ(ωᵢ)·qᵢ*
```
- α > 0 is the discount factor
- Q* ≠ real-world probability measure P

---

## 8. Interpretation of Pricing Probabilities

*Pricing weights reflect economic value of payoffs in each state, not likelihood; "bad" states (recessions, crashes) carry higher pricing weight because investors pay more to be insured against them.*

- qᵢ* = price today of 1 unit of payoff in state ωᵢ
- Bad states → higher pricing weight (risk premium)
- **Normalization:** Σᵢ qᵢ* = 1 ensures risk-free asset priced correctly

---

## 9. Market Completeness

*A market where any desired payoff pattern can be constructed from available assets; allows perfect hedging and unique pricing.*

**Complete Market:** Every contingent claim Cₜ ∈ ℝᵐ can be replicated by some portfolio

**Equivalent Conditions:**
- Payoff matrix has full row rank (m assets span all states)
- Pricing measure Q* is **unique**

**Incomplete Markets:** Multiple valid pricing measures exist → price bounds rather than unique prices

---

## 10. Volatility Estimation

*Volatility (σ) is unobservable and must be inferred from price data; choice of estimator affects accuracy and robustness.*

| Estimator | Characteristics |
|-----------|-----------------|
| Close-to-Close | Simple, uses only closing prices |
| Parkinson | Uses high-low range |
| Garman-Klass | Uses OHLC, assumes no drift |
| Rogers-Satchell | Handles drift |
| **Yang-Zhang** | Robust to drift, handles overnight gaps, lowest variance |

---

## 11. Time-Series Modeling of Volatility

*Volatility clusters (high vol follows high vol) and persists; ARIMA models capture these autocorrelation patterns for forecasting.*

**ARIMA(p, d, q):**
- p = autoregressive terms (past values)
- d = differencing order (stationarity)
- q = moving-average terms (past shocks)

**Volatility Properties:**
- Persistence: shocks decay slowly
- Clustering: calm and turbulent periods bunch together

---

## 12. ACF-Based Model Identification

*The autocorrelation function (ACF) pattern reveals appropriate model structure; slow decay suggests AR components, sharp cutoff suggests MA.*

| ACF Pattern | Model Implication |
|-------------|-------------------|
| Slow decay | AR or ARMA structure |
| Sharp cutoff at lag q | Pure MA(q) |
| No sharp cutoff | Not pure MA |

**Examples from slides:**
- Close-to-close volatility → ARIMA(3,1,0)
- Yang-Zhang volatility → ARIMA(1,1,1)

---

## 13. Seasonal ARIMA (SARIMA)

*Extends ARIMA to capture periodic patterns; rolling window calculations induce artificial seasonality at the window length.*

**Model:** ARIMA(p, d, q)(P, D, Q)ₛ

**Example:** Rolling 21-day volatility → correlation at lag 21 → use seasonal component with s = 21

**Key Insight:** Seasonality often comes from **data construction** (rolling windows), not underlying economics

---

## Key Takeaways

| Concept | Role |
|---------|------|
| Stochastic matrices | Govern probability evolution |
| Pricing measures | Govern asset valuation |
| No-arbitrage | Links payoffs to prices |
| Volatility estimation | Quality affects all downstream models |
| Seasonal structure | Often artifact of methodology, not economics |
