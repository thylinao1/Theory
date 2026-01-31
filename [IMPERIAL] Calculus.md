# Mathematics for Machine Learning: Multivariate Calculus

> **Complete Course Notes — Extended Edition**  
> A comprehensive and detailed guide to calculus concepts essential for machine learning, covering differentiation, multivariate systems, optimization, neural networks, Taylor series, and regression with full derivations, worked examples, and intuitive explanations.

---

## Table of Contents

1. [Module 1: Foundations of Calculus](#module-1-foundations-of-calculus)
   - [1.1 What Are Functions?](#11-what-are-functions)
   - [1.2 Gradients and Derivatives — The Core Idea](#12-gradients-and-derivatives--the-core-idea)
   - [1.3 The Formal Definition of a Derivative](#13-the-formal-definition-of-a-derivative)
   - [1.4 The Sum Rule](#14-the-sum-rule)
   - [1.5 The Power Rule](#15-the-power-rule)
   - [1.6 Special Functions and Their Derivatives](#16-special-functions-and-their-derivatives)
   - [1.7 The Product Rule](#17-the-product-rule)
   - [1.8 The Chain Rule](#18-the-chain-rule)
   - [1.9 Comprehensive Worked Examples](#19-comprehensive-worked-examples)
2. [Module 2: Multivariate Calculus](#module-2-multivariate-calculus)
   - [2.1 Variables, Constants, and Parameters](#21-variables-constants-and-parameters)
   - [2.2 Partial Differentiation](#22-partial-differentiation)
   - [2.3 The Total Derivative](#23-the-total-derivative)
   - [2.4 The Jacobian Vector](#24-the-jacobian-vector)
   - [2.5 The Jacobian Matrix](#25-the-jacobian-matrix)
   - [2.6 The Hessian Matrix](#26-the-hessian-matrix)
   - [2.7 Numerical Differentiation](#27-numerical-differentiation)
3. [Module 3: Optimization and Neural Networks](#module-3-optimization-and-neural-networks)
   - [3.1 Introduction to Optimization](#31-introduction-to-optimization)
   - [3.2 Neural Network Architecture](#32-neural-network-architecture)
   - [3.3 Backpropagation — The Chain Rule in Action](#33-backpropagation--the-chain-rule-in-action)
4. [Module 4: Taylor Series and Approximations](#module-4-taylor-series-and-approximations)
   - [4.1 Why We Need Approximations](#41-why-we-need-approximations)
   - [4.2 Power Series — Building Intuition](#42-power-series--building-intuition)
   - [4.3 The Maclaurin Series](#43-the-maclaurin-series)
   - [4.4 The Taylor Series](#44-the-taylor-series)
   - [4.5 Linearization and Error Analysis](#45-linearization-and-error-analysis)
   - [4.6 Multivariate Taylor Series](#46-multivariate-taylor-series)
5. [Module 5: Gradient-Based Optimization](#module-5-gradient-based-optimization)
   - [5.1 The Newton-Raphson Method](#51-the-newton-raphson-method)
   - [5.2 The Gradient Vector](#52-the-gradient-vector)
   - [5.3 Gradient Descent](#53-gradient-descent)
   - [5.4 Lagrange Multipliers](#54-lagrange-multipliers)
6. [Module 6: Regression and Least Squares Fitting](#module-6-regression-and-least-squares-fitting)
   - [6.1 Introduction to Data Fitting](#61-introduction-to-data-fitting)
   - [6.2 Linear Regression — Complete Derivation](#62-linear-regression--complete-derivation)
   - [6.3 Generalized Nonlinear Least Squares](#63-generalized-nonlinear-least-squares)
   - [6.4 Practical Implementation and Advanced Methods](#64-practical-implementation-and-advanced-methods)
7. [Complete Reference Tables](#complete-reference-tables)

---

## Module 1: Foundations of Calculus

This module establishes the fundamental theory of calculus, building from the intuitive concept of "slope" to the formal definition of derivatives and the powerful rules that make differentiation practical. By the end, you'll have a complete toolkit for differentiating virtually any function you encounter.

---

### 1.1 What Are Functions?

#### The Basic Concept

A **function** is a mathematical relationship that maps inputs to outputs. For every valid input (or combination of inputs), a function produces exactly one output. Functions are the mathematical language we use to describe how quantities relate to each other in the real world.

**Formal Definition:** A function f: A → B is a rule that assigns to each element x in set A (the domain) exactly one element f(x) in set B (the codomain).

#### Example: Modeling Temperature

Consider modeling temperature distribution in a room. We might define:

$$
T = f(x, y, z, t)
$$

This function takes four inputs:
- x, y, z — spatial coordinates (where in the room)
- t — time (when we're measuring)

And returns one output:
- T — temperature at that location and time

The function f encapsulates all the complex physics of heat transfer, air circulation, and thermal properties of materials into a single mathematical object.

#### Understanding Function Notation

The notation f(x) = x² + 3 means "f is a function of x, and when you input x, you get x² + 3 as output."

**Important Clarification:** The expression f(x) denotes "function f applied to input x" — it is NOT multiplication of f times x. This can be confusing because:
- In algebra, a(b) might mean a × b
- In function notation, f(x) means "evaluate function f at point x"

**Context Matters:** When you see an expression like f(x) = g(x)/(h(x) · a(x)), you need context to determine:
- Is g a function being applied to x?
- Is h a function or a constant?
- Is a a function or just a parameter?

This ambiguity is resolved through mathematical convention and context — something you'll develop intuition for with practice.

#### Types of Functions

**Univariate Functions:** Single input, single output

$$
f(x) = x^2 + 3x - 5
$$

**Multivariate Functions:** Multiple inputs, single output

$$
f(x, y) = x^2 + xy + y^2
$$

**Vector-Valued Functions:** Single or multiple inputs, multiple outputs

$$
\mathbf{f}(t) = \begin{pmatrix} \cos(t) \\\\ \sin(t) \end{pmatrix}
$$

#### Why Functions Matter for Machine Learning

Machine learning models are, at their core, functions. Consider:

**Linear Regression:**

$$
f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b
$$

**Neural Network:**

$$
f(\mathbf{x}) = \sigma(W_n \cdot \sigma(W_{n-1} \cdot \ldots \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \ldots + \mathbf{b}_{n-1}) + \mathbf{b}_n)
$$

**The Central Question:** Given a function (model), how do we find the parameters that make it best fit our data? Calculus provides the tools to answer this by analyzing how the function's output changes when we adjust its parameters.

---

### 1.2 Gradients and Derivatives — The Core Idea

#### The Fundamental Concept

**Calculus is the study of change.** Specifically, it asks: "How does a function's output change when we change its input?"

The **derivative** quantifies this rate of change. If you know the derivative at a point, you know:
1. **Direction:** Is the function increasing or decreasing?
2. **Magnitude:** How fast is it changing?

#### Visualizing with a Speed-Time Graph

Consider a car's speed recorded over time:

```
Speed (v)
    │        ____
    │       /    \
    │      /      \
    │     /        \
    │    /          \
    │   /            \____
    │  /
    │_/
    └────────────────────── Time (t)
         A    B    C    D
```

**Reading the Graph:**

At point **A**: The line slopes upward → Speed is increasing → Car is accelerating

At point **B**: The line is at a peak → Speed is momentarily constant → Acceleration is zero

At point **C**: The line slopes downward → Speed is decreasing → Car is decelerating

At point **D**: The line is flat → Speed is constant → No acceleration

#### Gradient = Rate of Change

The **gradient** (slope) at any point tells us the instantaneous rate of change:

$$
\text{Gradient} = \frac{\text{Rise}}{\text{Run}} = \frac{\Delta v}{\Delta t} = \frac{\text{Change in speed}}{\text{Change in time}}
$$

For our speed-time graph, this gradient IS the acceleration!

**Key Insight:** The gradient of speed with respect to time equals acceleration. This isn't a coincidence — it's the definition of acceleration.

#### The Tangent Line

At any point on a curve, we can draw a **tangent line** — a straight line that:
1. Touches the curve at exactly that point
2. Has the same slope as the curve at that point

The tangent line represents the "best linear approximation" to the curve near that point. This concept is fundamental to calculus.

#### From Gradient to Derivative Function

Here's the powerful insight: Instead of calculating the gradient at one point, what if we calculated it at EVERY point?

The result is a new function — the **derivative function** — that tells us the gradient at any point we choose.

**Example:**
- Original function: Speed vs. Time
- Derivative function: Acceleration vs. Time

We can continue this process:
- Derivative of position → Velocity
- Derivative of velocity → Acceleration
- Derivative of acceleration → Jerk (rate of change of acceleration)

#### The Anti-Derivative (Preview of Integration)

If differentiation finds the gradient, what about the reverse operation?

The **anti-derivative** (or **integral**) asks: "What function would have THIS function as its derivative?"

For our car example:
- Anti-derivative of speed → Distance traveled

This makes intuitive sense: if you travel at 60 mph for 2 hours, you've covered 120 miles. The area under the speed-time curve gives the total distance — and calculating areas under curves is exactly what integration does.

---

### 1.3 The Formal Definition of a Derivative

Now we translate our intuitive understanding into precise mathematics.

#### Rise Over Run for Linear Functions

For a straight line, the gradient is the same everywhere:

$$
\text{Gradient} = \frac{\text{Rise}}{\text{Run}} = \frac{y_2 - y_1}{x_2 - x_1} = \frac{\Delta y}{\Delta x}
$$

Pick any two points, calculate rise/run, and you get the same answer.

#### The Challenge with Curves

For nonlinear functions, the gradient varies from point to point. We need a way to find the gradient at a single specific point.

**Strategy:** 
1. Pick a point x where we want to know the gradient
2. Pick a nearby point x + Δx
3. Calculate rise/run between these two points
4. Make Δx smaller and smaller
5. See what value the gradient approaches

#### Building the Formula

Let's formalize this. We have:
- Point 1: (x, f(x))
- Point 2: (x + Δx, f(x + Δx))

The rise (vertical change):

$$
\text{Rise} = f(x + \Delta x) - f(x)
$$

The run (horizontal change):

$$
\text{Run} = (x + \Delta x) - x = \Delta x
$$

The approximate gradient:

$$
\text{Approximate Gradient} = \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

#### The Limit — Making It Exact

As Δx gets smaller, our approximation gets better. The **exact** gradient is what we get when Δx approaches zero:

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

This is the **formal definition of the derivative**.

**Notation Variants:**

| Notation | Name | When Used |
|----------|------|-----------|
| f'(x) | Lagrange (prime) | General use, compact |
| df/dx | Leibniz | Emphasizes "with respect to x" |
| (d/dx)f(x) | Operator form | When applying to expressions |
| ḟ | Newton (dot) | Physics, when variable is time |

#### Complete Worked Example: Differentiating f(x) = 3x + 2

**Step 1:** Write the definition

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**Step 2:** Compute f(x + Δx)

$$
f(x + \Delta x) = 3(x + \Delta x) + 2 = 3x + 3\Delta x + 2
$$

**Step 3:** Compute the numerator f(x + Δx) - f(x)

$$
(3x + 3\Delta x + 2) - (3x + 2) = 3\Delta x
$$

**Step 4:** Form the ratio

$$
\frac{f(x + \Delta x) - f(x)}{\Delta x} = \frac{3\Delta x}{\Delta x} = 3
$$

**Step 5:** Take the limit

$$
f'(x) = \lim_{\Delta x \to 0} 3 = 3
$$

**Result:** The derivative of f(x) = 3x + 2 is f'(x) = 3.

**Interpretation:** A linear function has constant slope. The "2" is just a vertical shift that doesn't affect the slope.

#### Complete Worked Example: Differentiating f(x) = x²

**Step 1:** Write the definition

$$
f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**Step 2:** Compute f(x + Δx)

$$
f(x + \Delta x) = (x + \Delta x)^2 = x^2 + 2x\Delta x + (\Delta x)^2
$$

**Step 3:** Compute the numerator

$$
(x^2 + 2x\Delta x + (\Delta x)^2) - x^2 = 2x\Delta x + (\Delta x)^2
$$

**Step 4:** Form the ratio

$$
\frac{2x\Delta x + (\Delta x)^2}{\Delta x} = 2x + \Delta x
$$

**Step 5:** Take the limit

$$
f'(x) = \lim_{\Delta x \to 0} (2x + \Delta x) = 2x
$$

**Result:** The derivative of f(x) = x² is f'(x) = 2x.

**Interpretation:** 
- At x = 0: slope is 0 (the bottom of the parabola)
- At x = 1: slope is 2 (rising)
- At x = -1: slope is -2 (falling)
- At x = 3: slope is 6 (rising steeply)

---

### 1.4 The Sum Rule

When differentiating a sum of functions, we can differentiate each part separately.

#### The Rule

$$
\frac{d}{dx}[f(x) + g(x)] = \frac{df}{dx} + \frac{dg}{dx}
$$

Or more compactly:

$$
(f + g)' = f' + g'
$$

#### Why It Works — Proof from First Principles

Let h(x) = f(x) + g(x). Then:

$$
h'(x) = \lim_{\Delta x \to 0} \frac{h(x + \Delta x) - h(x)}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{[f(x + \Delta x) + g(x + \Delta x)] - [f(x) + g(x)]}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{[f(x + \Delta x) - f(x)] + [g(x + \Delta x) - g(x)]}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} + \lim_{\Delta x \to 0} \frac{g(x + \Delta x) - g(x)}{\Delta x}
$$

$$
= f'(x) + g'(x)
$$

**Key Step:** The limit of a sum equals the sum of the limits (for well-behaved functions).

#### The Difference Rule (Corollary)

The same logic applies to subtraction:

$$
\frac{d}{dx}[f(x) - g(x)] = \frac{df}{dx} - \frac{dg}{dx}
$$

#### The Constant Multiple Rule (Corollary)

For any constant c:

$$
\frac{d}{dx}[c \cdot f(x)] = c \cdot \frac{df}{dx}
$$

**Proof:** 

$$
\frac{d}{dx}[cf(x)] = \lim_{\Delta x \to 0} \frac{cf(x + \Delta x) - cf(x)}{\Delta x} = c \cdot \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} = c \cdot f'(x)
$$

#### Example

Differentiate f(x) = 3x² + 5x - 7:

$$
f'(x) = \frac{d}{dx}[3x^2] + \frac{d}{dx}[5x] + \frac{d}{dx}[-7]
$$

$$
= 3 \cdot \frac{d}{dx}[x^2] + 5 \cdot \frac{d}{dx}[x] + 0
$$

(We'll complete this once we have the power rule.)

---

### 1.5 The Power Rule

The power rule is the workhorse of differentiation, handling all polynomial terms.

#### The Rule

For any real number n:

$$
\frac{d}{dx}[x^n] = nx^{n-1}
$$

With a coefficient a:

$$
\frac{d}{dx}[ax^n] = anx^{n-1}
$$

#### Intuition

- **Multiply by the power:** The exponent comes down as a coefficient
- **Reduce the power by 1:** The new exponent is one less than before

#### Complete Proof for Positive Integer n

We'll prove this using the binomial theorem. For f(x) = xⁿ:

$$
f'(x) = \lim_{\Delta x \to 0} \frac{(x + \Delta x)^n - x^n}{\Delta x}
$$

Using the binomial expansion:

$$
(x + \Delta x)^n = x^n + nx^{n-1}\Delta x + \frac{n(n-1)}{2}x^{n-2}(\Delta x)^2 + \ldots + (\Delta x)^n
$$

Subtracting xⁿ:

$$
(x + \Delta x)^n - x^n = nx^{n-1}\Delta x + \frac{n(n-1)}{2}x^{n-2}(\Delta x)^2 + \ldots + (\Delta x)^n
$$

Dividing by Δx:

$$
\frac{(x + \Delta x)^n - x^n}{\Delta x} = nx^{n-1} + \frac{n(n-1)}{2}x^{n-2}\Delta x + \ldots + (\Delta x)^{n-1}
$$

Taking the limit as Δx → 0:

All terms with Δx vanish, leaving:

$$
f'(x) = nx^{n-1}
$$

#### Examples with Positive Integer Powers

| Function | Derivative | Calculation |
|----------|------------|-------------|
| x¹ | 1 · x⁰ = 1 | Power 1 comes down, new power is 0 |
| x² | 2x¹ = 2x | Power 2 comes down, new power is 1 |
| x³ | 3x² | Power 3 comes down, new power is 2 |
| x¹⁰ | 10x⁹ | Power 10 comes down, new power is 9 |
| 5x⁴ | 5 · 4x³ = 20x³ | Coefficient multiplies the result |

#### The Rule Works for ALL Real Powers

The power rule extends beyond positive integers:

**Negative Powers:**

$$
\frac{d}{dx}[x^{-1}] = -1 \cdot x^{-2} = -\frac{1}{x^2}
$$

$$
\frac{d}{dx}[x^{-3}] = -3x^{-4} = -\frac{3}{x^4}
$$

**Fractional Powers:**

$$
\frac{d}{dx}[x^{1/2}] = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}
$$

$$
\frac{d}{dx}[x^{2/3}] = \frac{2}{3}x^{-1/3} = \frac{2}{3\sqrt[3]{x}}
$$

**Irrational Powers:**

$$
\frac{d}{dx}[x^{\pi}] = \pi x^{\pi - 1}
$$

#### Special Case: Constants

A constant c can be written as c · x⁰:

$$
\frac{d}{dx}[c] = \frac{d}{dx}[c \cdot x^0] = c \cdot 0 \cdot x^{-1} = 0
$$

**The derivative of any constant is zero.** This makes sense — a constant doesn't change, so its rate of change is zero.

#### Complete Example: Polynomial Differentiation

Differentiate f(x) = 4x⁵ - 3x³ + 7x² - 2x + 9:

Using sum rule and power rule:

$$
f'(x) = 4(5x^4) - 3(3x^2) + 7(2x) - 2(1) + 0
$$

$$
f'(x) = 20x^4 - 9x^2 + 14x - 2
$$

---

### 1.6 Special Functions and Their Derivatives

Some functions have remarkable differentiation properties that make them essential tools in calculus and machine learning.

#### The Reciprocal Function: f(x) = 1/x

**Graph Shape:**
```
    │
    │\
    │ \
    │  \_____
────┼─────────
    │     ___
    │    /
    │   /
    │  /
```

The function has a **discontinuity** at x = 0 — it's undefined there and approaches ±∞ as x → 0.

**Derivative:**

$$
\frac{d}{dx}\left[\frac{1}{x}\right] = -\frac{1}{x^2}
$$

**Complete Derivation:**

$$
f'(x) = \lim_{\Delta x \to 0} \frac{\frac{1}{x + \Delta x} - \frac{1}{x}}{\Delta x}
$$

Find common denominator for the numerator:

$$
= \lim_{\Delta x \to 0} \frac{\frac{x - (x + \Delta x)}{x(x + \Delta x)}}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{-\Delta x}{x(x + \Delta x) \cdot \Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{-1}{x(x + \Delta x)}
$$

$$
= \frac{-1}{x \cdot x} = -\frac{1}{x^2}
$$

**Key Observations:**
- The derivative is negative everywhere (where defined) — the function is always decreasing
- Both function and derivative are undefined at x = 0
- This matches the power rule: d/dx[x⁻¹] = -1 · x⁻²

---

#### The Exponential Function: f(x) = eˣ

This is perhaps the most important function in all of calculus.

**The Magical Property:**

$$
\frac{d}{dx}[e^x] = e^x
$$

**The exponential function is its own derivative!**

**Graph:**
```
        │        /
        │       /
        │      /
        │    _/
        │  _/
        │_/
   ─────┼──────────
        │
```

**What is e?**

Euler's number e ≈ 2.71828... is defined as:

$$
e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n
$$

Or equivalently:

$$
e = \sum_{n=0}^{\infty} \frac{1}{n!} = 1 + 1 + \frac{1}{2} + \frac{1}{6} + \frac{1}{24} + \ldots
$$

**Why eˣ Equals Its Own Derivative:**

For a function to equal its own derivative, we need f(x) = f'(x). If f(x) tried to cross zero, then f'(x) = 0 at that point, meaning the function would be flat there and could never escape zero. So f(x) must be always positive or always negative.

The unique non-trivial solution (up to scaling) is eˣ.

**Properties of eˣ:**
- e⁰ = 1
- eˣ > 0 for all real x
- eˣ⁺ʸ = eˣ · eʸ
- dⁿ/dxⁿ[eˣ] = eˣ for all n (any number of derivatives gives the same function)

**General Exponential Rule:**

For eᵏˣ where k is a constant:

$$
\frac{d}{dx}[e^{kx}] = ke^{kx}
$$

This follows from the chain rule (covered later).

**Why It Matters for ML:**

The exponential function appears everywhere:
- **Softmax activation:** softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ
- **Sigmoid function:** σ(x) = 1 / (1 + e⁻ˣ)
- **Probability distributions:** Normal, Poisson, exponential families
- **Loss functions:** Cross-entropy involves logarithms (inverse of exponential)

---

#### The Natural Logarithm: f(x) = ln(x)

The natural logarithm is the inverse of eˣ: if y = eˣ, then x = ln(y).

**Derivative:**

$$
\frac{d}{dx}[\ln(x)] = \frac{1}{x}
$$

**Graph:**
```
        │      _____
        │    _/
        │  _/
        │_/
   ─────┼──────────
       /│
      / │
        │
```

**Derivation (using inverse function theorem):**

If y = ln(x), then x = eʸ.

Differentiating implicitly: 1 = eʸ dy/dx

So: dy/dx = 1/eʸ = 1/x

**Properties:**
- ln(1) = 0
- ln(e) = 1
- ln(ab) = ln(a) + ln(b)
- ln(aᵇ) = b·ln(a)

---

#### Trigonometric Functions: sin(x) and cos(x)

These functions are fundamental to modeling periodic phenomena.

**Derivatives:**

$$
\frac{d}{dx}[\sin(x)] = \cos(x)
$$

$$
\frac{d}{dx}[\cos(x)] = -\sin(x)
$$

**The Cyclic Pattern:**

Differentiating repeatedly creates a cycle:

```
sin(x) → cos(x) → -sin(x) → -cos(x) → sin(x) → ...
```

After four derivatives, we return to the original function!

| Derivative | Result |
|------------|--------|
| d/dx[sin x] | cos x |
| d²/dx²[sin x] | -sin x |
| d³/dx³[sin x] | -cos x |
| d⁴/dx⁴[sin x] | sin x |

**Derivation of d/dx[sin x] = cos x:**

Using the limit definition and the sum-to-product identity:

$$
\frac{d}{dx}[\sin x] = \lim_{h \to 0} \frac{\sin(x + h) - \sin(x)}{h}
$$

Using sin(x + h) = sin x cos h + cos x sin h:

$$
= \lim_{h \to 0} \frac{\sin x \cos h + \cos x \sin h - \sin x}{h}
$$

$$
= \lim_{h \to 0} \frac{\sin x (\cos h - 1) + \cos x \sin h}{h}
$$

$$
= \sin x \lim_{h \to 0} \frac{\cos h - 1}{h} + \cos x \lim_{h \to 0} \frac{\sin h}{h}
$$

Using the fundamental limits (sin h)/h → 1 and (cos h - 1)/h → 0 as h → 0:

$$
= \sin x \cdot 0 + \cos x \cdot 1 = \cos x
$$

**Other Trigonometric Derivatives:**

| Function | Derivative |
|----------|------------|
| tan(x) | sec²(x) = 1/cos²(x) |
| cot(x) | -csc²(x) |
| sec(x) | sec(x)tan(x) |
| csc(x) | -csc(x)cot(x) |

**The Deep Connection:**

Remarkably, the trigonometric functions are related to exponentials through Euler's formula:

$$
e^{ix} = \cos(x) + i\sin(x)
$$

This explains why they share the self-similarity property under differentiation!

---

### 1.7 The Product Rule

When differentiating a product of two functions, we cannot simply differentiate each factor separately.

#### The Rule

$$
\frac{d}{dx}[f(x) \cdot g(x)] = f(x) \cdot g'(x) + g(x) \cdot f'(x)
$$

Or more memorably: "First times derivative of second, plus second times derivative of first."

#### Geometric Intuition

Imagine a rectangle with:
- Width = f(x)
- Height = g(x)
- Area = A(x) = f(x) · g(x)

When x increases by Δx:
- Width changes by Δf
- Height changes by Δg

The new area is (f + Δf)(g + Δg) = fg + fΔg + gΔf + ΔfΔg

The change in area:

$$
\Delta A = f\Delta g + g\Delta f + \Delta f \Delta g
$$

The rate of change:

$$
\frac{\Delta A}{\Delta x} = f\frac{\Delta g}{\Delta x} + g\frac{\Delta f}{\Delta x} + \Delta f \frac{\Delta g}{\Delta x}
$$

As Δx → 0, the last term vanishes (it's a product of two small quantities), leaving:

$$
\frac{dA}{dx} = f\frac{dg}{dx} + g\frac{df}{dx} = fg' + gf'
$$

#### Rigorous Proof

$$
\frac{d}{dx}[f(x)g(x)] = \lim_{\Delta x \to 0} \frac{f(x+\Delta x)g(x+\Delta x) - f(x)g(x)}{\Delta x}
$$

Add and subtract f(x+Δx)g(x) in the numerator:

$$
= \lim_{\Delta x \to 0} \frac{f(x+\Delta x)g(x+\Delta x) - f(x+\Delta x)g(x) + f(x+\Delta x)g(x) - f(x)g(x)}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \left[ f(x+\Delta x) \frac{g(x+\Delta x) - g(x)}{\Delta x} + g(x) \frac{f(x+\Delta x) - f(x)}{\Delta x} \right]
$$

$$
= f(x) \cdot g'(x) + g(x) \cdot f'(x)
$$

#### Detailed Examples

**Example 1:** Differentiate h(x) = x² sin(x)

Let f(x) = x² and g(x) = sin(x).
- f'(x) = 2x
- g'(x) = cos(x)

$$
h'(x) = x^2 \cdot \cos(x) + \sin(x) \cdot 2x = x^2\cos(x) + 2x\sin(x)
$$

**Example 2:** Differentiate h(x) = eˣ ln(x)

Let f(x) = eˣ and g(x) = ln(x).
- f'(x) = eˣ
- g'(x) = 1/x

$$
h'(x) = e^x \cdot \frac{1}{x} + \ln(x) \cdot e^x = e^x\left(\frac{1}{x} + \ln(x)\right)
$$

**Example 3:** Differentiate h(x) = (x² + 1)(x³ - 2x)

Let f(x) = x² + 1 and g(x) = x³ - 2x.
- f'(x) = 2x
- g'(x) = 3x² - 2

$$
h'(x) = (x^2 + 1)(3x^2 - 2) + (x^3 - 2x)(2x)
$$

$$
= 3x^4 - 2x^2 + 3x^2 - 2 + 2x^4 - 4x^2
$$

$$
= 5x^4 - 3x^2 - 2
$$

#### The Quotient Rule (Derived from Product Rule)

For f(x)/g(x), we can write this as f(x) · [g(x)]⁻¹ and apply the product and chain rules, or use the quotient rule directly:

$$
\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{g(x)f'(x) - f(x)g'(x)}{[g(x)]^2}
$$

Memory aid: "Low d-high minus high d-low, all over low squared."

**Example:** Differentiate (x² + 1)/(x - 3)

$$
= \frac{(x-3)(2x) - (x^2+1)(1)}{(x-3)^2} = \frac{2x^2 - 6x - x^2 - 1}{(x-3)^2} = \frac{x^2 - 6x - 1}{(x-3)^2}
$$

---

### 1.8 The Chain Rule

The chain rule is perhaps the most important differentiation rule, especially for machine learning where we deal with compositions of many functions.

#### The Rule

For nested functions h(x) = f(g(x)):

$$
h'(x) = f'(g(x)) \cdot g'(x)
$$

Or in Leibniz notation:

$$
\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

**Interpretation:** The rate of change of h with respect to x equals the rate of change of f with respect to g, multiplied by the rate of change of g with respect to x.

#### The Happiness-Pizza-Money Example (Detailed)

Let's build an economic model:

**Happiness as a function of pizza:**

$$
h(p) = -\frac{1}{3}p^2 + p + \frac{1}{5}
$$

This captures:
- Some baseline happiness (1/5)
- Happiness increases with pizza (the +p term)
- But too much pizza makes you sick (the -p²/3 term)

**Pizza as a function of money:**

$$
p(m) = e^m - 1
$$

This captures:
- No money = no pizza (p(0) = 0)
- More money = exponentially more pizza-buying power

**Question:** How does happiness change with money? We need dh/dm.

**Step 1:** Find individual derivatives

$$
\frac{dh}{dp} = -\frac{2}{3}p + 1
$$

$$
\frac{dp}{dm} = e^m
$$

**Step 2:** Apply the chain rule

$$
\frac{dh}{dm} = \frac{dh}{dp} \cdot \frac{dp}{dm} = \left(-\frac{2}{3}p + 1\right) \cdot e^m
$$

**Step 3:** Express in terms of m only

Substitute p = eᵐ - 1:

$$
\frac{dh}{dm} = \left(-\frac{2}{3}(e^m - 1) + 1\right) \cdot e^m
$$

$$
= \left(-\frac{2}{3}e^m + \frac{2}{3} + 1\right) \cdot e^m
$$

$$
= \left(\frac{5}{3} - \frac{2}{3}e^m\right) \cdot e^m
$$

$$
= \frac{e^m}{3}(5 - 2e^m)
$$

**Interpretation:** 
- When eᵐ < 5/2 (i.e., m < ln(2.5) ≈ 0.92), more money increases happiness
- When eᵐ > 5/2 (i.e., m > 0.92), more money decreases happiness (too much pizza!)

#### Why the Chain Rule Works — Intuition

Think of it as unit conversion:

$$
\frac{dh}{dm} = \frac{dh}{dp} \cdot \frac{dp}{dm}
$$

- dh/dp: happiness per pizza
- dp/dm: pizzas per dollar

The product gives: happiness per dollar

The "dp" terms "cancel" (though this is a heuristic, not rigorous mathematics).

#### Extended Chain Rule for Multiple Compositions

For h(x) = f(g(k(x))):

$$
h'(x) = f'(g(k(x))) \cdot g'(k(x)) \cdot k'(x)
$$

**Example:** Differentiate h(x) = sin(e^(x²))

Let:
- k(x) = x² → k'(x) = 2x
- g(u) = eᵘ → g'(u) = eᵘ
- f(v) = sin(v) → f'(v) = cos(v)

Then:

$$
h'(x) = \cos(e^{x^2}) \cdot e^{x^2} \cdot 2x = 2x e^{x^2} \cos(e^{x^2})
$$

#### Practical Pattern Recognition

When differentiating composite functions, identify:
1. The "outer" function
2. The "inner" function
3. Differentiate outer, keep inner unchanged
4. Multiply by derivative of inner

**Examples:**

| Function | Outer | Inner | Derivative |
|----------|-------|-------|------------|
| (3x+1)⁵ | u⁵ | 3x+1 | 5(3x+1)⁴ · 3 |
| e^(x²) | eᵘ | x² | e^(x²) · 2x |
| sin(2x) | sin(u) | 2x | cos(2x) · 2 |
| ln(x²+1) | ln(u) | x²+1 | 1/(x²+1) · 2x |
| √(1-x²) | u^(1/2) | 1-x² | 1/(2√(1-x²)) · (-2x) |

---

### 1.9 Comprehensive Worked Examples

Let's tackle some challenging problems using all four rules together.

#### Example 1: Complex Composite Function

Differentiate:

$$
f(x) = \frac{\sin(2x^5 + 3x)}{e^{7x}}
$$

**Step 1:** Rewrite as a product (avoiding quotient rule)

$$
f(x) = \sin(2x^5 + 3x) \cdot e^{-7x}
$$

**Step 2:** Define components

- g(x) = sin(2x⁵ + 3x)
- h(x) = e⁻⁷ˣ

**Step 3:** Differentiate g(x) using chain rule

Let u = 2x⁵ + 3x, so g = sin(u)

$$
\frac{du}{dx} = 10x^4 + 3
$$

$$
\frac{dg}{du} = \cos(u)
$$

$$
g'(x) = \cos(2x^5 + 3x) \cdot (10x^4 + 3)
$$

**Step 4:** Differentiate h(x) using chain rule

Let v = -7x, so h = eᵛ

$$
h'(x) = e^{-7x} \cdot (-7) = -7e^{-7x}
$$

**Step 5:** Apply product rule

$$
f'(x) = g(x) \cdot h'(x) + h(x) \cdot g'(x)
$$

$$
= \sin(2x^5 + 3x) \cdot (-7e^{-7x}) + e^{-7x} \cdot \cos(2x^5 + 3x) \cdot (10x^4 + 3)
$$

$$
= e^{-7x}\left[(10x^4 + 3)\cos(2x^5 + 3x) - 7\sin(2x^5 + 3x)\right]
$$

---

#### Example 2: Logarithmic Function with Chain Rule

Differentiate:

$$
f(x) = \ln\left(\frac{x^2 + 1}{x - 1}\right)
$$

**Method 1:** Use log properties first

$$
f(x) = \ln(x^2 + 1) - \ln(x - 1)
$$

Now differentiate each term:

$$
f'(x) = \frac{2x}{x^2 + 1} - \frac{1}{x - 1}
$$

To combine:

$$
f'(x) = \frac{2x(x-1) - (x^2+1)}{(x^2+1)(x-1)} = \frac{2x^2 - 2x - x^2 - 1}{(x^2+1)(x-1)} = \frac{x^2 - 2x - 1}{(x^2+1)(x-1)}
$$

---

#### Example 3: Implicit Differentiation Preview

Sometimes we have an equation relating x and y without y being explicitly solved:

$$
x^2 + y^2 = 25 \quad \text{(a circle)}
$$

To find dy/dx, differentiate both sides with respect to x, treating y as a function of x:

$$
\frac{d}{dx}[x^2] + \frac{d}{dx}[y^2] = \frac{d}{dx}[25]
$$

$$
2x + 2y\frac{dy}{dx} = 0
$$

Solving for dy/dx:

$$
\frac{dy}{dx} = -\frac{x}{y}
$$

This gives the slope of the tangent to the circle at any point (x, y).

---

## Module 1 Summary — The Differentiation Toolbox

| Rule | Formula | Application |
|------|---------|-------------|
| **Limit Definition** | f'(x) = lim[Δx→0] (f(x+Δx) - f(x))/Δx | Fundamentals, proofs |
| **Sum Rule** | (f + g)' = f' + g' | Polynomials, sums |
| **Constant Multiple** | (cf)' = cf' | Scaling |
| **Power Rule** | (xⁿ)' = nxⁿ⁻¹ | Polynomials, radicals |
| **Product Rule** | (fg)' = fg' + gf' | Products of functions |
| **Quotient Rule** | (f/g)' = (gf' - fg')/g² | Fractions |
| **Chain Rule** | [f(g(x))]' = f'(g(x)) · g'(x) | Composite functions |

**Essential Derivatives:**

| Function | Derivative |
|----------|------------|
| c (constant) | 0 |
| xⁿ | nxⁿ⁻¹ |
| eˣ | eˣ |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |
| tan(x) | sec²(x) |
| eᵏˣ | keᵏˣ |
| ln(kx) | 1/x |

---

## Module 2: Multivariate Calculus

This module extends everything from Module 1 to functions of multiple variables — essential for machine learning where we optimize functions with thousands or millions of parameters.

---

### 2.1 Variables, Constants, and Parameters

Before computing derivatives of multivariate functions, we must understand what we're differentiating with respect to.

#### The Context-Dependent Nature of Variables

What counts as a "variable" versus a "constant" depends entirely on the problem you're solving.

**Example: The Ideal Gas Law**

$$
PV = nRT
$$

Where:
- P = pressure
- V = volume  
- n = amount of gas (moles)
- R = gas constant
- T = temperature

**Scenario 1: Heating gas in a rigid container**
- V and n are constants (container is sealed and rigid)
- T is the independent variable (what you control)
- P is the dependent variable (what you measure)

**Scenario 2: Inflating a balloon**
- P and T are constants (atmospheric pressure, room temperature)
- n is the independent variable (how much air you add)
- V is the dependent variable (balloon size)

**Scenario 3: Designing a pressure vessel**
- P and T are fixed specifications
- V is a design variable
- Material properties become relevant

**The Key Insight:** The same equation can describe completely different physical situations depending on what we hold fixed and what we allow to vary.

#### Parameters vs. Variables

A **parameter** is a quantity that:
- Is fixed for a particular problem instance
- But might be varied to study a family of related problems
- Or optimized to fit data

**Machine Learning Perspective:**

In a neural network y = σ(Wx + b):

- **Inputs (x):** Variables that change with each data point
- **Outputs (y):** Dependent variables we're predicting
- **Weights (W) and biases (b):** Parameters we optimize during training
- **Architecture choices:** Hyperparameters (number of layers, etc.)

When we train the network, we differentiate the loss with respect to W and b, treating x and y as fixed for each training example.

---

### 2.2 Partial Differentiation

When a function depends on multiple variables, a **partial derivative** measures how the function changes when we vary ONE variable while holding all others constant.

#### Notation and Definition

For f(x, y), the partial derivative with respect to x is:

$$
\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x}
$$

Note that y is held constant throughout.

**Notation variants:**
- ∂f/∂x — Leibniz notation
- fₓ — Subscript notation
- ∂ₓf — Operator notation

#### The Procedure

To find ∂f/∂x:
1. Treat all variables except x as constants
2. Differentiate with respect to x using the usual rules

#### Detailed Example: Metal Can

The mass of a cylindrical can with radius r, height h, wall thickness t, and metal density ρ:

$$
m = (2\pi r^2 + 2\pi rh) \cdot t \cdot \rho
$$

Let's expand this:

$$
m = 2\pi r^2 t\rho + 2\pi rht\rho
$$

**Partial derivative with respect to r:**

Treat h, t, ρ as constants:

$$
\frac{\partial m}{\partial r} = 2\pi \cdot 2r \cdot t\rho + 2\pi h t\rho = 4\pi rt\rho + 2\pi ht\rho
$$

**Interpretation:** This tells us how sensitive the can's mass is to changes in radius. If we increase r slightly, the mass increases by approximately (∂m/∂r) · Δr.

**Partial derivative with respect to h:**

$$
\frac{\partial m}{\partial h} = 0 + 2\pi r t\rho = 2\pi rt\rho
$$

The first term (2πr²tρ) doesn't contain h, so its derivative is 0.

**Partial derivative with respect to t:**

$$
\frac{\partial m}{\partial t} = 2\pi r^2 \rho + 2\pi rh\rho
$$

**Partial derivative with respect to ρ:**

$$
\frac{\partial m}{\partial \rho} = 2\pi r^2 t + 2\pi rht
$$

#### More Complex Example

Let f(x, y, z) = sin(x) · e^(yz²)

**Finding ∂f/∂x:**

Treat y and z as constants. The e^(yz²) term is just a constant with respect to x:

$$
\frac{\partial f}{\partial x} = \cos(x) \cdot e^{yz^2}
$$

**Finding ∂f/∂y:**

Treat x and z as constants. Now sin(x) is a constant, and we need to differentiate e^(yz²) with respect to y.

Using the chain rule with u = yz²:

$$
\frac{\partial}{\partial y}[e^{yz^2}] = e^{yz^2} \cdot \frac{\partial}{\partial y}[yz^2] = e^{yz^2} \cdot z^2
$$

So:

$$
\frac{\partial f}{\partial y} = \sin(x) \cdot e^{yz^2} \cdot z^2
$$

**Finding ∂f/∂z:**

Using the chain rule with u = yz²:

$$
\frac{\partial}{\partial z}[e^{yz^2}] = e^{yz^2} \cdot \frac{\partial}{\partial z}[yz^2] = e^{yz^2} \cdot 2yz
$$

So:

$$
\frac{\partial f}{\partial z} = \sin(x) \cdot e^{yz^2} \cdot 2yz
$$

#### Higher-Order Partial Derivatives

We can take partial derivatives of partial derivatives:

**Second-order partials:**

$$
\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)
$$

**Mixed partials:**

$$
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)
$$

**Clairaut's Theorem:** For functions with continuous second derivatives:

$$
\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}
$$

The order of differentiation doesn't matter!

---

### 2.3 The Total Derivative

When all variables in a function depend on a single parameter, we can find how the function changes with that parameter.

#### The Setup

Given f(x, y, z) where:
- x = x(t)
- y = y(t)
- z = z(t)

We want df/dt.

#### The Formula

$$
\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} + \frac{\partial f}{\partial z}\frac{dz}{dt}
$$

This is the **multivariate chain rule**.

#### Intuition

Think of it as adding up all the ways t can affect f:
- t affects x, which affects f → contribution: (∂f/∂x)(dx/dt)
- t affects y, which affects f → contribution: (∂f/∂y)(dy/dt)
- t affects z, which affects f → contribution: (∂f/∂z)(dz/dt)

Total effect = sum of all contributions.

#### Detailed Example

Let f(x, y, z) = sin(x) · e^(yz²) with:
- x(t) = t - 1
- y(t) = t²
- z(t) = 1/t

**Step 1:** Find all partial derivatives (done above)

$$
\frac{\partial f}{\partial x} = \cos(x) \cdot e^{yz^2}
$$

$$
\frac{\partial f}{\partial y} = \sin(x) \cdot e^{yz^2} \cdot z^2
$$

$$
\frac{\partial f}{\partial z} = \sin(x) \cdot e^{yz^2} \cdot 2yz
$$

**Step 2:** Find derivatives of x, y, z with respect to t

$$
\frac{dx}{dt} = 1
$$

$$
\frac{dy}{dt} = 2t
$$

$$
\frac{dz}{dt} = -\frac{1}{t^2}
$$

**Step 3:** Apply the total derivative formula

$$
\frac{df}{dt} = \cos(x)e^{yz^2} \cdot 1 + \sin(x)e^{yz^2}z^2 \cdot 2t + \sin(x)e^{yz^2} \cdot 2yz \cdot \left(-\frac{1}{t^2}\right)
$$

**Step 4:** Substitute x = t-1, y = t², z = 1/t

Note that yz² = t² · (1/t²) = 1, so e^(yz²) = e.

Also z² = 1/t² and yz = t² · (1/t) = t.

$$
\frac{df}{dt} = \cos(t-1) \cdot e + \sin(t-1) \cdot e \cdot \frac{1}{t^2} \cdot 2t - \sin(t-1) \cdot e \cdot 2t \cdot \frac{1}{t^2}
$$

$$
= e\cos(t-1) + \frac{2e\sin(t-1)}{t} - \frac{2e\sin(t-1)}{t}
$$

$$
= e\cos(t-1)
$$

The second and third terms cancel!

---

### 2.4 The Jacobian Vector

The **Jacobian** collects all partial derivatives of a scalar function into a single vector.

#### Definition

For f: ℝⁿ → ℝ (scalar-valued function of n variables):

$$
J = \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

By convention, this is written as a **row vector**.

The symbol ∇ (nabla or "del") is called the **gradient operator**.

#### Example: Three-Variable Function

For f(x, y, z) = x²y + 3z:

$$
\frac{\partial f}{\partial x} = 2xy
$$

$$
\frac{\partial f}{\partial y} = x^2
$$

$$
\frac{\partial f}{\partial z} = 3
$$

Therefore:

$$
J = \nabla f = \begin{pmatrix} 2xy & x^2 & 3 \end{pmatrix}
$$

#### Evaluating at Specific Points

The Jacobian is itself a function of position. At (0, 0, 0):

$$
J(0, 0, 0) = \begin{pmatrix} 0 & 0 & 3 \end{pmatrix}
$$

At (1, 2, 0):

$$
J(1, 2, 0) = \begin{pmatrix} 4 & 1 & 3 \end{pmatrix}
$$

#### Geometric Interpretation

The Jacobian vector has profound geometric meaning:

1. **Direction:** Points in the direction of steepest increase of f
2. **Magnitude:** The length |∇f| equals the rate of change in that steepest direction
3. **Perpendicular to Level Sets:** ∇f is perpendicular to the contour lines/surfaces where f is constant

**Visualizing in 2D:**

Imagine a topographic map with contour lines showing elevation. The gradient at any point:
- Points directly uphill
- Is perpendicular to the contour line at that point
- Is longest where contours are closest together (steep slope)
- Is zero at peaks, valleys, and saddle points

#### The Total Derivative Revisited

Using vector notation, the total derivative becomes:

$$
\frac{df}{dt} = \nabla f \cdot \frac{d\mathbf{x}}{dt}
$$

This is the **dot product** of the gradient with the velocity vector of **x**(t).

---

### 2.5 The Jacobian Matrix

When both inputs and outputs are vectors, the Jacobian becomes a **matrix**.

#### The Setup

Consider a vector-valued function **F**: ℝⁿ → ℝᵐ:

$$
\mathbf{F}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\\\ f_2(x_1, \ldots, x_n) \\\\ \vdots \\\\ f_m(x_1, \ldots, x_n) \end{pmatrix}
$$

#### The Jacobian Matrix

$$
J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\\\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}
$$

Each row is the gradient of one output function.

Dimensions: If **F**: ℝⁿ → ℝᵐ, then J is an m × n matrix.

#### Example: Linear Transformation

Consider the mapping from (x, y) to (u, v):
- u(x, y) = x + 2y
- v(x, y) = 3y - 2x

**Finding the Jacobian:**

$$
J = \begin{pmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\\\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{pmatrix} = \begin{pmatrix} 1 & 2 \\\\ -2 & 3 \end{pmatrix}
$$

**Key Observation:** For linear functions, the Jacobian is constant — it's the transformation matrix itself!

**Applying the transformation:**

$$
\begin{pmatrix} u \\\\ v \end{pmatrix} = \begin{pmatrix} 1 & 2 \\\\ -2 & 3 \end{pmatrix} \begin{pmatrix} x \\\\ y \end{pmatrix}
$$

For input (2, 3):

$$
\begin{pmatrix} u \\\\ v \end{pmatrix} = \begin{pmatrix} 1 & 2 \\\\ -2 & 3 \end{pmatrix} \begin{pmatrix} 2 \\\\ 3 \end{pmatrix} = \begin{pmatrix} 8 \\\\ 5 \end{pmatrix}
$$

#### Example: Coordinate Transformation (Polar to Cartesian)

The transformation from polar (r, θ) to Cartesian (x, y):
- x = r cos(θ)
- y = r sin(θ)

**The Jacobian:**

$$
J = \begin{pmatrix} \frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta} \\\\ \frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta} \end{pmatrix} = \begin{pmatrix} \cos\theta & -r\sin\theta \\\\ \sin\theta & r\cos\theta \end{pmatrix}
$$

**The Jacobian Determinant:**

$$
|J| = \cos\theta \cdot r\cos\theta - (-r\sin\theta) \cdot \sin\theta = r\cos^2\theta + r\sin^2\theta = r
$$

**Significance:** The determinant |J| = r tells us how areas scale under the transformation. This is why polar integrals include an r factor:

$$
\iint f(x, y) \, dx \, dy = \iint f(r\cos\theta, r\sin\theta) \cdot r \, dr \, d\theta
$$

The r is the Jacobian determinant!

#### Multivariate Chain Rule with Jacobians

For **F**(**G**(**x**)), the Jacobian of the composition is:

$$
J_{\mathbf{F} \circ \mathbf{G}} = J_{\mathbf{F}} \cdot J_{\mathbf{G}}
$$

This is matrix multiplication! The chain rule becomes a simple matrix product.

---

### 2.6 The Hessian Matrix

The **Hessian** collects all second-order partial derivatives, providing information about curvature.

#### Definition

For f: ℝⁿ → ℝ:

$$
H = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}
$$

**Key Property:** The Hessian is symmetric (by Clairaut's theorem): H = Hᵀ

#### Computing the Hessian

**Step 1:** Find the gradient (Jacobian)
**Step 2:** Differentiate each component of the gradient again

#### Example

For f(x, y, z) = x²yz:

**Step 1: The gradient**

$$
\nabla f = \begin{pmatrix} 2xyz & x^2z & x^2y \end{pmatrix}
$$

**Step 2: Differentiate each component**

First row (differentiate 2xyz with respect to x, y, z):

$$
\frac{\partial}{\partial x}[2xyz] = 2yz, \quad \frac{\partial}{\partial y}[2xyz] = 2xz, \quad \frac{\partial}{\partial z}[2xyz] = 2xy
$$

Second row (differentiate x²z with respect to x, y, z):

$$
\frac{\partial}{\partial x}[x^2z] = 2xz, \quad \frac{\partial}{\partial y}[x^2z] = 0, \quad \frac{\partial}{\partial z}[x^2z] = x^2
$$

Third row (differentiate x²y with respect to x, y, z):

$$
\frac{\partial}{\partial x}[x^2y] = 2xy, \quad \frac{\partial}{\partial y}[x^2y] = x^2, \quad \frac{\partial}{\partial z}[x^2y] = 0
$$

**The Hessian:**

$$
H = \begin{pmatrix} 2yz & 2xz & 2xy \\\\ 2xz & 0 & x^2 \\\\ 2xy & x^2 & 0 \end{pmatrix}
$$

Notice the symmetry: H₁₂ = H₂₁ = 2xz, H₁₃ = H₃₁ = 2xy, H₂₃ = H₃₂ = x².

#### Classifying Critical Points

A **critical point** is where ∇f = **0**.

The Hessian at a critical point tells us what type of point it is:

**The Second Derivative Test (2D):**

Let D = det(H) = H₁₁H₂₂ - H₁₂²

| Condition | Classification |
|-----------|----------------|
| D > 0 and H₁₁ > 0 | **Local minimum** |
| D > 0 and H₁₁ < 0 | **Local maximum** |
| D < 0 | **Saddle point** |
| D = 0 | Inconclusive (need higher-order analysis) |

**General Case (n dimensions):**

- All eigenvalues of H positive → Local minimum
- All eigenvalues of H negative → Local maximum
- Mixed sign eigenvalues → Saddle point

#### Example: Classifying Critical Points

**Function:** f(x, y) = x² + y²

**Gradient:**

$$
\nabla f = \begin{pmatrix} 2x & 2y \end{pmatrix}
$$

Critical point: ∇f = **0** → (x, y) = (0, 0)

**Hessian:**

$$
H = \begin{pmatrix} 2 & 0 \\\\ 0 & 2 \end{pmatrix}
$$

**Classification:**
- D = 2 · 2 - 0 = 4 > 0
- H₁₁ = 2 > 0

**Conclusion:** (0, 0) is a **local minimum** (actually global minimum).

---

**Function:** f(x, y) = x² - y²

**Gradient:**

$$
\nabla f = \begin{pmatrix} 2x & -2y \end{pmatrix}
$$

Critical point: (0, 0)

**Hessian:**

$$
H = \begin{pmatrix} 2 & 0 \\\\ 0 & -2 \end{pmatrix}
$$

**Classification:**
- D = 2 · (-2) - 0 = -4 < 0

**Conclusion:** (0, 0) is a **saddle point**.

This function looks like a horse saddle or a Pringles chip — it curves up in the x-direction but down in the y-direction.

---

### 2.7 Numerical Differentiation

In practice, we often can't find analytical expressions for derivatives. Numerical methods provide approximations.

#### The Forward Difference

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h}
$$

This is our original "rise over run" with a small but finite h.

**Error:** O(h) — first-order accurate

#### The Central Difference

$$
f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

This uses points on both sides of x.

**Error:** O(h²) — second-order accurate (much better!)

#### Why Central Difference is Better

Using Taylor series:

$$
f(x + h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)
$$

$$
f(x - h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)
$$

Subtracting:

$$
f(x + h) - f(x - h) = 2hf'(x) + O(h^3)
$$

Therefore:

$$
\frac{f(x + h) - f(x - h)}{2h} = f'(x) + O(h^2)
$$

The second-order terms cancel!

#### Numerical Partial Derivatives

For multivariate functions, apply the same approach to each variable:

$$
\frac{\partial f}{\partial x} \approx \frac{f(x + h, y) - f(x - h, y)}{2h}
$$

$$
\frac{\partial f}{\partial y} \approx \frac{f(x, y + h) - f(x, y - h)}{2h}
$$

#### Choosing Step Size h

**Too large:** Poor approximation (truncation error)

**Too small:** Numerical precision issues (round-off error)

A common choice: h ≈ √ε where ε is machine precision (≈ 10⁻¹⁶ for 64-bit floats), giving h ≈ 10⁻⁸.

---

## Module 3: Optimization and Neural Networks

This module connects calculus to machine learning, showing how derivatives enable neural networks to learn.

---

### 3.1 Introduction to Optimization

#### What is Optimization?

**Optimization** is the process of finding input values that maximize or minimize a function.

In mathematical notation:

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{or} \quad \max_{\mathbf{x}} f(\mathbf{x})
$$

We seek the argument **x*** that achieves the extreme value:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} f(\mathbf{x})
$$

#### Applications

**Route Planning:**
- Variables: sequence of roads/turns
- Objective: minimize travel time or distance
- Constraints: one-way streets, road closures

**Portfolio Selection:**
- Variables: fraction of money in each asset
- Objective: maximize return or minimize risk
- Constraints: budget, no short-selling

**Machine Learning:**
- Variables: model parameters (weights, biases)
- Objective: minimize prediction error (loss function)
- Constraints: regularization, architecture

#### The Landscape Analogy

Think of f(**x**) as the elevation of a landscape where **x** = (x, y) are coordinates.

- **Local minimum:** Bottom of a valley (lowest point nearby)
- **Global minimum:** Lowest point overall
- **Local maximum:** Top of a hill
- **Global maximum:** Highest point overall
- **Saddle point:** Low in one direction, high in another

#### Challenges in Optimization

1. **Multiple Local Minima:** How do we know we've found the global minimum?

2. **High Dimensionality:** Can't visualize beyond 3D; intuition fails

3. **Expensive Evaluation:** Each function call might require extensive computation

4. **Non-Smoothness:** Discontinuities or sharp corners break gradient methods

5. **Noise:** Real data has measurement error

6. **Constraints:** Some regions of the search space may be forbidden

#### The Sandpit Analogy

Imagine finding the lowest point of an irregularly-shaped sandpit:

- You're blindfolded (can't see the whole landscape)
- You have a long stick to measure depth at any point
- Each measurement takes time/resources
- You can't move the stick sideways underground

This is optimization without a closed-form solution:
- We can evaluate f(**x**) at any point
- But we can't see f everywhere at once
- We need a strategy to find the minimum efficiently

---

### 3.2 Neural Network Architecture

A neural network is a function built from layers of simpler functions.

#### The Single Neuron

The simplest network: one input a₀, one output a₁:

$$
a_1 = \sigma(wa_0 + b)
$$

**Components:**

| Symbol | Name | Role |
|--------|------|------|
| a₀ | Input activation | Data entering the network |
| w | Weight | Scales the input (learned parameter) |
| b | Bias | Shifts the result (learned parameter) |
| σ | Activation function | Introduces nonlinearity |
| a₁ | Output activation | Result of the computation |

#### Activation Functions

Activation functions introduce **nonlinearity**, enabling networks to learn complex patterns.

**Sigmoid / Logistic:**

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- Output range: (0, 1)
- Derivative: σ'(x) = σ(x)(1 - σ(x))
- Used for: Binary classification (output layer)

**Hyperbolic Tangent (tanh):**

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

- Output range: (-1, 1)
- Derivative: tanh'(x) = 1 - tanh²(x)
- Zero-centered (often better than sigmoid)

**ReLU (Rectified Linear Unit):**

$$
\text{ReLU}(x) = \max(0, x)
$$

- Output range: [0, ∞)
- Derivative: 0 if x < 0, 1 if x > 0, undefined at x = 0
- Most popular for hidden layers (fast, avoids vanishing gradients)

**Why Nonlinearity Matters:**

Without activation functions, a multi-layer network collapses to a single linear transformation:

$$
W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + \mathbf{b}_2) = W'\mathbf{x} + \mathbf{b}'
$$

No matter how many layers, you just get a linear function. Nonlinear activations enable networks to approximate any continuous function.

#### Scaling Up: Multiple Inputs

With n inputs a₀,₁, a₀,₂, ..., a₀,ₙ:

$$
a_1 = \sigma\left(\sum_{i=1}^{n} w_i a_{0,i} + b\right) = \sigma(\mathbf{w}^T \mathbf{a}_0 + b)
$$

Each input has its own weight. The neuron computes a weighted sum, adds a bias, then applies the activation.

#### Scaling Up: Multiple Outputs

With m output neurons:

$$
\mathbf{a}_1 = \sigma(W\mathbf{a}_0 + \mathbf{b})
$$

Where:
- W is an m × n **weight matrix**
- **b** is an m-dimensional **bias vector**
- σ is applied element-wise

#### Adding Hidden Layers

**Deep learning** uses multiple layers between input and output:

```
Input → Hidden Layer 1 → Hidden Layer 2 → ... → Output
```

Each layer transforms the representation:

$$
\mathbf{a}^{(l+1)} = \sigma\left(W^{(l)}\mathbf{a}^{(l)} + \mathbf{b}^{(l)}\right)
$$

**Why Depth Matters:**

Deep networks can learn hierarchical representations:
- Early layers: Simple features (edges, colors)
- Middle layers: Combinations (shapes, textures)
- Later layers: Complex concepts (objects, faces)

#### Counting Parameters

For a network with layers of sizes n₀ → n₁ → n₂ → ... → nₗ:

- Weights between layer l and l+1: nₗ × nₗ₊₁
- Biases in layer l+1: nₗ₊₁

**Example:** 784 → 128 → 64 → 10

| Connection | Weights | Biases |
|------------|---------|--------|
| Input → Hidden 1 | 784 × 128 = 100,352 | 128 |
| Hidden 1 → Hidden 2 | 128 × 64 = 8,192 | 64 |
| Hidden 2 → Output | 64 × 10 = 640 | 10 |

**Total:** 100,352 + 8,192 + 640 + 128 + 64 + 10 = **109,386 parameters**

Modern networks have millions or billions of parameters!

---

### 3.3 Backpropagation — The Chain Rule in Action

**Backpropagation** is the algorithm for computing gradients of the loss function with respect to all parameters. It's the chain rule, systematically applied.

#### The Training Objective

**Goal:** Find weights W and biases **b** that minimize prediction error.

**Loss Function:** Measures how wrong our predictions are.

For regression (predicting continuous values):

$$
L = \frac{1}{2}\sum_i (y_i - \hat{y}_i)^2
$$

For classification (predicting categories):

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

Where:
- yᵢ = true label
- ŷᵢ = network prediction

#### The Training Loop

1. **Forward Pass:** Compute network output for training input
2. **Compute Loss:** Measure error between output and true label
3. **Backward Pass:** Compute gradient of loss with respect to all parameters
4. **Update Parameters:** Adjust parameters to reduce loss

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \frac{\partial L}{\partial \theta}
$$

Where θ represents any parameter and η is the learning rate.

#### Backpropagation for a Simple Network

Consider a two-layer network:

$$
z^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}
$$

$$
\mathbf{a}^{(1)} = \sigma(z^{(1)})
$$

$$
z^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}
$$

$$
\hat{\mathbf{y}} = \sigma(z^{(2)})
$$

$$
L = \frac{1}{2}||\mathbf{y} - \hat{\mathbf{y}}||^2
$$

**Goal:** Find ∂L/∂W⁽¹⁾, ∂L/∂**b**⁽¹⁾, ∂L/∂W⁽²⁾, ∂L/∂**b**⁽²⁾

**Step 1: Output Layer Gradients**

$$
\frac{\partial L}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}} - \mathbf{y}
$$

$$
\frac{\partial L}{\partial z^{(2)}} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \odot \sigma'(z^{(2)})
$$

(where ⊙ is element-wise multiplication)

$$
\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial z^{(2)}} (\mathbf{a}^{(1)})^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \frac{\partial L}{\partial z^{(2)}}
$$

**Step 2: Hidden Layer Gradients (Backpropagation)**

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}} = (W^{(2)})^T \frac{\partial L}{\partial z^{(2)}}
$$

$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial \mathbf{a}^{(1)}} \odot \sigma'(z^{(1)})
$$

$$
\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(1)}} \mathbf{x}^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \frac{\partial L}{\partial z^{(1)}}
$$

#### The Key Insight

Notice how gradients **propagate backward**:

```
Loss → Output layer → Hidden layer → ... → Input layer
```

The gradient at each layer depends on the gradient from the layer above, multiplied by local derivatives. This is the chain rule in matrix form!

#### Why It's Called "Backpropagation"

- **Forward pass:** Information flows from input to output
- **Backward pass:** Gradients flow from output back to input

The error "propagates back" through the network, telling each layer how to adjust.

#### Computational Efficiency

Without backpropagation, computing gradients for N parameters would require N forward passes (one per parameter using finite differences).

With backpropagation: **one forward pass + one backward pass** computes all gradients.

For networks with millions of parameters, this is the difference between feasible and impossible.

---

## Module 4: Taylor Series and Approximations

Taylor series allow us to approximate complex functions with polynomials, connecting abstract calculus to practical computation.

---

### 4.1 Why We Need Approximations

#### The Practical Problem

Many real-world functions are:
- Too complex to evaluate efficiently
- Only known at certain points (from experiments/simulations)
- Not available in closed form

Approximations let us work with simpler, tractable expressions.

#### The Chicken Cooking Example

Suppose the ideal cooking time T for a chicken depends on:
- Mass m
- Oven characteristics
- Chicken shape and composition
- Ambient conditions

A "complete" model might be a nightmare:

$$
T = f(m, \text{oven}, \text{shape}, \text{humidity}, \ldots)
$$

For a cookbook, we need something practical. Taylor series can give us:

$$
T \approx 50m + 15 \quad \text{(minutes, with } m \text{ in kg)}
$$

This linear approximation works well for typical chickens (1-3 kg).

#### When Approximations Are Essential

1. **Numerical Methods:** Computers use finite precision; we can't compute infinite series exactly

2. **Real-Time Systems:** Exact solutions may be too slow; approximations enable fast decisions

3. **Understanding Behavior:** Approximations reveal the essential structure of a function

4. **Solving Equations:** Many equations can't be solved analytically but can be solved approximately

---

### 4.2 Power Series — Building Intuition

A **power series** expresses a function as a sum of terms with increasing powers:

$$
g(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \ldots = \sum_{n=0}^{\infty} a_n x^n
$$

The coefficients aₙ determine the function.

#### Building Approximations Step by Step

**Zeroth Order (Constant):**

$$
g_0(x) = a_0
$$

Just match the function's value at one point.

**First Order (Linear):**

$$
g_1(x) = a_0 + a_1 x
$$

Also match the gradient (slope).

**Second Order (Quadratic):**

$$
g_2(x) = a_0 + a_1 x + a_2 x^2
$$

Also match the curvature (second derivative).

Each additional term captures more detail about the function's behavior.

#### Visual Progression

For an arbitrary curve:

```
Function:         Approximations:
                  
   _____          ____  (zeroth: horizontal line)
  /     \         
 /       \        ____  (first: tangent line)
/         \           \
                        \
                  
                   ____  (second: parabola)
                  /    \
                 /      \
```

The approximations match the function better and better near the expansion point.

---

### 4.3 The Maclaurin Series

The **Maclaurin series** is a Taylor series centered at x = 0.

#### The Formula

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} x^n
$$

$$
= f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \ldots
$$

Where f⁽ⁿ⁾(0) is the n-th derivative evaluated at zero.

#### Derivation

We want to find coefficients aₙ such that:

$$
f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + \ldots
$$

**Finding a₀:** Set x = 0:

$$
f(0) = a_0
$$

**Finding a₁:** Differentiate once, then set x = 0:

$$
f'(x) = a_1 + 2a_2 x + 3a_3 x^2 + \ldots
$$

$$
f'(0) = a_1
$$

**Finding a₂:** Differentiate twice, then set x = 0:

$$
f''(x) = 2a_2 + 6a_3 x + \ldots
$$

$$
f''(0) = 2a_2 \implies a_2 = \frac{f''(0)}{2}
$$

**Finding a₃:** Differentiate three times:

$$
f'''(x) = 6a_3 + \ldots
$$

$$
f'''(0) = 6a_3 \implies a_3 = \frac{f'''(0)}{6} = \frac{f'''(0)}{3!}
$$

**General Pattern:**

$$
a_n = \frac{f^{(n)}(0)}{n!}
$$

#### Example: eˣ

Since d/dx[eˣ] = eˣ and e⁰ = 1:

All derivatives at x = 0 equal 1.

$$
e^x = \sum_{n=0}^{\infty} \frac{1}{n!} x^n = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \ldots
$$

**Verification:** Differentiating term by term:

$$
\frac{d}{dx}\left[1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots\right] = 0 + 1 + x + \frac{x^2}{2!} + \ldots = e^x
$$

The series equals its own derivative!

#### Example: cos(x)

**Derivatives at x = 0:**

| n | f⁽ⁿ⁾(x) | f⁽ⁿ⁾(0) |
|---|---------|---------|
| 0 | cos(x) | 1 |
| 1 | -sin(x) | 0 |
| 2 | -cos(x) | -1 |
| 3 | sin(x) | 0 |
| 4 | cos(x) | 1 |

**Pattern:** Nonzero only for even n, alternating between 1 and -1.

$$
\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!} x^{2n} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \ldots
$$

**No x terms with odd powers!** This reflects that cosine is an even function: cos(-x) = cos(x).

#### Example: sin(x)

By similar analysis:

$$
\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \ldots
$$

**Only odd powers!** Sine is an odd function: sin(-x) = -sin(x).

#### Convergence

A series **converges** if the partial sums approach a finite limit.

For eˣ, sin(x), cos(x): The series converge for all real x.

But not all functions have globally convergent series...

---

### 4.4 The Taylor Series

The **Taylor series** generalizes to expansion around any point p.

#### The Formula

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(p)}{n!} (x - p)^n
$$

$$
= f(p) + f'(p)(x-p) + \frac{f''(p)}{2!}(x-p)^2 + \frac{f'''(p)}{3!}(x-p)^3 + \ldots
$$

Note: Setting p = 0 recovers the Maclaurin series.

#### Derivation of the First-Order Term

We want a line passing through (p, f(p)) with slope f'(p).

Line equation: y = mx + c where m = f'(p).

Using point (p, f(p)):

$$
f(p) = f'(p) \cdot p + c
$$

$$
c = f(p) - f'(p) \cdot p
$$

Substituting back:

$$
y = f'(p) \cdot x + f(p) - f'(p) \cdot p = f(p) + f'(p)(x - p)
$$

This is the first-order Taylor approximation!

#### Example: Expanding 1/x around x = 1

The function 1/x is undefined at x = 0, so we can't use a Maclaurin series. Let's expand around x = 1.

**Derivatives:**

$$
f(x) = x^{-1} \implies f(1) = 1
$$

$$
f'(x) = -x^{-2} \implies f'(1) = -1
$$

$$
f''(x) = 2x^{-3} \implies f''(1) = 2
$$

$$
f'''(x) = -6x^{-4} \implies f'''(1) = -6
$$

**Pattern:** f⁽ⁿ⁾(1) = (-1)ⁿ n!

**The Series:**

$$
\frac{1}{x} = \sum_{n=0}^{\infty} (-1)^n (x-1)^n = 1 - (x-1) + (x-1)^2 - (x-1)^3 + \ldots
$$

**Convergence:** This series only converges for |x - 1| < 1, i.e., 0 < x < 2.

Outside this interval, the partial sums diverge wildly!

#### Why Some Series Have Limited Convergence

The series for 1/x can't converge past x = 0 because the function has a singularity there. The radius of convergence equals the distance to the nearest singularity (even in the complex plane).

---

### 4.5 Linearization and Error Analysis

#### Alternative Notation

Using Δx for a small step from point x:

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x
$$

**Interpretation:** The change in f approximately equals the gradient times the step size.

#### Truncation Error

When we truncate the Taylor series, we introduce error. The **first-order approximation** has error proportional to (Δx)²:

$$
f(x + \Delta x) = f(x) + f'(x)\Delta x + \frac{f''(\xi)}{2}(\Delta x)^2
$$

for some ξ between x and x + Δx.

We write this as:

$$
f(x + \Delta x) = f(x) + f'(x)\Delta x + O((\Delta x)^2)
$$

This is **second-order accurate** — the error decreases quadratically as the step size shrinks.

#### Error in Finite Differences

Rearranging the Taylor series:

$$
f'(x) = \frac{f(x + \Delta x) - f(x)}{\Delta x} - \frac{f''(\xi)}{2}\Delta x
$$

The **forward difference** has error O(Δx) — first-order accurate.

For the **central difference**:

$$
f'(x) = \frac{f(x + \Delta x) - f(x - \Delta x)}{2\Delta x} + O((\Delta x)^2)
$$

The central difference is second-order accurate!

#### Practical Implications

| Method | Formula | Error Order | Use Case |
|--------|---------|-------------|----------|
| Forward difference | (f(x+h) - f(x))/h | O(h) | Simple, one-sided |
| Backward difference | (f(x) - f(x-h))/h | O(h) | Boundary conditions |
| Central difference | (f(x+h) - f(x-h))/(2h) | O(h²) | Best accuracy |

---

### 4.6 Multivariate Taylor Series

#### First-Order Expansion in 2D

$$
f(x + \Delta x, y + \Delta y) \approx f(x, y) + \frac{\partial f}{\partial x}\Delta x + \frac{\partial f}{\partial y}\Delta y
$$

In vector notation:

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f \cdot \Delta\mathbf{x}
$$

#### Second-Order Expansion

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f \cdot \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T H \Delta\mathbf{x}
$$

Where H is the Hessian matrix.

**In 2D:**

$$
f(x + \Delta x, y + \Delta y) \approx f + f_x \Delta x + f_y \Delta y + \frac{1}{2}\left(f_{xx}(\Delta x)^2 + 2f_{xy}\Delta x \Delta y + f_{yy}(\Delta y)^2\right)
$$

#### Geometric Interpretation

- **Zeroth order:** Flat surface at height f(x, y)
- **First order:** Tangent plane (matches gradient)
- **Second order:** Paraboloid (matches curvature)

#### Application: Understanding Critical Points

At a critical point where ∇f = **0**:

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \frac{1}{2}\Delta\mathbf{x}^T H \Delta\mathbf{x}
$$

The Hessian determines the shape:
- **H positive definite:** Bowl shape → local minimum
- **H negative definite:** Upside-down bowl → local maximum
- **H indefinite:** Saddle shape → saddle point

---

## Module 5: Gradient-Based Optimization

This module presents practical algorithms for finding function minima.

---

### 5.1 The Newton-Raphson Method

#### The Problem

Find x* such that f(x*) = 0 (a root of the equation).

#### The Algorithm

Starting from initial guess x₀, iterate:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

#### Geometric Interpretation

1. Evaluate f and f' at current guess xₙ
2. Draw the tangent line at that point
3. Find where the tangent crosses zero
4. Use that as the next guess

The tangent line approximates f locally, and we solve the linear approximation instead of the original equation.

#### Derivation

The tangent line at (xₙ, f(xₙ)) with slope f'(xₙ):

$$
y = f(x_n) + f'(x_n)(x - x_n)
$$

Setting y = 0 and solving for x:

$$
0 = f(x_n) + f'(x_n)(x - x_n)
$$

$$
x = x_n - \frac{f(x_n)}{f'(x_n)}
$$

#### Detailed Example

Solve x³ - 2x + 2 = 0.

**Setup:**
- f(x) = x³ - 2x + 2
- f'(x) = 3x² - 2

**Starting Guess:** x₀ = -2 (we guess the root is negative)

**Iteration 1:**

$$
f(-2) = -8 + 4 + 2 = -2
$$

$$
f'(-2) = 12 - 2 = 10
$$

$$
x_1 = -2 - \frac{-2}{10} = -2 + 0.2 = -1.8
$$

**Iteration 2:**

$$
f(-1.8) = -5.832 + 3.6 + 2 = -0.232
$$

$$
f'(-1.8) = 9.72 - 2 = 7.72
$$

$$
x_2 = -1.8 - \frac{-0.232}{7.72} = -1.8 + 0.03 = -1.77
$$

**Iteration 3:**

$$
f(-1.77) \approx -0.0046
$$

$$
f'(-1.77) \approx 7.40
$$

$$
x_3 \approx -1.7693
$$

**After just 3 iterations:** Error < 0.00001

#### Convergence

Near a simple root, Newton-Raphson converges **quadratically** — the number of correct digits roughly doubles each iteration.

#### When It Fails

1. **Starting near a critical point:** If f'(xₙ) ≈ 0, the step f(xₙ)/f'(xₙ) becomes huge

2. **Cycling:** Some starting points lead to periodic orbits

3. **Wrong root:** May converge to a different root than intended

4. **No convergence:** For badly-behaved functions, may diverge

**Remedy:** Use a hybrid method that switches to bisection when Newton fails.

---

### 5.2 The Gradient Vector

The **gradient** is the key to multivariate optimization.

#### Definition

For f: ℝⁿ → ℝ:

$$
\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\\\ \frac{\partial f}{\partial x_2} \\\\ \vdots \\\\ \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

#### Key Properties

**1. Direction of Steepest Ascent**

The gradient points in the direction where f increases most rapidly.

**2. Magnitude Equals Steepest Rate**

$$
|\nabla f| = \max_{||\mathbf{u}|| = 1} \nabla f \cdot \mathbf{u}
$$

The length of the gradient equals the rate of change in the steepest direction.

**3. Perpendicular to Level Sets**

The gradient is perpendicular to contour lines (2D) or level surfaces (higher dimensions).

#### The Directional Derivative

The rate of change of f in direction **u** (unit vector):

$$
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = |\nabla f| \cos\theta
$$

Where θ is the angle between ∇f and **u**.

**Maximum:** When **u** = ∇f/|∇f| (parallel to gradient), D**u**f = |∇f|

**Zero:** When **u** ⊥ ∇f (perpendicular to gradient)

**Minimum:** When **u** = -∇f/|∇f| (opposite to gradient), D**u**f = -|∇f|

#### Example

For f(x, y) = x²y:

$$
\nabla f = \begin{pmatrix} 2xy \\\\ x^2 \end{pmatrix}
$$

At (1, 2):

$$
\nabla f(1, 2) = \begin{pmatrix} 4 \\\\ 1 \end{pmatrix}
$$

- Direction of steepest increase: (4, 1) (or normalized: (4, 1)/√17)
- Rate in that direction: |∇f| = √(16 + 1) = √17 ≈ 4.12
- Direction of steepest decrease: (-4, -1)

---

### 5.3 Gradient Descent

**Gradient descent** finds minima by following the gradient downhill.

#### The Algorithm

$$
\mathbf{x}_{n+1} = \mathbf{x}_n - \gamma \nabla f(\mathbf{x}_n)
$$

Where γ > 0 is the **learning rate** (step size).

#### Intuition: The Foggy Mountain

Imagine descending a mountain in dense fog:
- You can't see the whole landscape
- You can feel the slope beneath your feet
- You step in the steepest downhill direction
- Repeat until you reach a valley

#### The Learning Rate

**Too small:** Convergence is very slow

**Too large:** Overshoot, potentially diverge

**Just right:** Efficient convergence

#### Variants

**Batch Gradient Descent:**
- Use all training data to compute gradient
- Stable but slow for large datasets

**Stochastic Gradient Descent (SGD):**
- Use one random example per update
- Noisy but much faster per iteration

**Mini-Batch Gradient Descent:**
- Use a small batch (e.g., 32 examples)
- Balance of stability and speed

#### Challenges

1. **Local Minima:** May get stuck in a local minimum instead of global

2. **Saddle Points:** In high dimensions, saddle points are common; gradient is zero but it's not a minimum

3. **Ill-Conditioning:** If the function is much steeper in some directions than others, convergence is slow

4. **Choosing Learning Rate:** Too small = slow; too large = unstable

#### Advanced Methods

**Momentum:** Add a fraction of the previous step

$$
\mathbf{v}_{n+1} = \beta \mathbf{v}_n + \gamma \nabla f(\mathbf{x}_n)
$$

$$
\mathbf{x}_{n+1} = \mathbf{x}_n - \mathbf{v}_{n+1}
$$

**Adam:** Adaptive learning rates per parameter (most popular in deep learning)

**L-BFGS:** Uses Hessian approximation for faster convergence (good for smaller problems)

---

### 5.4 Lagrange Multipliers

**Lagrange multipliers** find extrema subject to constraints.

#### The Problem

Maximize or minimize f(**x**) subject to g(**x**) = c.

#### The Key Insight

At the constrained optimum, the gradient of f is parallel to the gradient of g.

**Why?**

If ∇f had any component along the constraint curve, we could move along the constraint and improve f. So at the optimum, the only component of ∇f is perpendicular to the constraint — which is the direction of ∇g.

#### The Method

Set up the equations:

$$
\nabla f = \lambda \nabla g
$$

$$
g(\mathbf{x}) = c
$$

Solve for **x** and λ.

#### Detailed Example

Maximize f(x, y) = x²y subject to x² + y² = a².

**Step 1: Compute gradients**

$$
\nabla f = \begin{pmatrix} 2xy \\\\ x^2 \end{pmatrix}, \quad \nabla g = \begin{pmatrix} 2x \\\\ 2y \end{pmatrix}
$$

**Step 2: Set up equations**

$$
2xy = \lambda \cdot 2x \quad \text{...(1)}
$$

$$
x^2 = \lambda \cdot 2y \quad \text{...(2)}
$$

$$
x^2 + y^2 = a^2 \quad \text{...(3)}
$$

**Step 3: Solve**

From (1): If x ≠ 0, then y = λ.

Substituting into (2): x² = 2λ² = 2y², so x = ±√2 · y.

Substituting into (3): 2y² + y² = 3y² = a², so y = ±a/√3.

**Step 4: Find all solutions**

$$
(x, y) = \left(\frac{a\sqrt{2}}{\sqrt{3}}, \frac{a}{\sqrt{3}}\right), \left(-\frac{a\sqrt{2}}{\sqrt{3}}, \frac{a}{\sqrt{3}}\right), \left(\frac{a\sqrt{2}}{\sqrt{3}}, -\frac{a}{\sqrt{3}}\right), \left(-\frac{a\sqrt{2}}{\sqrt{3}}, -\frac{a}{\sqrt{3}}\right)
$$

**Step 5: Evaluate f at each point**

For y > 0: f = x²y = 2y² · y = 2y³ = 2a³/(3√3)

For y < 0: f = -2a³/(3√3)

**Maximum:** f = 2a³/(3√3) at y > 0 points

**Minimum:** f = -2a³/(3√3) at y < 0 points

---

## Module 6: Regression and Least Squares Fitting

This module applies optimization to fitting functions to data.

---

### 6.1 Introduction to Data Fitting

#### The Goal

Given data points {(xᵢ, yᵢ)}ᵢ₌₁ⁿ, find parameters **a** for a model ŷ = f(x; **a**) that best fits the data.

#### The Data Science Workflow

1. **Collect data:** Measurements, experiments, observations

2. **Clean data:** Handle missing values, outliers, errors

3. **Explore data:** Visualize, compute statistics, identify patterns

4. **Choose a model:** Based on domain knowledge or data patterns

5. **Fit the model:** Find optimal parameters

6. **Validate:** Check if the fit makes sense

7. **Interpret:** Draw conclusions

#### Choosing a Model

**Physical knowledge:** If you know the underlying process, use the appropriate functional form.

**Pattern recognition:** If data looks linear, try a line; if curved, try a polynomial or other function.

**Flexibility vs. overfitting:** More parameters can fit better but may capture noise instead of signal.

---

### 6.2 Linear Regression — Complete Derivation

#### The Model

$$
\hat{y} = mx + c
$$

Where m is the slope and c is the intercept.

#### The Residual

For each data point (xᵢ, yᵢ):

$$
r_i = y_i - \hat{y}_i = y_i - (mx_i + c)
$$

This is the vertical distance from the point to the line.

#### The Cost Function (Chi-Squared)

$$
\chi^2 = \sum_{i=1}^n r_i^2 = \sum_{i=1}^n (y_i - mx_i - c)^2
$$

We minimize the sum of squared residuals.

**Why squared?**

1. Penalizes points above and below the line equally
2. Penalizes large deviations more than small ones
3. Results in a smooth, differentiable function
4. Has a unique minimum (convex)

#### Finding the Minimum

Set partial derivatives to zero:

$$
\frac{\partial \chi^2}{\partial m} = 0, \quad \frac{\partial \chi^2}{\partial c} = 0
$$

**Derivative with respect to c:**

$$
\frac{\partial \chi^2}{\partial c} = \sum_{i=1}^n 2(y_i - mx_i - c)(-1) = -2\sum_{i=1}^n (y_i - mx_i - c) = 0
$$

$$
\sum_{i=1}^n y_i - m\sum_{i=1}^n x_i - nc = 0
$$

$$
n\bar{y} - mn\bar{x} - nc = 0
$$

$$
c = \bar{y} - m\bar{x}
$$

**The regression line passes through (x̄, ȳ)!**

**Derivative with respect to m:**

$$
\frac{\partial \chi^2}{\partial m} = \sum_{i=1}^n 2(y_i - mx_i - c)(-x_i) = 0
$$

$$
\sum_{i=1}^n x_i(y_i - mx_i - c) = 0
$$

$$
\sum_{i=1}^n x_i y_i - m\sum_{i=1}^n x_i^2 - c\sum_{i=1}^n x_i = 0
$$

Substituting c = ȳ - mx̄:

$$
\sum_{i=1}^n x_i y_i - m\sum_{i=1}^n x_i^2 - (\bar{y} - m\bar{x})\sum_{i=1}^n x_i = 0
$$

$$
\sum_{i=1}^n x_i y_i - m\sum_{i=1}^n x_i^2 - n\bar{x}\bar{y} + mn\bar{x}^2 = 0
$$

$$
m\left(\sum_{i=1}^n x_i^2 - n\bar{x}^2\right) = \sum_{i=1}^n x_i y_i - n\bar{x}\bar{y}
$$

$$
m = \frac{\sum_{i=1}^n x_i y_i - n\bar{x}\bar{y}}{\sum_{i=1}^n x_i^2 - n\bar{x}^2} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

#### Summary of Formulas

**Slope:**

$$
m = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}
$$

**Intercept:**

$$
c = \bar{y} - m\bar{x}
$$

#### Uncertainty Estimates

The variance of the residuals:

$$
s^2 = \frac{1}{n-2}\sum_{i=1}^n (y_i - mx_i - c)^2
$$

**Uncertainty in slope:**

$$
\sigma_m = s \sqrt{\frac{1}{\sum_i (x_i - \bar{x})^2}}
$$

**Uncertainty in intercept:**

$$
\sigma_c = s \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{\sum_i (x_i - \bar{x})^2}}
$$

#### Anscombe's Quartet — A Warning

Four datasets with identical:
- Mean of x and y
- Variance of x and y
- Correlation
- Regression line
- R²

But very different appearances!

**Lesson:** ALWAYS visualize your data. Statistics can hide important patterns.

---

### 6.3 Generalized Nonlinear Least Squares

#### The General Problem

Fit a model y = f(x; **a**) where **a** = (a₁, a₂, ..., aₘ) are parameters, possibly appearing nonlinearly.

#### The Cost Function

$$
\chi^2(\mathbf{a}) = \sum_{i=1}^n \frac{(y_i - f(x_i; \mathbf{a}))^2}{\sigma_i^2}
$$

Where σᵢ is the uncertainty in measurement yᵢ. If unknown, set all σᵢ = 1.

#### Gradient Descent for Fitting

$$
\mathbf{a}_{k+1} = \mathbf{a}_k - \gamma \nabla_{\mathbf{a}} \chi^2
$$

**The gradient:**

$$
\frac{\partial \chi^2}{\partial a_j} = -2\sum_{i=1}^n \frac{(y_i - f(x_i; \mathbf{a}))}{\sigma_i^2} \cdot \frac{\partial f}{\partial a_j}
$$

**Algorithm:**

1. Choose initial parameter guess **a**₀
2. Compute ∇χ² at current **a**
3. Update: **a** ← **a** - γ∇χ²
4. Repeat until χ² stops decreasing (or gradient is small)

#### Example: Fitting a Gaussian

Model:

$$
f(x; A, \mu, \sigma) = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

Parameters: A (amplitude), μ (mean), σ (width)

Partial derivatives:

$$
\frac{\partial f}{\partial A} = \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

$$
\frac{\partial f}{\partial \mu} = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \cdot \frac{x - \mu}{\sigma^2}
$$

$$
\frac{\partial f}{\partial \sigma} = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \cdot \frac{(x - \mu)^2}{\sigma^3}
$$

---

### 6.4 Practical Implementation and Advanced Methods

#### The Levenberg-Marquardt Algorithm

Combines gradient descent (far from minimum) with Newton's method (near minimum):

- Far from minimum: Take gradient descent steps
- Near minimum: Use Hessian information for faster convergence
- Automatically adjusts based on whether χ² is improving

#### Implementation in Python

```python
from scipy.optimize import curve_fit
import numpy as np

# Define the model
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Generate some data
x_data = np.linspace(-5, 5, 100)
y_data = gaussian(x_data, 2.5, 0.5, 1.2) + 0.2 * np.random.randn(100)

# Fit the model
params, covariance = curve_fit(gaussian, x_data, y_data, p0=[1, 0, 1])

# Extract results
A, mu, sigma = params
uncertainties = np.sqrt(np.diag(covariance))
```

#### Best Practices

1. **Good starting guess:** Essential for convergence
   - For Gaussian: Use peak location, peak height, estimated width

2. **Visualize the fit:** Always plot data and model together

3. **Check residuals:** Should be randomly distributed around zero

4. **Report uncertainties:** Parameters are meaningless without error estimates

5. **Consider model selection:** Maybe a different model fits better

---

## Complete Reference Tables

### Differentiation Rules

| Rule | Formula |
|------|---------|
| Sum | (d/dx)[f + g] = f' + g' |
| Constant Multiple | (d/dx)[cf] = cf' |
| Product | (d/dx)[fg] = fg' + gf' |
| Quotient | (d/dx)[f/g] = (gf' - fg')/g² |
| Chain | (d/dx)[f(g(x))] = f'(g(x)) · g'(x) |
| Power | (d/dx)[xⁿ] = nxⁿ⁻¹ |

### Common Derivatives

| Function | Derivative |
|----------|------------|
| c (constant) | 0 |
| xⁿ | nxⁿ⁻¹ |
| eˣ | eˣ |
| eᵏˣ | keᵏˣ |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |
| tan(x) | sec²(x) |
| arcsin(x) | 1/√(1-x²) |
| arccos(x) | -1/√(1-x²) |
| arctan(x) | 1/(1+x²) |

### Multivariate Concepts

| Concept | Definition | Dimension |
|---------|------------|-----------|
| Partial Derivative | ∂f/∂xᵢ | Scalar |
| Gradient | ∇f = (∂f/∂x₁, ..., ∂f/∂xₙ)ᵀ | n × 1 |
| Jacobian (scalar function) | Row vector of partials | 1 × n |
| Jacobian (vector function) | Jᵢⱼ = ∂fᵢ/∂xⱼ | m × n |
| Hessian | Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ | n × n |

### Taylor Series

**General form (around x = p):**

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(p)}{n!}(x-p)^n
$$

**Common series (around x = 0):**

| Function | Series |
|----------|--------|
| eˣ | Σ xⁿ/n! |
| sin(x) | Σ (-1)ⁿ x²ⁿ⁺¹/(2n+1)! |
| cos(x) | Σ (-1)ⁿ x²ⁿ/(2n)! |
| ln(1+x) | Σ (-1)ⁿ⁺¹ xⁿ/n |
| 1/(1-x) | Σ xⁿ |

### Optimization Algorithms

| Method | Update Rule | Use Case |
|--------|-------------|----------|
| Newton-Raphson | xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ) | Finding roots |
| Gradient Descent | **x**ₙ₊₁ = **x**ₙ - γ∇f | Finding minima |
| Lagrange Multipliers | ∇f = λ∇g | Constrained optimization |

### Critical Point Classification (2D)

| Determinant of H | Sign of H₁₁ | Type |
|------------------|-------------|------|
| > 0 | > 0 | Local minimum |
| > 0 | < 0 | Local maximum |
| < 0 | — | Saddle point |
| = 0 | — | Inconclusive |

---

## Key Takeaways for Machine Learning

1. **Derivatives measure sensitivity** — how outputs change with inputs/parameters

2. **The chain rule is backpropagation** — the fundamental algorithm for training neural networks

3. **The gradient points uphill** — gradient descent goes downhill to minimize loss

4. **The Hessian describes curvature** — helps understand optimization difficulty

5. **Taylor series enable approximation** — linearization is the foundation of many numerical methods

6. **Least squares is optimization** — training = minimizing a cost function

7. **Always visualize** — mathematics can hide data pathologies

---

> **Congratulations!**
>
> You now possess the mathematical foundation to understand how machine learning algorithms work at a deep level. The concepts of gradients, optimization, and function approximation appear throughout deep learning, and this calculus toolkit will serve you well as you dive deeper into the field.
