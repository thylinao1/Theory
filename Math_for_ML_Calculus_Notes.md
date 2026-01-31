# Mathematics for Machine Learning: Multivariate Calculus

> **Complete Course Notes**  
> A comprehensive guide to calculus concepts essential for machine learning, covering differentiation, multivariate systems, optimization, neural networks, Taylor series, and regression.

---

## Table of Contents

1. [Module 1: Foundations of Calculus](#module-1-foundations-of-calculus)
   - [1.1 What Are Functions?](#11-what-are-functions)
   - [1.2 Gradients and Derivatives](#12-gradients-and-derivatives)
   - [1.3 The Formal Definition of a Derivative](#13-the-formal-definition-of-a-derivative)
   - [1.4 The Sum Rule](#14-the-sum-rule)
   - [1.5 The Power Rule](#15-the-power-rule)
   - [1.6 Special Functions](#16-special-functions)
   - [1.7 The Product Rule](#17-the-product-rule)
   - [1.8 The Chain Rule](#18-the-chain-rule)
2. [Module 2: Multivariate Calculus](#module-2-multivariate-calculus)
   - [2.1 Variables, Constants, and Context](#21-variables-constants-and-context)
   - [2.2 Partial Differentiation](#22-partial-differentiation)
   - [2.3 The Total Derivative](#23-the-total-derivative)
   - [2.4 The Jacobian Vector](#24-the-jacobian-vector)
   - [2.5 The Jacobian Matrix](#25-the-jacobian-matrix)
   - [2.6 The Hessian Matrix](#26-the-hessian-matrix)
3. [Module 3: Optimization and Neural Networks](#module-3-optimization-and-neural-networks)
   - [3.1 Introduction to Optimization](#31-introduction-to-optimization)
   - [3.2 Neural Network Fundamentals](#32-neural-network-fundamentals)
   - [3.3 Backpropagation](#33-backpropagation)
4. [Module 4: Taylor Series and Approximations](#module-4-taylor-series-and-approximations)
   - [4.1 Motivation for Approximations](#41-motivation-for-approximations)
   - [4.2 Power Series Intuition](#42-power-series-intuition)
   - [4.3 The Maclaurin Series](#43-the-maclaurin-series)
   - [4.4 The Taylor Series](#44-the-taylor-series)
   - [4.5 Linearization and Error Analysis](#45-linearization-and-error-analysis)
   - [4.6 Multivariate Taylor Series](#46-multivariate-taylor-series)
5. [Module 5: Gradient-Based Optimization](#module-5-gradient-based-optimization)
   - [5.1 The Newton-Raphson Method](#51-the-newton-raphson-method)
   - [5.2 The Gradient Vector (Grad)](#52-the-gradient-vector-grad)
   - [5.3 Gradient Descent](#53-gradient-descent)
   - [5.4 Lagrange Multipliers](#54-lagrange-multipliers)
6. [Module 6: Regression and Least Squares](#module-6-regression-and-least-squares)
   - [6.1 Introduction to Data Fitting](#61-introduction-to-data-fitting)
   - [6.2 Linear Regression](#62-linear-regression)
   - [6.3 Nonlinear Least Squares](#63-nonlinear-least-squares)
   - [6.4 Practical Implementation](#64-practical-implementation)

---

## Module 1: Foundations of Calculus

This module establishes the fundamental theory of calculus, focusing on derivatives and the rules that make differentiation practical.

### 1.1 What Are Functions?

A **function** is a relationship between inputs and an output. Functions are the mathematical models we use to describe real-world phenomena.

**Example: Temperature in a Room**

If we want to model temperature distribution, we might define:

$$
T = f(x, y, z, t)
$$

This function takes spatial coordinates $(x, y, z)$ and time $t$ as inputs, returning the temperature at that specific point and moment.

**Notation Conventions**

The expression $f(x) = x^2 + 3$ means "$f$ is a function of $x$." While it might look like multiplication, $f(x)$ denotes function application, not $f \times x$. This distinction becomes important when expressions contain multiple bracket terms.

**Why This Matters for ML:** Machine learning models are essentially complex functions. A neural network is a function that maps inputs (like pixels) to outputs (like class probabilities). Understanding function behavior is fundamental to understanding how these models learn.

---

### 1.2 Gradients and Derivatives

**The Core Idea:** Calculus is the study of how functions change with respect to their input variables.

**Speed-Time Graph Example**

Consider a car's speed over time:

```
Speed (v)
    │     ╭──╮
    │    ╱    ╲
    │   ╱      ╲
    │  ╱        ╲
    │ ╱          ╲
    └──────────────── Time (t)
```

Key observations:
- A **horizontal line** implies constant speed (zero acceleration)
- An **upward slope** indicates acceleration
- A **downward slope** indicates deceleration

**Gradient as Acceleration**

The **gradient** (or slope) at any point represents the rate of change:

$$
\text{Acceleration} = \frac{\text{Change in Speed}}{\text{Change in Time}} = \frac{\Delta v}{\Delta t}
$$

The **tangent line** at any point is a straight line that touches the curve and shares the same gradient as the curve at that point.

**Derivative as a New Function**

By recording the gradient at every point, we construct a new function — the **derivative**. If the original function describes speed vs. time, the derivative describes acceleration vs. time.

**Higher-Order Derivatives**

We can continue this process:
- **First derivative of position** → Velocity
- **Second derivative of position** → Acceleration  
- **Third derivative of position** → Jerk (rate of change of acceleration)

**Anti-Derivative (Integration)**

The reverse process — finding a function whose derivative gives us our current function — is called the **anti-derivative** or **integral**. For our example, the anti-derivative of speed gives us distance traveled.

---

### 1.3 The Formal Definition of a Derivative

The derivative formalizes our intuitive understanding of "rise over run."

**For a Linear Function**

$$
\text{Gradient} = \frac{\text{Rise}}{\text{Run}} = \frac{\Delta y}{\Delta x}
$$

**For Nonlinear Functions**

For curves where the gradient varies, we pick a point $x$ and a nearby point $x + \Delta x$:

$$
\text{Approximate Gradient} = \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

**The Limit Definition**

As $\Delta x$ approaches zero, our approximation becomes exact:

$$
f'(x) = \frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}
$$

This is the **formal definition of a derivative**.

**Notation Styles:**
- $f'(x)$ — Lagrange notation (prime notation)
- $\frac{df}{dx}$ — Leibniz notation (ratio notation)

**Example: Differentiating $f(x) = 3x + 2$**

$$
f'(x) = \lim_{\Delta x \to 0} \frac{[3(x + \Delta x) + 2] - [3x + 2]}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{3x + 3\Delta x + 2 - 3x - 2}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{3\Delta x}{\Delta x} = 3
$$

The derivative of $f(x) = 3x + 2$ is simply $3$, confirming that linear functions have constant gradients.

---

### 1.4 The Sum Rule

When differentiating a sum of functions, we can differentiate each part separately and add the results.

**The Rule:**

$$
\frac{d}{dx}[f(x) + g(x)] = \frac{df}{dx} + \frac{dg}{dx}
$$

**Why It Works:** Differentiation is a linear operation. The limit of a sum equals the sum of limits.

**Example:**

$$
\frac{d}{dx}[x^2 + 5x + 3] = \frac{d}{dx}[x^2] + \frac{d}{dx}[5x] + \frac{d}{dx}[3]
$$

---

### 1.5 The Power Rule

The power rule handles polynomial terms efficiently.

**The Rule:**

If $f(x) = ax^b$, then:

$$
f'(x) = abx^{b-1}
$$

The original power multiplies the coefficient, then the power decreases by one.

**Derivation for $f(x) = 5x^2$:**

$$
f'(x) = \lim_{\Delta x \to 0} \frac{5(x + \Delta x)^2 - 5x^2}{\Delta x}
$$

Expanding $(x + \Delta x)^2 = x^2 + 2x\Delta x + (\Delta x)^2$:

$$
= \lim_{\Delta x \to 0} \frac{5x^2 + 10x\Delta x + 5(\Delta x)^2 - 5x^2}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{10x\Delta x + 5(\Delta x)^2}{\Delta x}
$$

$$
= \lim_{\Delta x \to 0} [10x + 5\Delta x] = 10x
$$

**Examples:**

| Function | Derivative |
|----------|------------|
| $x^3$ | $3x^2$ |
| $4x^5$ | $20x^4$ |
| $x^{-1} = \frac{1}{x}$ | $-x^{-2} = -\frac{1}{x^2}$ |
| $x^{1/2} = \sqrt{x}$ | $\frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$ |

---

### 1.6 Special Functions

Some functions have unique differentiation properties that make them invaluable in calculus.

#### The Function $f(x) = \frac{1}{x}$

$$
f'(x) = -\frac{1}{x^2}
$$

**Derivation:**

$$
f'(x) = \lim_{\Delta x \to 0} \frac{\frac{1}{x+\Delta x} - \frac{1}{x}}{\Delta x}
$$

Combining fractions:

$$
= \lim_{\Delta x \to 0} \frac{x - (x + \Delta x)}{x(x + \Delta x) \cdot \Delta x}
$$

$$
= \lim_{\Delta x \to 0} \frac{-\Delta x}{x(x + \Delta x) \cdot \Delta x} = \lim_{\Delta x \to 0} \frac{-1}{x(x + \Delta x)}
$$

$$
= \frac{-1}{x^2}
$$

**Note:** Both the function and its derivative are undefined at $x = 0$ (discontinuity).

#### The Exponential Function $e^x$

The exponential function has a magical property: **it equals its own derivative**.

$$
\frac{d}{dx}[e^x] = e^x
$$

**Euler's Number $e \approx 2.718$** is one of the most important constants in mathematics, appearing throughout calculus, probability, and physics.

**Why This Matters:** This self-similarity property makes $e^x$ incredibly useful for modeling growth and decay processes, and it simplifies many calculations in neural networks.

#### Trigonometric Functions

The derivatives of sine and cosine form a cyclic pattern:

$$
\frac{d}{dx}[\sin x] = \cos x
$$

$$
\frac{d}{dx}[\cos x] = -\sin x
$$

$$
\frac{d}{dx}[-\sin x] = -\cos x
$$

$$
\frac{d}{dx}[-\cos x] = \sin x
$$

After four differentiations, we return to the original function!

**The Cycle:**

```
sin(x) → cos(x) → -sin(x) → -cos(x) → sin(x)
```

---

### 1.7 The Product Rule

When differentiating a product of two functions, we use the product rule.

**The Rule:**

If $A(x) = f(x) \cdot g(x)$, then:

$$
A'(x) = f(x) \cdot g'(x) + g(x) \cdot f'(x)
$$

**Geometric Intuition:**

Imagine a rectangle with sides $f(x)$ and $g(x)$. The area is $A = f \cdot g$.

When $x$ increases by $\Delta x$:
- Side $f$ changes by $\Delta f$
- Side $g$ changes by $\Delta g$

The new area includes:
1. The original area: $f \cdot g$
2. A strip of width $f$ and height $\Delta g$
3. A strip of width $\Delta f$ and height $g$
4. A tiny corner of area $\Delta f \cdot \Delta g$ (negligible as $\Delta x \to 0$)

The change in area is approximately:

$$
\Delta A \approx f \cdot \Delta g + g \cdot \Delta f
$$

Dividing by $\Delta x$ and taking the limit gives us the product rule.

**Example:** Differentiate $f(x) = x^2 \cdot \sin(x)$

$$
f'(x) = x^2 \cdot \cos(x) + \sin(x) \cdot 2x = x^2\cos(x) + 2x\sin(x)
$$

---

### 1.8 The Chain Rule

The chain rule handles **nested functions** — functions of functions.

**The Rule:**

If $h(x) = f(g(x))$, then:

$$
h'(x) = f'(g(x)) \cdot g'(x)
$$

Or in Leibniz notation:

$$
\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

**The Happiness-Pizza-Money Example**

Let's relate happiness $h$ to money $m$ through pizza $p$:

- $h(p) = -\frac{1}{3}p^2 + p + \frac{1}{5}$ — Happiness as a function of pizza
- $p(m) = e^m - 1$ — Pizza as a function of money

To find $\frac{dh}{dm}$ (how happiness changes with money):

**Step 1:** Find individual derivatives:

$$
\frac{dh}{dp} = -\frac{2}{3}p + 1
$$

$$
\frac{dp}{dm} = e^m
$$

**Step 2:** Apply the chain rule:

$$
\frac{dh}{dm} = \frac{dh}{dp} \cdot \frac{dp}{dm} = \left(-\frac{2}{3}p + 1\right) \cdot e^m
$$

**Step 3:** Substitute $p = e^m - 1$:

$$
\frac{dh}{dm} = \left(-\frac{2}{3}(e^m - 1) + 1\right) \cdot e^m = \frac{1}{3}e^m(5 - 2e^m)
$$

**Why It Matters for ML:** Neural networks are compositions of many functions (layers). Backpropagation uses the chain rule to compute gradients through these nested functions.

---

### Comprehensive Differentiation Example

Let's differentiate a complex function using all four rules:

$$
f(x) = \frac{\sin(2x^5 + 3x)}{e^{7x}}
$$

**Step 1:** Rewrite as a product:

$$
f(x) = \sin(2x^5 + 3x) \cdot e^{-7x}
$$

**Step 2:** Define the components:

- $g(x) = \sin(2x^5 + 3x)$
- $h(x) = e^{-7x}$

**Step 3:** Differentiate $g(x)$ using the chain rule:

Let $u = 2x^5 + 3x$, so $g = \sin(u)$

$$
\frac{dg}{du} = \cos(u)
$$

$$
\frac{du}{dx} = 10x^4 + 3
$$

$$
g'(x) = \cos(2x^5 + 3x) \cdot (10x^4 + 3)
$$

**Step 4:** Differentiate $h(x)$ using the chain rule:

$$
h'(x) = e^{-7x} \cdot (-7) = -7e^{-7x}
$$

**Step 5:** Apply the product rule:

$$
f'(x) = g(x) \cdot h'(x) + h(x) \cdot g'(x)
$$

$$
= \sin(2x^5 + 3x) \cdot (-7e^{-7x}) + e^{-7x} \cdot \cos(2x^5 + 3x) \cdot (10x^4 + 3)
$$

---

## Calculus Toolbox Summary — Module 1

| Rule | Formula | When to Use |
|------|---------|-------------|
| **Sum Rule** | $\frac{d}{dx}[f + g] = f' + g'$ | Adding/subtracting functions |
| **Power Rule** | $\frac{d}{dx}[ax^b] = abx^{b-1}$ | Polynomial terms |
| **Product Rule** | $\frac{d}{dx}[f \cdot g] = fg' + gf'$ | Multiplying functions |
| **Chain Rule** | $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ | Nested/composite functions |

**Special Derivatives:**

| Function | Derivative |
|----------|------------|
| $e^x$ | $e^x$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\frac{1}{x}$ | $-\frac{1}{x^2}$ |

---

## Module 2: Multivariate Calculus

This module extends differentiation to functions of multiple variables, introducing concepts essential for navigating high-dimensional optimization landscapes.

### 2.1 Variables, Constants, and Context

Before diving into multivariate calculus, we must understand how context determines what's a "variable" versus a "constant."

**The Key Insight:** What we label as a variable or constant depends on the problem we're solving.

**Example: Car Force Equation**

$$
F = ma + dv^2
$$

Where:
- $F$ = Force from engine
- $m$ = Mass of car
- $a$ = Acceleration
- $d$ = Aerodynamic drag coefficient
- $v$ = Velocity

**Context 1: Driver's Perspective**

If you're driving:
- **Independent variable:** $F$ (you control the accelerator)
- **Dependent variables:** $a$, $v$ (consequences of your force)
- **Constants:** $m$, $d$ (fixed car properties)

**Context 2: Car Designer's Perspective**

If you're designing a car fleet:
- **Independent variable:** $F$ (engine design choice)
- **Parameters to optimize:** $m$, $d$ (you can redesign these)
- **Constants:** $a$, $v$ (target performance specs)

**Parameters** are variable-like constants — we might vary them to explore a family of similar functions, or optimize them to fit data.

**Why This Matters for ML:** When training a model, the **weights** are parameters we optimize, while the **inputs** are variables the model processes. Understanding this distinction is crucial for implementing learning algorithms.

---

### 2.2 Partial Differentiation

When a function depends on multiple variables, we differentiate with respect to one variable while treating all others as constants.

**The Concept:**

For $f(x, y, z)$, the **partial derivative** with respect to $x$ is:

$$
\frac{\partial f}{\partial x}
$$

The curly $\partial$ symbol (partial) indicates we're holding other variables constant.

**Example: Metal Can Manufacturing**

The mass of a cylindrical can:

$$
m = (2\pi r^2 + 2\pi rh) \cdot t \cdot \rho
$$

Where:
- $r$ = radius
- $h$ = height
- $t$ = wall thickness
- $\rho$ = metal density

**Finding Partial Derivatives:**

**With respect to height $h$:**

$$
\frac{\partial m}{\partial h} = 2\pi r t \rho
$$

(The first term doesn't contain $h$, so it vanishes. The second term has $h$ multiplied by constants.)

**With respect to radius $r$:**

$$
\frac{\partial m}{\partial r} = 4\pi r t \rho + 2\pi h t \rho
$$

(Apply the power rule to $r^2$ in the first term, and treat $h$ as constant in the second.)

**With respect to thickness $t$:**

$$
\frac{\partial m}{\partial t} = 2\pi r^2 \rho + 2\pi rh\rho
$$

**With respect to density $\rho$:**

$$
\frac{\partial m}{\partial \rho} = 2\pi r^2 t + 2\pi rht
$$

**More Complex Example:**

For $f(x, y, z) = \sin(x) \cdot e^{yz^2}$:

$$
\frac{\partial f}{\partial x} = \cos(x) \cdot e^{yz^2}
$$

$$
\frac{\partial f}{\partial y} = \sin(x) \cdot e^{yz^2} \cdot z^2
$$

$$
\frac{\partial f}{\partial z} = \sin(x) \cdot e^{yz^2} \cdot 2yz
$$

---

### 2.3 The Total Derivative

When all variables depend on a single parameter, we can compute how the function changes with that parameter.

**Setup:**

Given $f(x, y, z)$ where:
- $x = x(t)$
- $y = y(t)$
- $z = z(t)$

**The Total Derivative:**

$$
\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} + \frac{\partial f}{\partial z}\frac{dz}{dt}
$$

This is the **multivariate chain rule** — we sum the contributions from each variable.

**Example:**

Let $f(x, y, z) = \sin(x) \cdot e^{yz^2}$ with:
- $x = t - 1$
- $y = t^2$
- $z = \frac{1}{t}$

We found the partial derivatives above. Now find:

$$
\frac{dx}{dt} = 1, \quad \frac{dy}{dt} = 2t, \quad \frac{dz}{dt} = -\frac{1}{t^2}
$$

Then:

$$
\frac{df}{dt} = \cos(x) \cdot e^{yz^2} \cdot 1 + \sin(x) \cdot e^{yz^2} \cdot z^2 \cdot 2t + \sin(x) \cdot e^{yz^2} \cdot 2yz \cdot \left(-\frac{1}{t^2}\right)
$$

After substituting $x = t-1$, $y = t^2$, $z = \frac{1}{t}$ and simplifying (the second and third terms cancel), we get:

$$
\frac{df}{dt} = \cos(t-1) \cdot e
$$

---

### 2.4 The Jacobian Vector

The **Jacobian** collects all partial derivatives into a single vector, pointing in the direction of steepest ascent.

**Definition:**

For a scalar function $f(x_1, x_2, \ldots, x_n)$:

$$
J = \nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} & \cdots & \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

By convention, this is written as a **row vector**.

**Example:**

For $f(x, y, z) = x^2y + 3z$:

$$
\frac{\partial f}{\partial x} = 2xy, \quad \frac{\partial f}{\partial y} = x^2, \quad \frac{\partial f}{\partial z} = 3
$$

$$
J = \begin{pmatrix} 2xy & x^2 & 3 \end{pmatrix}
$$

**Evaluating at a Point:**

At $(0, 0, 0)$:

$$
J(0, 0, 0) = \begin{pmatrix} 0 & 0 & 3 \end{pmatrix}
$$

This vector points in the $z$-direction, meaning at the origin, the steepest ascent is purely in $z$.

**Key Properties:**

1. **Direction:** The Jacobian points in the direction of steepest increase
2. **Magnitude:** The length of the Jacobian equals the steepness of that slope
3. **Perpendicular to Contours:** The Jacobian is always perpendicular to level curves/surfaces

**Visualization:**

On a contour plot (like a topographic map):
- Contour lines show regions of equal function value
- Jacobian vectors point uphill, perpendicular to contour lines
- Where contours are tightly packed → large Jacobian magnitude (steep slope)
- Where contours are spread out → small Jacobian magnitude (gentle slope)

---

### 2.5 The Jacobian Matrix

When both inputs and outputs are vectors, the Jacobian becomes a **matrix**.

**Setup:**

Given vector-valued function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$:

$$
\mathbf{F}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\\ f_2(x_1, \ldots, x_n) \\\ \vdots \\\ f_m(x_1, \ldots, x_n) \end{pmatrix}
$$

**The Jacobian Matrix:**

$$
J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\\ \vdots & \vdots & \ddots & \vdots \\\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}
$$

Each row is the Jacobian of one output function.

**Example: Linear Transformation**

Consider:
- $u(x, y) = x + 2y$
- $v(x, y) = 3y - 2x$

$$
J = \begin{pmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{pmatrix} = \begin{pmatrix} 1 & 2 \\\ -2 & 3 \end{pmatrix}
$$

For linear functions, the Jacobian is constant — it's simply the transformation matrix!

**Application: Coordinate Transformations**

Converting from polar $(r, \theta)$ to Cartesian $(x, y)$:

$$
x = r\cos(\theta), \quad y = r\sin(\theta)
$$

$$
J = \begin{pmatrix} \cos\theta & -r\sin\theta \\\ \sin\theta & r\cos\theta \end{pmatrix}
$$

The **determinant** $|J| = r$ tells us how areas scale under this transformation. This is why $r$ appears in polar integration!

---

### 2.6 The Hessian Matrix

The **Hessian** collects all second-order partial derivatives, providing information about curvature.

**Definition:**

For $f(x_1, x_2, \ldots, x_n)$:

$$
H = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\\ \vdots & \vdots & \ddots & \vdots \\\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}
$$

**Key Property:** For continuous functions, mixed partials are equal: $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$

This means the Hessian is **symmetric**.

**Example:**

For $f(x, y, z) = x^2yz$:

First, find the Jacobian:

$$
J = \begin{pmatrix} 2xyz & x^2z & x^2y \end{pmatrix}
$$

Then differentiate each component again:

$$
H = \begin{pmatrix} 2yz & 2xz & 2xy \\\ 2xz & 0 & x^2 \\\ 2xy & x^2 & 0 \end{pmatrix}
$$

**Classifying Critical Points:**

When the Jacobian is zero (critical point), the Hessian tells us what type of point we have:

| Determinant of H | Top-left element | Classification |
|------------------|------------------|----------------|
| $> 0$ | $> 0$ | **Local minimum** |
| $> 0$ | $< 0$ | **Local maximum** |
| $< 0$ | — | **Saddle point** |
| $= 0$ | — | Inconclusive |

**Example: Classifying $f(x, y) = x^2 + y^2$**

$$
J = \begin{pmatrix} 2x & 2y \end{pmatrix} = \mathbf{0} \text{ at } (0, 0)
$$

$$
H = \begin{pmatrix} 2 & 0 \\\ 0 & 2 \end{pmatrix}
$$

- $\det(H) = 4 > 0$
- Top-left element $= 2 > 0$

**Conclusion:** $(0, 0)$ is a **local minimum**.

**Saddle Point Example: $f(x, y) = x^2 - y^2$**

$$
H = \begin{pmatrix} 2 & 0 \\\ 0 & -2 \end{pmatrix}
$$

- $\det(H) = -4 < 0$

**Conclusion:** $(0, 0)$ is a **saddle point** — a minimum in the $x$-direction but a maximum in the $y$-direction.

---

## Module 3: Optimization and Neural Networks

This module connects calculus concepts to neural network training through the optimization lens.

### 3.1 Introduction to Optimization

**The Goal:** Find input values that maximize or minimize a function.

**Real-World Applications:**
- Route planning through cities
- Factory production scheduling
- Stock portfolio selection
- **Training machine learning models**

**The Challenge:**

For simple functions, we can set the gradient to zero and solve analytically:

$$
\nabla f = \mathbf{0}
$$

For complex functions with multiple variables, this becomes impractical. We need **numerical methods**.

**The Sandpit Analogy:**

Imagine finding the deepest point in a sandpit by probing with a stick:
- You can measure depth at any point
- You can't see the bottom
- You can't move the stick sideways underground

This is analogous to optimization when:
- We can evaluate the function at any point
- We can't visualize the entire landscape
- Each evaluation may be expensive

**Key Challenges:**

1. **Local vs. Global Optima:** A local minimum might not be the global minimum
2. **Saddle Points:** Flat regions that aren't actually optima
3. **High Dimensionality:** Can't visualize beyond 3D
4. **Noisy Data:** Gradients may be unreliable
5. **Discontinuities:** Sharp features confuse gradient methods

---

### 3.2 Neural Network Fundamentals

A neural network is a mathematical function built from layers of simpler functions.

**The Simplest Network:**

Single input $a_0$, single output $a_1$:

$$
a_1 = \sigma(wa_0 + b)
$$

Where:
- $a$ = **activations** (input/output values)
- $w$ = **weight** (scales the input)
- $b$ = **bias** (shifts the result)
- $\sigma$ = **activation function** (introduces nonlinearity)

**Activation Functions:**

The activation function gives neural networks their power. Common choices:

**Sigmoid / Logistic:**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Hyperbolic Tangent:**
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

Both have an S-shape, outputting values between fixed bounds. This mimics biological neurons that "fire" when stimulation exceeds a threshold.

**Scaling Up — Multiple Inputs:**

With $n$ inputs:

$$
a_1 = \sigma\left(\sum_{i=1}^{n} w_i a_{0i} + b\right) = \sigma(\mathbf{w} \cdot \mathbf{a}_0 + b)
$$

**Scaling Up — Multiple Outputs:**

With $m$ outputs:

$$
\mathbf{a}_1 = \sigma(W\mathbf{a}_0 + \mathbf{b})
$$

Where:
- $W$ is an $m \times n$ **weight matrix**
- $\mathbf{b}$ is an $m$-dimensional **bias vector**
- $\sigma$ is applied element-wise

**Adding Hidden Layers:**

The magic of deep learning comes from stacking layers:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

Each layer's output becomes the next layer's input:

$$
\mathbf{a}_{l+1} = \sigma(W_l \mathbf{a}_l + \mathbf{b}_l)
$$

---

### 3.3 Backpropagation

**Backpropagation** is the algorithm for training neural networks by computing gradients efficiently.

**The Training Process:**

1. **Forward Pass:** Compute network output for training input
2. **Compute Cost:** Measure error between output and desired label
3. **Backward Pass:** Compute gradients of cost with respect to all weights/biases
4. **Update Parameters:** Adjust weights/biases to reduce cost

**The Cost Function:**

$$
C = \sum_i (y_i - \hat{y}_i)^2
$$

Where $y_i$ is the desired output and $\hat{y}_i$ is the network's prediction.

**Why Backpropagation Works:**

The network is a composition of functions. The chain rule lets us compute:

$$
\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a_{\text{out}}} \cdot \frac{\partial a_{\text{out}}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Where $z = wa + b$ (pre-activation value).

**For a Simple Two-Node Network:**

$$
\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w}
$$

$$
\frac{\partial C}{\partial b} = \frac{\partial C}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial b}
$$

**The Gradient Vector:**

We collect all derivatives into a gradient vector (the Jacobian of $C$ with respect to parameters):

$$
\nabla C = \begin{pmatrix} \frac{\partial C}{\partial w_1} & \frac{\partial C}{\partial w_2} & \cdots & \frac{\partial C}{\partial b_1} & \frac{\partial C}{\partial b_2} & \cdots \end{pmatrix}
$$

**The Update Rule:**

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla C
$$

Where $\theta$ represents all parameters and $\eta$ is the **learning rate**.

---

## Module 4: Taylor Series and Approximations

Taylor series allow us to approximate complex functions with polynomials, connecting back to our "rise over run" intuition.

### 4.1 Motivation for Approximations

**The Chicken Cooking Problem:**

Suppose we have a complex function for cooking time $T$ based on chicken mass $m$:

$$
T = f(m, \text{oven type}, \text{heat properties}, \ldots)
$$

For a cookbook, we need a simple formula. Taylor series let us approximate this monster with something practical.

**When Approximations Are Useful:**

1. **Simplification:** Complex functions → manageable polynomials
2. **Numerical Methods:** Computers work with finite precision
3. **Analysis:** Understanding function behavior near a point
4. **Efficiency:** Faster computation than evaluating complex functions

---

### 4.2 Power Series Intuition

A **power series** expresses a function as a sum of increasing powers:

$$
g(x) = a + bx + cx^2 + dx^3 + \cdots
$$

**Building Approximations:**

**Zeroth Order:** Match the function value (horizontal line)

$$
g_0(x) = f(p)
$$

**First Order:** Also match the gradient (tangent line)

$$
g_1(x) = f(p) + f'(p)(x - p)
$$

**Second Order:** Also match the curvature (parabola)

$$
g_2(x) = f(p) + f'(p)(x - p) + \frac{f''(p)}{2}(x - p)^2
$$

Each additional term improves the approximation in a region around $p$.

---

### 4.3 The Maclaurin Series

The **Maclaurin series** is a Taylor series centered at $x = 0$.

**The Formula:**

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!} x^n = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots
$$

Where $f^{(n)}(0)$ is the $n$-th derivative evaluated at zero.

**Example: $e^x$**

Since $\frac{d}{dx}e^x = e^x$ and $e^0 = 1$:

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

**Magical Property:** Differentiating this series term-by-term returns the same series!

**Example: $\cos(x)$**

Derivatives at zero:

| $n$ | $f^{(n)}(x)$ | $f^{(n)}(0)$ |
|-----|--------------|--------------|
| 0 | $\cos(x)$ | 1 |
| 1 | $-\sin(x)$ | 0 |
| 2 | $-\cos(x)$ | -1 |
| 3 | $\sin(x)$ | 0 |
| 4 | $\cos(x)$ | 1 |

Pattern: Non-zero only for even $n$, alternating signs.

$$
\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots
$$

---

### 4.4 The Taylor Series

The **Taylor series** generalizes to expansion around any point $p$.

**The Formula:**

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(p)}{n!} (x - p)^n
$$

$$
= f(p) + f'(p)(x-p) + \frac{f''(p)}{2!}(x-p)^2 + \frac{f'''(p)}{3!}(x-p)^3 + \cdots
$$

**Derivation of the First-Order Term:**

We want a line $y = mx + c$ that:
1. Passes through $(p, f(p))$
2. Has slope $f'(p)$

From $f(p) = f'(p) \cdot p + c$, we get $c = f(p) - f'(p) \cdot p$.

Substituting:

$$
y = f'(p) \cdot x + f(p) - f'(p) \cdot p = f(p) + f'(p)(x - p)
$$

**Example: Expanding $\frac{1}{x}$ around $x = 1$**

The function $\frac{1}{x}$ has a discontinuity at $x = 0$, so we can't use Maclaurin series.

Derivatives at $x = 1$:

| $n$ | $f^{(n)}(x)$ | $f^{(n)}(1)$ |
|-----|--------------|--------------|
| 0 | $x^{-1}$ | 1 |
| 1 | $-x^{-2}$ | -1 |
| 2 | $2x^{-3}$ | 2 |
| 3 | $-6x^{-4}$ | -6 |

The $n$-th derivative at 1 is $(-1)^n n!$, so:

$$
\frac{1}{x} = \sum_{n=0}^{\infty} (-1)^n (x-1)^n = 1 - (x-1) + (x-1)^2 - (x-1)^3 + \cdots
$$

**Important:** This series only converges for $0 < x < 2$!

---

### 4.5 Linearization and Error Analysis

**Linearization** uses the first-order Taylor approximation.

**Alternative Notation:**

Using $\Delta x$ to denote a small step from $x$:

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x
$$

This says: "The change in $f$ equals the gradient times the step size."

**Error Analysis:**

The error from truncating at the first-order term is **on the order of** $(\Delta x)^2$:

$$
f(x + \Delta x) = f(x) + f'(x)\Delta x + O((\Delta x)^2)
$$

This is called **second-order accuracy** — the error decreases quadratically as $\Delta x$ shrinks.

**Connection to Finite Differences:**

Rearranging the Taylor series:

$$
f'(x) = \frac{f(x + \Delta x) - f(x)}{\Delta x} - \frac{f''(\xi)}{2}\Delta x
$$

The **forward difference** approximation $\frac{f(x + \Delta x) - f(x)}{\Delta x}$ has error proportional to $\Delta x$ — it's **first-order accurate**.

---

### 4.6 Multivariate Taylor Series

**Second-Order Expansion in 2D:**

$$
f(x + \Delta x, y + \Delta y) \approx f(x,y) + \underbrace{\begin{pmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \end{pmatrix} \begin{pmatrix} \Delta x \\\ \Delta y \end{pmatrix}}_{\text{First order: } J \cdot \Delta \mathbf{x}}
$$

$$
+ \frac{1}{2} \underbrace{\begin{pmatrix} \Delta x & \Delta y \end{pmatrix} \begin{pmatrix} \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\\ \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} \end{pmatrix} \begin{pmatrix} \Delta x \\\ \Delta y \end{pmatrix}}_{\text{Second order: } \frac{1}{2}\Delta \mathbf{x}^T H \Delta \mathbf{x}}
$$

**Compact Form:**

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + J \cdot \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T H \Delta\mathbf{x}
$$

Where:
- $J$ = Jacobian (first derivatives)
- $H$ = Hessian (second derivatives)

**Geometric Interpretation:**

- **Zeroth order:** Flat surface at $f(x, y)$
- **First order:** Tangent plane matching gradient
- **Second order:** Parabolic surface matching curvature

---

## Module 5: Gradient-Based Optimization

This module presents practical algorithms for finding function minima.

### 5.1 The Newton-Raphson Method

**Goal:** Find roots of an equation (where $f(x) = 0$).

**The Algorithm:**

Starting from initial guess $x_0$:

$$
x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}
$$

**Intuition:**

1. Evaluate function and gradient at current guess
2. Approximate function as a straight line (tangent)
3. Find where this line crosses zero
4. Use that as the new guess
5. Repeat until convergence

**Example: Solving $x^3 - 2x + 2 = 0$**

$f(x) = x^3 - 2x + 2$, $f'(x) = 3x^2 - 2$

Starting at $x_0 = -2$:

| Iteration | $x_i$ | $f(x_i)$ | $f'(x_i)$ | $x_{i+1}$ |
|-----------|-------|----------|-----------|-----------|
| 0 | -2.000 | -2.000 | 10.000 | -1.800 |
| 1 | -1.800 | -0.232 | 7.720 | -1.770 |
| 2 | -1.770 | -0.050 | 7.397 | -1.769 |
| 3 | -1.769 | -0.000002 | 7.388 | **-1.7693** |

Convergence in just 3 iterations!

**Potential Problems:**

1. **Cycling:** Some starting points lead to infinite loops
2. **Divergence:** Starting near turning points can send estimates to infinity
3. **Multiple Roots:** Only finds one root, depending on starting point

---

### 5.2 The Gradient Vector (Grad)

The **gradient** $\nabla f$ combines calculus and vectors into a powerful tool.

**Definition:**

$$
\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x} \\\ \frac{\partial f}{\partial y} \\\ \vdots \end{pmatrix}
$$

(Note: Sometimes written as row vector for Jacobian; here as column for gradient descent.)

**Properties:**

1. **Direction:** Points toward steepest increase
2. **Magnitude:** Equals the rate of steepest increase
3. **Perpendicular to Level Sets:** $\nabla f$ is perpendicular to contour lines

**The Directional Derivative:**

The rate of change in direction $\hat{\mathbf{r}}$ (unit vector):

$$
D_{\hat{\mathbf{r}}} f = \nabla f \cdot \hat{\mathbf{r}} = |\nabla f| \cos\theta
$$

This is maximized when $\hat{\mathbf{r}}$ points in the same direction as $\nabla f$.

**Maximum Value:**

$$
\max_{\hat{\mathbf{r}}} (D_{\hat{\mathbf{r}}} f) = |\nabla f|
$$

---

### 5.3 Gradient Descent

**Goal:** Find the minimum of a function by following the gradient downhill.

**The Algorithm:**

$$
\mathbf{x}_{n+1} = \mathbf{x}_n - \gamma \nabla f(\mathbf{x}_n)
$$

Where $\gamma > 0$ is the **learning rate** (step size).

**Intuition:**

1. Stand on the function surface
2. Feel which direction is steepest uphill (gradient)
3. Take a small step in the opposite direction (downhill)
4. Repeat until you reach a valley

**The Foggy Mountain Analogy:**

Imagine hiking down a mountain in thick fog:
- You can't see the whole landscape
- You can feel the local slope with your feet
- You always step in the steepest downhill direction
- Eventually you reach a valley (possibly local, not global)

**Choosing the Learning Rate:**

- **Too large:** Overshoot the minimum, potentially diverge
- **Too small:** Convergence is very slow
- **Just right:** Efficient convergence

**Advantages:**

- Simple to implement
- Works in any number of dimensions
- Only requires gradient computation

**Disadvantages:**

- May get stuck in local minima
- Convergence can be slow in flat regions
- Sensitive to learning rate choice

---

### 5.4 Lagrange Multipliers

**Goal:** Find extrema of a function **subject to constraints**.

**The Setup:**

Maximize/minimize $f(x, y)$ subject to $g(x, y) = c$.

**The Key Insight:**

At the optimum, the gradient of $f$ is parallel to the gradient of $g$:

$$
\nabla f = \lambda \nabla g
$$

Where $\lambda$ is the **Lagrange multiplier**.

**Why This Works:**

At the optimum point:
- Moving along the constraint doesn't change $f$ (it's extremal)
- $\nabla f$ is perpendicular to the constraint curve
- $\nabla g$ is also perpendicular to the constraint curve
- Therefore, $\nabla f$ and $\nabla g$ must be parallel

**The Method:**

Solve the system:

$$
\nabla f = \lambda \nabla g
$$
$$
g(x, y) = c
$$

**Example: Maximize $f(x, y) = x^2 y$ on circle $x^2 + y^2 = a^2$**

**Step 1:** Compute gradients:

$$
\nabla f = \begin{pmatrix} 2xy \\\ x^2 \end{pmatrix}, \quad \nabla g = \begin{pmatrix} 2x \\\ 2y \end{pmatrix}
$$

**Step 2:** Set up equations:

$$
2xy = \lambda \cdot 2x \quad \Rightarrow \quad y = \lambda \text{ (if } x \neq 0\text{)}
$$

$$
x^2 = \lambda \cdot 2y = 2y^2 \quad \Rightarrow \quad x = \pm\sqrt{2}y
$$

**Step 3:** Apply constraint:

$$
x^2 + y^2 = 2y^2 + y^2 = 3y^2 = a^2 \quad \Rightarrow \quad y = \pm\frac{a}{\sqrt{3}}
$$

**Step 4:** Find all solutions:

$$
(x, y) = \left(\pm\frac{a\sqrt{2}}{\sqrt{3}}, \pm\frac{a}{\sqrt{3}}\right)
$$

The maximum value is $f = \frac{2a^3}{3\sqrt{3}}$ (at points where $y > 0$).

---

## Module 6: Regression and Least Squares

This module applies everything we've learned to fit functions to data.

### 6.1 Introduction to Data Fitting

**The Goal:** Find parameters that make a model function best match observed data.

**The Process:**

1. **Clean the data:** Handle missing values, outliers, duplicates
2. **Visualize:** Plot to understand patterns
3. **Choose a model:** Select an appropriate function form
4. **Fit parameters:** Find values that minimize error
5. **Validate:** Check if the fit makes sense

**Example: Fitting a Straight Line**

Given data points $(x_i, y_i)$, fit:

$$
y = mx + c
$$

---

### 6.2 Linear Regression

**The Chi-Squared Metric:**

Define the **residual** for each point:

$$
r_i = y_i - (mx_i + c)
$$

The **chi-squared** measures total fit quality:

$$
\chi^2 = \sum_{i=1}^{n} r_i^2 = \sum_{i=1}^{n} (y_i - mx_i - c)^2
$$

**Finding the Best Fit:**

Set partial derivatives to zero:

$$
\frac{\partial \chi^2}{\partial m} = -2\sum_{i} x_i(y_i - mx_i - c) = 0
$$

$$
\frac{\partial \chi^2}{\partial c} = -2\sum_{i} (y_i - mx_i - c) = 0
$$

**Solution:**

From the second equation:

$$
c = \bar{y} - m\bar{x}
$$

Where $\bar{x}$ and $\bar{y}$ are the means.

The **best-fit line passes through the center of mass** $(\bar{x}, \bar{y})$.

For the slope:

$$
m = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}
$$

**Uncertainties in Parameters:**

$$
\sigma_m^2 = \frac{\sigma^2}{\sum_i (x_i - \bar{x})^2}
$$

$$
\sigma_c^2 = \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_i (x_i - \bar{x})^2}\right)
$$

Where $\sigma^2$ is the variance of residuals.

**Anscombe's Quartet Warning:**

Four very different datasets can have identical:
- Means
- Variances
- Best-fit lines
- Correlation coefficients

**Always visualize your data!** Statistical measures can hide important patterns.

---

### 6.3 Nonlinear Least Squares

**The General Problem:**

Fit a nonlinear model $y = f(x; \mathbf{a})$ where $\mathbf{a}$ are parameters.

**The Chi-Squared Function:**

$$
\chi^2(\mathbf{a}) = \sum_{i=1}^{n} \frac{(y_i - f(x_i; \mathbf{a}))^2}{\sigma_i^2}
$$

Where $\sigma_i$ is the uncertainty in data point $i$.

**Gradient Descent for Fitting:**

Update parameters iteratively:

$$
\mathbf{a}_{\text{new}} = \mathbf{a}_{\text{old}} - \gamma \nabla \chi^2
$$

**Computing the Gradient:**

$$
\frac{\partial \chi^2}{\partial a_k} = -2\sum_{i} \frac{y_i - f(x_i; \mathbf{a})}{\sigma_i^2} \cdot \frac{\partial f}{\partial a_k}
$$

**Example: Fitting a Quadratic**

For $f(x; a_1, a_2) = (x - a_1)^2 + a_2$:

$$
\frac{\partial f}{\partial a_1} = -2(x - a_1), \quad \frac{\partial f}{\partial a_2} = 1
$$

---

### 6.4 Practical Implementation

**Advanced Methods:**

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Gradient Descent** | Follow gradient downhill | Simple problems |
| **Newton's Method** | Use Hessian for step size | Near minimum |
| **Gauss-Newton** | Approximates Hessian | Least squares problems |
| **Levenberg-Marquardt** | Hybrid of above | Most nonlinear fits |
| **BFGS** | Builds Hessian iteratively | General optimization |

**In Python:**

```python
from scipy.optimize import curve_fit
import numpy as np

# Define the model
def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2))

# Fit to data
params, covariance = curve_fit(gaussian, x_data, y_data, p0=[1, 0, 1])

# Extract parameters and uncertainties
amplitude, mean, sigma = params
uncertainties = np.sqrt(np.diag(covariance))
```

**In MATLAB:**

```matlab
% Using Curve Fitting Toolbox
ft = fittype('a*exp(-((x-b)/c)^2)');
f = fit(x_data, y_data, ft, 'StartPoint', [1, 0, 1]);
```

**Best Practices:**

1. **Good Starting Guess:** Essential for convergence
2. **Visualize the Fit:** Always plot data and model together
3. **Check Residuals:** Should be randomly distributed
4. **Report Uncertainties:** Parameters are meaningless without error estimates
5. **Consider Alternatives:** The fitted model might not be the right one

---

## Complete Calculus Toolbox

### Differentiation Rules

| Rule | Formula |
|------|---------|
| Sum | $\frac{d}{dx}[f + g] = f' + g'$ |
| Product | $\frac{d}{dx}[fg] = fg' + gf'$ |
| Quotient | $\frac{d}{dx}\left[\frac{f}{g}\right] = \frac{gf' - fg'}{g^2}$ |
| Chain | $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ |
| Power | $\frac{d}{dx}[x^n] = nx^{n-1}$ |

### Common Derivatives

| Function | Derivative |
|----------|------------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln(x)$ | $\frac{1}{x}$ |
| $\sin(x)$ | $\cos(x)$ |
| $\cos(x)$ | $-\sin(x)$ |
| $\tan(x)$ | $\sec^2(x)$ |

### Multivariate Tools

| Concept | Definition | Dimension |
|---------|------------|-----------|
| Partial Derivative | $\frac{\partial f}{\partial x_i}$ | Scalar |
| Gradient / Jacobian Vector | $\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$ | $1 \times n$ |
| Jacobian Matrix | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ | $m \times n$ |
| Hessian | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | $n \times n$ |

### Optimization Algorithms

| Method | Update Rule | Use Case |
|--------|-------------|----------|
| Newton-Raphson | $x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$ | Finding roots |
| Gradient Descent | $\mathbf{x}_{n+1} = \mathbf{x}_n - \gamma\nabla f$ | Finding minima |
| Lagrange Multipliers | $\nabla f = \lambda \nabla g$ | Constrained optimization |

### Taylor Series

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(p)}{n!}(x-p)^n
$$

**Common Series (around $x = 0$):**

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

$$
\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}
$$

$$
\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}
$$

$$
\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n \quad (|x| < 1)
$$

---

## Key Takeaways for Machine Learning

1. **Derivatives measure change** — essential for understanding how model parameters affect predictions

2. **The chain rule enables backpropagation** — the fundamental algorithm for training neural networks

3. **The Jacobian points uphill** — gradient descent goes the opposite direction to minimize loss

4. **The Hessian describes curvature** — helps understand optimization landscape and accelerate convergence

5. **Taylor series connect local to global** — linearization enables efficient numerical methods

6. **Least squares fitting is optimization** — training a model means minimizing a cost function

7. **Always visualize** — mathematical elegance can hide data pathologies

---

> **Course Completion**
> 
> You now have the mathematical foundation to understand how machine learning algorithms work under the hood. The concepts of gradients, optimization, and function approximation appear throughout deep learning, and this calculus toolkit will serve you well as you dive deeper into the field.
