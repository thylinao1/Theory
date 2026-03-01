# Supervised Machine Learning: Regression and Classification

> **Comprehensive Course Notes — Andrew Ng's Machine Learning Specialization (Course 1)**
> Detailed notes covering linear regression, gradient descent, multiple features, logistic regression, overfitting, and regularization with full mathematical derivations, intuitive explanations, and Python code snippets.

---

## Table of Contents

1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
   - [1.1 What is Machine Learning?](#11-what-is-machine-learning)
   - [1.2 Supervised Learning](#12-supervised-learning)
   - [1.3 Unsupervised Learning](#13-unsupervised-learning)
2. [Linear Regression with One Variable](#2-linear-regression-with-one-variable)
   - [2.1 The Linear Regression Model](#21-the-linear-regression-model)
   - [2.2 Notation and Training Sets](#22-notation-and-training-sets)
   - [2.3 The Cost Function](#23-the-cost-function)
   - [2.4 Intuition Behind the Cost Function](#24-intuition-behind-the-cost-function)
   - [2.5 Visualizing the Cost Function in 3D](#25-visualizing-the-cost-function-in-3d)
3. [Gradient Descent](#3-gradient-descent)
   - [3.1 The Gradient Descent Algorithm](#31-the-gradient-descent-algorithm)
   - [3.2 Understanding the Learning Rate](#32-understanding-the-learning-rate)
   - [3.3 Understanding the Derivative Term](#33-understanding-the-derivative-term)
   - [3.4 Gradient Descent for Linear Regression](#34-gradient-descent-for-linear-regression)
4. [Multiple Linear Regression](#4-multiple-linear-regression)
   - [4.1 Multiple Features](#41-multiple-features)
   - [4.2 Vectorization](#42-vectorization)
   - [4.3 Gradient Descent for Multiple Regression](#43-gradient-descent-for-multiple-regression)
   - [4.4 The Normal Equation (Alternative)](#44-the-normal-equation-alternative)
5. [Practical Tips for Linear Regression](#5-practical-tips-for-linear-regression)
   - [5.1 Feature Scaling](#51-feature-scaling)
   - [5.2 Checking Gradient Descent Convergence](#52-checking-gradient-descent-convergence)
   - [5.3 Choosing the Learning Rate](#53-choosing-the-learning-rate)
   - [5.4 Feature Engineering](#54-feature-engineering)
   - [5.5 Polynomial Regression](#55-polynomial-regression)
6. [Logistic Regression](#6-logistic-regression)
   - [6.1 Binary Classification](#61-binary-classification)
   - [6.2 Why Not Linear Regression for Classification?](#62-why-not-linear-regression-for-classification)
   - [6.3 The Sigmoid Function](#63-the-sigmoid-function)
   - [6.4 The Logistic Regression Model](#64-the-logistic-regression-model)
   - [6.5 Decision Boundaries](#65-decision-boundaries)
   - [6.6 Cost Function for Logistic Regression](#66-cost-function-for-logistic-regression)
   - [6.7 Simplified Loss Function](#67-simplified-loss-function)
   - [6.8 Gradient Descent for Logistic Regression](#68-gradient-descent-for-logistic-regression)
7. [Overfitting and Regularization](#7-overfitting-and-regularization)
   - [7.1 The Problem of Overfitting](#71-the-problem-of-overfitting)
   - [7.2 Addressing Overfitting](#72-addressing-overfitting)
   - [7.3 Regularization Intuition](#73-regularization-intuition)
   - [7.4 Regularized Linear Regression](#74-regularized-linear-regression)
   - [7.5 Regularized Logistic Regression](#75-regularized-logistic-regression)

---

## 1. Introduction to Machine Learning

### 1.1 What is Machine Learning?

Arthur Samuel (1959) defined machine learning as:

> *"The field of study that gives computers the ability to learn without being explicitly programmed."*

Samuel demonstrated this concept by writing a checkers-playing program. Despite not being a strong checkers player himself, he had the computer play tens of thousands of games against itself. Over time, the program learned which board positions tended to lead to wins and which led to losses, eventually becoming a better player than Samuel. The key insight here is that **more experience (data) leads to better performance** — if the computer had played fewer games, it would have performed worse.

Machine learning grew up as a subfield of Artificial Intelligence (AI). We wanted to build intelligent machines, but for most interesting tasks — such as web search, speech recognition, medical diagnosis, and self-driving — we simply could not write explicit programs. The only viable approach was to have machines **learn from data**.

According to McKinsey, AI and machine learning are estimated to create an additional **$13 trillion** of value annually by 2030, with massive untapped opportunities in retail, travel, transportation, automotive, manufacturing, and beyond.

**The two main types of machine learning are:**

- **Supervised Learning** — the most widely used, with the most rapid advancements
- **Unsupervised Learning** — finding patterns in data without labeled outcomes

Other important paradigms include **recommender systems** and **reinforcement learning**.

---

### 1.2 Supervised Learning

Supervised learning refers to algorithms that learn **input-to-output mappings** (X → Y). The key characteristic is that you provide the algorithm with examples that include the **correct answers** (labels). By seeing correct pairs of input X and desired output label Y, the algorithm eventually learns to take just the input alone and produce a reasonably accurate prediction.

**Examples of Supervised Learning Applications:**

| Input (X) | Output (Y) | Application |
|---|---|---|
| Email | Spam or not spam | Spam filtering |
| Audio clip | Text transcript | Speech recognition |
| English text | Spanish text | Machine translation |
| Ad + user info | Click or no click | Online advertising |
| Image + sensor data | Position of other cars | Self-driving cars |
| Image of product | Defect or no defect | Visual inspection |

**The two major types of supervised learning are:**

**Regression** — predicting a **continuous number** from infinitely many possible values.

For example, predicting house prices based on size. The output could be \$150,000, \$283,500, or any number. Given a dataset of house sizes and prices, you might fit a straight line or a curve to predict the price of a new house.

**Classification** — predicting a **discrete category** from a small, finite set of possible outputs.

For example, classifying breast tumors as malignant (1) or benign (0). Unlike regression, which predicts any number, classification predicts one of a limited set of categories. Classification can also handle more than two categories (e.g., Type 1 cancer, Type 2 cancer, benign).

A critical distinction: in classification, even when the output categories are numbers (0, 1, 2), you are predicting from a **finite, limited set** — not all possible numbers in between. The number 0.5 or 1.7 would not be valid classification outputs if the categories are {0, 1, 2}.

Classification can also use **multiple input features**. For instance, using both tumor size and patient age to predict malignancy. The learning algorithm then finds a **decision boundary** that separates the classes in the feature space.

---

### 1.3 Unsupervised Learning

In unsupervised learning, the data comes **without output labels**. Instead of being told the "right answer," the algorithm must find **structure, patterns, or something interesting** in the data on its own.

**Clustering** is the most common type of unsupervised learning. It automatically groups similar data points together without being told what the groups should be.

**Real-world clustering examples:**

- **Google News** — Automatically groups hundreds of thousands of news articles by topic each day. The algorithm identifies that articles mentioning "panda," "twin," and "zoo" belong together, without anyone explicitly defining these groupings.

- **DNA Microarray Analysis** — Clustering genetic data to discover different types of individuals. Each column represents a person's DNA activity, and each row represents a gene. The algorithm can automatically identify distinct genetic groupings (Type 1, Type 2, Type 3, etc.) without being told in advance what types of people exist.

- **Market Segmentation** — Companies like DeepLearning.AI cluster their community into groups: those primarily seeking knowledge, those focused on career development, those wanting to stay updated on AI trends, and others.

**Other types of unsupervised learning:**

- **Anomaly Detection** — Identifying unusual events, critically important for fraud detection in the financial system.

- **Dimensionality Reduction** — Compressing a large dataset to a much smaller one while losing as little information as possible.

---

## 2. Linear Regression with One Variable

### 2.1 The Linear Regression Model

Linear regression is probably the **most widely used learning algorithm** in the world today. It involves fitting a straight line to your data.

**Example:** Predicting house prices based on size using a Portland, Oregon housing dataset. The horizontal axis is size in square feet, and the vertical axis is price in thousands of dollars. Each data point (cross on the plot) represents one house.

If a client wants to know the price of their 1,250 square foot house, a linear regression model fits a straight line to the data. Where this line intersects 1,250 square feet gives the predicted price — perhaps around \$220,000.

This is supervised learning because we gave the algorithm a dataset with the **right answers** — the correct price Y for every house with input X. The algorithm's task is to produce more of these right answers for new inputs.

**Key distinction:** Linear regression is a **regression** model because it predicts numbers (prices), not categories.

---

### 2.2 Notation and Training Sets

The dataset used to train the model is called the **training set**. Here is the standard notation used throughout machine learning:

| Symbol | Meaning | Example |
|---|---|---|
| $x$ | Input variable / feature | Size of house |
| $y$ | Output variable / target | Price of house |
| $m$ | Number of training examples | 47 |
| $(x, y)$ | A single training example | (2104, 400) |
| $(x^{(i)}, y^{(i)})$ | The i-th training example | $(x^{(1)}, y^{(1)}) = (2104, 400)$ |

**Important:** The superscript $(i)$ in parentheses is **not exponentiation**. It is simply an index referring to the i-th row of the training data. So $x^{(2)}$ does not mean "x squared" — it means "the input features of the second training example."

**The supervised learning process:**

1. Feed the training set (input features + output targets) to the learning algorithm
2. The algorithm produces a function $f$ (historically called a "hypothesis")
3. The function $f$ takes a new input $x$ and outputs an estimate $\hat{y}$ (y-hat)

Here, $\hat{y}$ is the **prediction** — the estimated value of $y$. The actual true value is just $y$. The model's prediction $\hat{y}$ may or may not equal the true value $y$.

---

### 2.3 The Cost Function

The cost function tells us **how well the model is doing** so that we can try to improve it. It is one of the most universal and important ideas in machine learning.

**Model definition:**

$$
f_{w,b}(x) = wx + b
$$

The parameters $w$ and $b$ are the variables you adjust during training to improve the model. They are sometimes called **coefficients** or **weights**.

- $w$ controls the **slope** of the line
- $b$ controls the **y-intercept** of the line

**Goal:** Find values of $w$ and $b$ so that $\hat{y}^{(i)}$ is close to $y^{(i)}$ for all training examples.

**Building the cost function step by step:**

**Step 1 — Compute the error for one example:**

$$
\hat{y}^{(i)} - y^{(i)} = f_{w,b}(x^{(i)}) - y^{(i)}
$$

This is the difference between the prediction and the actual value.

**Step 2 — Square the error** (to make all errors positive and penalize large errors more):

$$
\left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

**Step 3 — Sum over all training examples:**

$$
\sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

**Step 4 — Take the average** (so the cost doesn't automatically grow with more data):

$$
\frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

**Step 5 — Divide by 2** (a convention that makes later calculus neater):

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

This is the **Squared Error Cost Function** (also called **Mean Squared Error** or MSE). It is by far the most commonly used cost function for regression problems.

**The optimization goal:**

$$
\min_{w, b} J(w, b)
$$

We want to find the values of $w$ and $b$ that minimize the cost function $J$.

---

### 2.4 Intuition Behind the Cost Function

To build intuition, let's temporarily simplify by setting $b = 0$. This gives us a model that passes through the origin:

$$
f_w(x) = wx
$$

And a cost function of just one parameter:

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} \left( wx^{(i)} - y^{(i)} \right)^2
$$

**Worked Example** with training set: {(1,1), (2,2), (3,3)}

**Case 1: w = 1**

The line $f(x) = x$ passes through all three points perfectly.

$$
J(1) = \frac{1}{2 \cdot 3} \left[ (1-1)^2 + (2-2)^2 + (3-3)^2 \right] = \frac{1}{6}(0 + 0 + 0) = 0
$$

**Case 2: w = 0.5**

The line $f(x) = 0.5x$ underestimates all points.

$$
J(0.5) = \frac{1}{6} \left[ (0.5-1)^2 + (1-2)^2 + (1.5-3)^2 \right] = \frac{1}{6}(0.25 + 1 + 2.25) = \frac{3.5}{6} \approx 0.58
$$

**Case 3: w = 0**

The flat line $f(x) = 0$ misses everything.

$$
J(0) = \frac{1}{6} \left[ (0-1)^2 + (0-2)^2 + (0-3)^2 \right] = \frac{1}{6}(1 + 4 + 9) = \frac{14}{6} \approx 2.33
$$

When you plot $J(w)$ against $w$, you get a **U-shaped curve** (a parabola). The minimum of this curve is at $w = 1$ where $J = 0$. This is the value of $w$ that best fits the data.

**Key insight:** Each value of parameter $w$ corresponds to a different straight line fit AND a single point on the cost function graph. The best line is the one corresponding to the lowest point on the cost curve.

---

### 2.5 Visualizing the Cost Function in 3D

When we use both parameters $w$ and $b$, the cost function $J(w, b)$ becomes a **3D surface** — shaped like a bowl (or a hammock).

- The axes at the bottom are $w$ and $b$
- The vertical axis is the cost $J(w, b)$
- Any single point on the surface represents a particular choice of $w$ and $b$

**Contour plots** provide a 2D alternative for visualizing this 3D surface. Think of a topographical map showing mountain elevations. Each oval (ellipse) connects all points at the **same height** — meaning the same cost $J$.

- The center of the concentric ovals is the **minimum** of the cost function
- Points further from the center have higher cost
- If you imagine flying directly above the bowl, the contour plot is what you'd see looking straight down

**Key observations from contour plots:**

- A pair of $(w, b)$ values far from the center corresponds to a poorly-fitting line and high cost
- A pair near the center gives a good fit and low cost
- The global minimum at the center gives the **best-fit line**

---

## 3. Gradient Descent

### 3.1 The Gradient Descent Algorithm

Gradient descent is one of the **most important algorithms in machine learning**. It is used not just for linear regression, but also to train neural networks and deep learning models.

**Overview:** Start with some initial guesses for $w$ and $b$ (commonly both set to 0). Then, keep changing the parameters a little bit each iteration to reduce the cost $J(w, b)$ until it settles at or near a minimum.

**The hill analogy:** Imagine standing on a hilly surface. At each step, you spin 360 degrees, find the direction of steepest descent, and take a baby step in that direction. Repeat until you reach the bottom of a valley.

**The algorithm:**

$$
w := w - \alpha \frac{\partial}{\partial w} J(w, b)
$$

$$
b := b - \alpha \frac{\partial}{\partial b} J(w, b)
$$

where:

- $\alpha$ is the **learning rate** — a small positive number (e.g., 0.01) that controls how big each step is
- $\frac{\partial}{\partial w} J(w, b)$ is the **partial derivative** of the cost function with respect to $w$ — it tells you the direction and magnitude of the steepest ascent

**Critical requirement: Simultaneous update.** You must compute both derivatives FIRST, then update both parameters at the same time.

**Correct (simultaneous) implementation:**

```python
# Compute both derivatives first
temp_w = w - alpha * dj_dw
temp_b = b - alpha * dj_db

# Then update simultaneously
w = temp_w
b = temp_b
```

**Incorrect (sequential) implementation:**

```python
# DON'T do this — w changes before b's derivative is computed
w = w - alpha * dj_dw        # w is already changed!
b = b - alpha * dj_db        # uses the NEW w, which is wrong
```

In the incorrect version, the updated $w$ leaks into the computation of the derivative for $b$, leading to a different (incorrect) algorithm.

---

### 3.2 Understanding the Learning Rate

The learning rate $\alpha$ controls **how big** each gradient descent step is.

**If $\alpha$ is too small:**

- Gradient descent will still work, but **extremely slowly**
- You take tiny baby steps and need many iterations to converge
- Computationally expensive and wastes time

**If $\alpha$ is too large:**

- Steps overshoot the minimum
- The cost may **increase** instead of decrease
- Gradient descent can fail to converge and may even **diverge** (cost goes to infinity)

**A crucial property — gradient descent automatically takes smaller steps near the minimum:**

Even with a **fixed** learning rate $\alpha$, the steps naturally get smaller as you approach a local minimum. This happens because the derivative (slope) gets closer to zero near the minimum. Since the update is $\alpha \times \text{derivative}$, smaller derivatives mean smaller steps.

**What happens at a local minimum?**

If $w$ is already at a local minimum, the derivative is zero. The update becomes:

$$
w := w - \alpha \cdot 0 = w
$$

Gradient descent leaves $w$ unchanged — which is exactly what we want.

---

### 3.3 Understanding the Derivative Term

The derivative tells you **which direction** to step and **how steep** the slope is.

**When the derivative is positive** (tangent line slopes upward to the right):

- $w := w - \alpha \cdot (\text{positive number})$
- $w$ decreases (moves left on the graph)
- This moves toward the minimum (which is to the left)

**When the derivative is negative** (tangent line slopes downward to the right):

- $w := w - \alpha \cdot (\text{negative number})$
- Subtracting a negative is adding a positive
- $w$ increases (moves right on the graph)
- This also moves toward the minimum (which is to the right)

In both cases, gradient descent moves $w$ in the correct direction — toward the minimum.

---

### 3.4 Gradient Descent for Linear Regression

Combining gradient descent with the squared error cost function gives us the **linear regression learning algorithm**.

**The derivatives for linear regression are:**

$$
\frac{\partial}{\partial w} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

$$
\frac{\partial}{\partial b} J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

Notice the key difference: the derivative with respect to $w$ has an extra $x^{(i)}$ factor at the end.

**Optional derivation (using calculus):**

Starting from the cost function and substituting $f_{w,b}(x^{(i)}) = wx^{(i)} + b$:

$$
\frac{\partial}{\partial w} J = \frac{\partial}{\partial w} \left[ \frac{1}{2m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)})^2 \right]
$$

Applying the chain rule: the derivative of $(u)^2$ is $2u$, and the derivative of $wx^{(i)} + b$ with respect to $w$ is $x^{(i)}$:

$$
= \frac{1}{2m} \sum_{i=1}^{m} 2(wx^{(i)} + b - y^{(i)}) \cdot x^{(i)}
$$

The 2's cancel (this is why we included $\frac{1}{2}$ in the cost function):

$$
= \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}
$$

**An important property:** The squared error cost function for linear regression is always a **convex function** — it has a single global minimum and no local minima. This means gradient descent is guaranteed to converge to the global minimum (with an appropriate learning rate).

This is unlike more general functions (like those in neural networks) that can have multiple local minima.

**The full gradient descent algorithm for linear regression:**

```python
import numpy as np

def gradient_descent(X, y, w, b, alpha, num_iterations):
    """
    Performs batch gradient descent for univariate linear regression.
    
    Parameters:
        X: input features (m,)
        y: target values (m,)
        w: initial weight
        b: initial bias
        alpha: learning rate
        num_iterations: number of iterations to run
    
    Returns:
        w, b: optimized parameters
        J_history: cost at each iteration
    """
    m = len(X)
    J_history = []
    
    for i in range(num_iterations):
        # Compute predictions
        f_wb = w * X + b
        
        # Compute derivatives (simultaneous update)
        dj_dw = (1/m) * np.sum((f_wb - y) * X)
        dj_db = (1/m) * np.sum(f_wb - y)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Track cost
        cost = (1/(2*m)) * np.sum((f_wb - y)**2)
        J_history.append(cost)
    
    return w, b, J_history
```

**Batch Gradient Descent:** On every step, we look at **all** training examples (the entire "batch") when computing the derivatives. This is the standard form for linear regression. Other variants exist that look at smaller subsets of data per step.

---

## 4. Multiple Linear Regression

### 4.1 Multiple Features

In the original linear regression, we had a single feature $x$ (e.g., house size). In reality, we often have **many features** that can help predict the output.

**Example — predicting house prices with multiple features:**

| Feature | Symbol | Description |
|---|---|---|
| Size (sq ft) | $x_1$ | 2104, 1416, ... |
| # Bedrooms | $x_2$ | 5, 3, ... |
| # Floors | $x_3$ | 1, 2, ... |
| Age (years) | $x_4$ | 45, 40, ... |

**Extended notation:**

| Symbol | Meaning |
|---|---|
| $n$ | Total number of features |
| $x_j$ | The j-th feature |
| $\vec{x}^{(i)}$ | Feature vector of the i-th training example |
| $x_j^{(i)}$ | The j-th feature of the i-th example |

For example, $\vec{x}^{(2)} = (1416, 3, 2, 40)$ is the feature vector for the second training example, and $x_3^{(2)} = 2$ is the number of floors in that example.

**The multiple linear regression model:**

$$
f_{w,b}(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots + w_n x_n + b
$$

**Concrete example:**

$$
\hat{y} = 0.1 x_1 + 4 x_2 + 10 x_3 - 2 x_4 + 80
$$

Interpretation:

- Base price: \$80,000 ($b = 80$)
- Each additional sq ft adds \$100 ($0.1 \times \$1000$)
- Each additional bedroom adds \$4,000
- Each additional floor adds \$10,000
- Each year of age **decreases** price by \$2,000 (negative coefficient)

**Using vector notation:**

Define $\vec{w} = (w_1, w_2, \ldots, w_n)$ and $\vec{x} = (x_1, x_2, \ldots, x_n)$.

The model becomes:

$$
f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b
$$

where $\vec{w} \cdot \vec{x}$ is the **dot product**:

$$
\vec{w} \cdot \vec{x} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \sum_{j=1}^{n} w_j x_j
$$

**Note:** This is called **multiple linear regression** (not "multivariate regression," which refers to something different). The prefix "multiple" refers to having multiple input features.

---

### 4.2 Vectorization

Vectorization is a technique that makes code **shorter** and **much faster** by leveraging parallel hardware (CPU or GPU).

**Three ways to compute $f = \vec{w} \cdot \vec{x} + b$:**

**Method 1 — Manual (not scalable):**

```python
f = w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + b
# This is fine for 3 features but terrible for 100+
```

**Method 2 — For loop (no vectorization):**

```python
f = 0
for j in range(n):
    f = f + w[j] * x[j]
f = f + b
# Computes each multiplication one at a time, sequentially
```

**Method 3 — Vectorized (using NumPy):**

```python
import numpy as np
f = np.dot(w, x) + b
# Single line, runs MUCH faster using parallel hardware
```

**Why is vectorization faster?**

Without vectorization, the computer performs operations **one at a time** sequentially. With vectorization, the computer uses **parallel processing hardware** to:

1. Multiply all $w_j \times x_j$ pairs **simultaneously** in one step
2. Sum all results using specialized hardware, much more efficiently than sequential addition

The speed difference becomes dramatic with large datasets. For $n = 100{,}000$ features, vectorized code might run in seconds while a for-loop takes minutes.

**Vectorized gradient descent update:**

Without vectorization:

```python
for j in range(n):
    w[j] = w[j] - alpha * d[j]  # Sequential updates
```

With vectorization:

```python
w = w - alpha * d  # All 16+ parameters updated in parallel
```

Behind the scenes, NumPy takes the entire vector $\vec{w}$, subtracts $\alpha \cdot \vec{d}$ element-wise using parallel hardware, and assigns the result back — all in one step.

---

### 4.3 Gradient Descent for Multiple Regression

The gradient descent algorithm extends naturally to multiple features.

**Using vector notation:**

$$
\vec{w} = (w_1, w_2, \ldots, w_n), \quad b \in \mathbb{R}
$$

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \vec{w} \cdot \vec{x}^{(i)} + b - y^{(i)} \right)^2
$$

**The gradient descent updates for multiple features:**

For each parameter $w_j$ (where $j = 1, 2, \ldots, n$):

$$
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

Notice the key difference from univariate regression: the derivative with respect to $w_j$ uses $x_j^{(i)}$ — the j-th feature of the i-th example. Each parameter $w_j$ is updated based on how much its corresponding feature $x_j$ contributes to the error.

**Complete vectorized implementation:**

```python
import numpy as np

def gradient_descent_multi(X, y, w, b, alpha, num_iterations):
    """
    Gradient descent for multiple linear regression.
    
    Parameters:
        X: input features, shape (m, n)
        y: target values, shape (m,)
        w: initial weights, shape (n,)
        b: initial bias (scalar)
        alpha: learning rate
        num_iterations: number of iterations
    
    Returns:
        w, b: optimized parameters
    """
    m, n = X.shape
    
    for i in range(num_iterations):
        # Vectorized prediction for all examples at once
        f_wb = np.dot(X, w) + b           # shape (m,)
        
        # Error term
        error = f_wb - y                    # shape (m,)
        
        # Vectorized gradient computation
        dj_dw = (1/m) * np.dot(X.T, error)  # shape (n,)
        dj_db = (1/m) * np.sum(error)        # scalar
        
        # Simultaneous update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b
```

---

### 4.4 The Normal Equation (Alternative)

There exists an alternative to gradient descent for linear regression called the **Normal Equation**. It solves for $w$ and $b$ directly in one step using linear algebra, without iteration.

**Disadvantages of the normal equation:**

1. **Not generalizable** — it only works for linear regression, not logistic regression, neural networks, or other models
2. **Slow for large $n$** — computational complexity grows with the number of features
3. In practice, no one implements it from scratch — but some ML libraries (like scikit-learn) may use it internally

For most practical applications, **gradient descent is the better approach** — especially since it generalizes to all other learning algorithms.

---

## 5. Practical Tips for Linear Regression

### 5.1 Feature Scaling

Feature scaling is a technique that can make gradient descent **run much faster**.

**The problem:** When features have very different ranges of values, gradient descent can oscillate and converge slowly.

**Example:**

- $x_1$ = size in square feet: ranges from 300 to 2,000
- $x_2$ = number of bedrooms: ranges from 0 to 5

Because $x_1$ is much larger, a small change in its parameter $w_1$ has a huge impact on predictions, while a large change in $w_2$ has relatively little impact. This creates elongated, oval-shaped contours in the cost function, causing gradient descent to bounce back and forth.

**When features are on similar scales,** the contours become more circular, and gradient descent can take a much more direct path to the minimum.

**Three common scaling methods:**

**Method 1 — Simple scaling (divide by maximum):**

$$
x_1^{\text{scaled}} = \frac{x_1}{2000}, \quad x_2^{\text{scaled}} = \frac{x_2}{5}
$$

Features now range from roughly 0 to 1.

**Method 2 — Mean normalization:**

$$
x_j^{\text{normalized}} = \frac{x_j - \mu_j}{x_{j,\max} - x_{j,\min}}
$$

where $\mu_j$ is the mean of feature $j$. This centers features around zero, typically in the range $[-1, 1]$.

**Example:** If $\mu_1 = 600$, then:

$$
x_1^{\text{normalized}} = \frac{x_1 - 600}{2000 - 300}
$$

**Method 3 — Z-score normalization:**

$$
x_j^{\text{normalized}} = \frac{x_j - \mu_j}{\sigma_j}
$$

where $\sigma_j$ is the **standard deviation** of feature $j$. This produces features with mean 0 and standard deviation 1.

**Rule of thumb:** Aim for features in roughly the range $[-1, 1]$ to $[-3, 3]$. Features in this range are fine as-is. Features with very large or very small ranges (e.g., $[-100, 100]$ or $[-0.001, 0.001]$) should be rescaled.

**There's almost never any harm in applying feature scaling.** When in doubt, just do it.

```python
def z_score_normalize(X):
    """
    Z-score normalize features.
    
    Parameters:
        X: input features, shape (m, n)
    
    Returns:
        X_norm: normalized features
        mu: mean of each feature
        sigma: standard deviation of each feature
    """
    mu = np.mean(X, axis=0)       # Mean of each column
    sigma = np.std(X, axis=0)     # Std dev of each column
    X_norm = (X - mu) / sigma     # Vectorized normalization
    return X_norm, mu, sigma
```

---

### 5.2 Checking Gradient Descent Convergence

To make sure gradient descent is working properly, plot the **learning curve**: the cost $J$ as a function of the iteration number.

**What a good learning curve looks like:**

- Cost $J$ **decreases after every single iteration**
- The curve eventually flattens out (levels off), indicating convergence
- Once the curve is flat, the parameters are no longer changing significantly

**Red flags:**

- If $J$ **increases** after an iteration: the learning rate $\alpha$ is too large, or there's a bug in the code
- If $J$ **oscillates** (goes up and down): also suggests $\alpha$ is too large

**Automatic convergence test:** Let $\varepsilon$ be a small number (e.g., $10^{-3}$). If $J$ decreases by less than $\varepsilon$ in one iteration, declare convergence. In practice, choosing the right $\varepsilon$ is difficult — most practitioners prefer to inspect the learning curve visually.

**Note:** The number of iterations needed for convergence varies enormously between applications — it could be 30 iterations or 100,000. There is no way to predict this in advance; hence the learning curve plot.

---

### 5.3 Choosing the Learning Rate

**Debugging with a tiny $\alpha$:**

If gradient descent is not working, try setting $\alpha$ to a very small number (e.g., $10^{-7}$). If the cost still doesn't decrease on every iteration, there is likely a **bug in the code**. A sufficiently small $\alpha$ should guarantee that $J$ decreases each iteration (just very slowly).

**Practical approach — try a range of values:**

Start with a small value and increase by roughly 3x each time:

$$
\ldots, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, \ldots
$$

For each value, run gradient descent for a handful of iterations and plot the learning curve. Pick the largest value of $\alpha$ where the cost decreases consistently.

---

### 5.4 Feature Engineering

Feature engineering means using **domain knowledge** to create new features that make the learning algorithm more effective.

**Example:** Predicting house price from lot dimensions.

Given: $x_1$ = frontage (width) and $x_2$ = depth of the lot.

A basic model uses both:

$$
f(\vec{x}) = w_1 x_1 + w_2 x_2 + b
$$

But you might realize that the **area** of the lot is more predictive of price. So you create a new feature:

$$
x_3 = x_1 \times x_2 \quad (\text{area})
$$

Now the model can learn:

$$
f(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
$$

The algorithm decides (via the learned parameters) whether frontage, depth, or area is most predictive. Feature engineering often involves **transforming or combining** existing features based on domain expertise.

---

### 5.5 Polynomial Regression

Polynomial regression uses **powers of features** to fit curves (non-linear functions) to data.

**Example:** If a straight line doesn't fit the housing data well, you might try:

**Quadratic:**

$$
f(\vec{x}) = w_1 x + w_2 x^2 + b
$$

This can capture curvature, but a quadratic eventually turns back down, which doesn't make sense for house prices (bigger houses should cost more).

**Cubic:**

$$
f(\vec{x}) = w_1 x + w_2 x^2 + w_3 x^3 + b
$$

This can capture more complex shapes and doesn't necessarily turn back down.

**Square root:**

$$
f(\vec{x}) = w_1 x + w_2 \sqrt{x} + b
$$

The square root grows but flattens over time, which may match real-world patterns.

**Important:** When using polynomial features, **feature scaling becomes critical**. If $x$ ranges from 1 to 1,000:

- $x^2$ ranges from 1 to 1,000,000
- $x^3$ ranges from 1 to 1,000,000,000

These vastly different scales will cause gradient descent to struggle if features are not normalized.

```python
# Creating polynomial features
def polynomial_features(X, degree):
    """
    Create polynomial features up to the given degree.
    
    Parameters:
        X: original feature, shape (m, 1)
        degree: highest power to include
    
    Returns:
        X_poly: feature matrix with columns [x, x^2, ..., x^degree]
    """
    X_poly = np.zeros((X.shape[0], degree))
    for d in range(1, degree + 1):
        X_poly[:, d-1] = X[:, 0] ** d
    return X_poly
```

---

## 6. Logistic Regression

### 6.1 Binary Classification

In binary classification, the output $y$ can only be **one of two values**: 0 or 1.

**Examples:**

| Problem | Class 0 (Negative) | Class 1 (Positive) |
|---|---|---|
| Spam detection | Not spam | Spam |
| Fraud detection | Legitimate | Fraudulent |
| Tumor diagnosis | Benign | Malignant |

By convention:

- **Negative class** (0): absence of the thing we're looking for
- **Positive class** (1): presence of the thing we're looking for

The terms "positive" and "negative" do not imply good vs. bad. They simply indicate absence vs. presence.

---

### 6.2 Why Not Linear Regression for Classification?

You might think: "Why not just apply linear regression and use a threshold (e.g., 0.5) to classify?"

**Problem:** Linear regression can be **easily distorted by outliers**.

Consider a tumor classification dataset where most tumors are small. A linear regression line with a 0.5 threshold might work reasonably well. But if you add one more data point — a very large benign tumor far to the right — the best-fit line shifts, moving the decision boundary to the right and causing **misclassification** of previously correct examples.

Adding one data point that shouldn't change the classification logic has fundamentally broken the model. This is why **linear regression is not appropriate for classification**.

---

### 6.3 The Sigmoid Function

The **sigmoid function** (also called the logistic function) is the foundation of logistic regression. It maps any real number to a value between 0 and 1, making it perfect for modeling probabilities.

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

where $e \approx 2.718$ is Euler's number.

**Key properties:**

- As $z \to +\infty$: $g(z) \to 1$ (because $e^{-z} \to 0$)
- As $z \to -\infty$: $g(z) \to 0$ (because $e^{-z} \to \infty$)
- At $z = 0$: $g(0) = \frac{1}{1+1} = 0.5$
- The output is **always between 0 and 1**
- The function has an S-shaped curve

```python
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid function.
    Works element-wise on numpy arrays.
    """
    return 1 / (1 + np.exp(-z))
```

---

### 6.4 The Logistic Regression Model

Logistic regression is built in two steps:

**Step 1:** Compute a linear combination:

$$
z = \vec{w} \cdot \vec{x} + b
$$

**Step 2:** Apply the sigmoid function:

$$
f_{\vec{w},b}(\vec{x}) = g(z) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}
$$

**Interpreting the output:**

The output of logistic regression is interpreted as the **probability** that $y = 1$ given the input $\vec{x}$:

$$
f_{\vec{w},b}(\vec{x}) = P(y = 1 \mid \vec{x}; \vec{w}, b)
$$

For example, if a patient has a tumor of a certain size and the model outputs 0.7, this means there is a **70% chance** the tumor is malignant. Since $y$ must be either 0 or 1:

$$
P(y = 0) + P(y = 1) = 1
$$

So a 70% chance of malignant implies a 30% chance of benign.

**Note on the name:** Despite having "regression" in its name, logistic regression is used for **classification**. The name is historical.

---

### 6.5 Decision Boundaries

To make a binary prediction (0 or 1) from the continuous probability output, we typically set a **threshold** at 0.5:

- If $f_{\vec{w},b}(\vec{x}) \geq 0.5$, predict $\hat{y} = 1$
- If $f_{\vec{w},b}(\vec{x}) < 0.5$, predict $\hat{y} = 0$

**When does $f \geq 0.5$?**

Since $f = g(z)$ and $g(z) \geq 0.5$ when $z \geq 0$:

$$
\hat{y} = 1 \quad \text{when} \quad \vec{w} \cdot \vec{x} + b \geq 0
$$

$$
\hat{y} = 0 \quad \text{when} \quad \vec{w} \cdot \vec{x} + b < 0
$$

The **decision boundary** is the set of points where $\vec{w} \cdot \vec{x} + b = 0$.

**Example 1 — Linear decision boundary:**

With features $x_1$, $x_2$ and parameters $w_1 = 1$, $w_2 = 1$, $b = -3$:

$$
z = x_1 + x_2 - 3
$$

The decision boundary is:

$$
x_1 + x_2 = 3
$$

This is a straight line. Points where $x_1 + x_2 \geq 3$ are classified as 1; points where $x_1 + x_2 < 3$ are classified as 0.

**Example 2 — Circular decision boundary (using polynomial features):**

With $z = x_1^2 + x_2^2 - 1$ and parameters $w_1 = 1$, $w_2 = 1$, $b = -1$:

The decision boundary is:

$$
x_1^2 + x_2^2 = 1
$$

This is a **circle** of radius 1 centered at the origin. Points outside the circle are classified as 1; points inside as 0.

**Even more complex boundaries** are possible with higher-order polynomial features — ellipses, figure-eights, and arbitrary shapes. However, if you only use the original features $x_1, x_2, \ldots, x_n$ (without polynomial terms), the decision boundary will always be a straight line (or hyperplane).

---

### 6.6 Cost Function for Logistic Regression

**Why not use the squared error cost function?**

If we plug the sigmoid function into the squared error cost function, the resulting surface becomes **non-convex** — full of local minima where gradient descent can get stuck. We need a different cost function.

**The logistic loss function:**

$$
L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = \begin{cases} -\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 1 \\ -\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 0 \end{cases}
$$

**Why this makes sense — Case $y = 1$:**

The loss is $-\log(f)$. Since $f$ is always between 0 and 1:

- If $f \approx 1$ (correctly predicts malignant): $-\log(1) = 0$ — **zero loss**
- If $f \approx 0$ (incorrectly predicts benign): $-\log(0) \to \infty$ — **infinite loss**

The loss increases as the prediction moves further from the true label, creating a strong incentive to predict correctly.

**Why this makes sense — Case $y = 0$:**

The loss is $-\log(1 - f)$:

- If $f \approx 0$ (correctly predicts benign): $-\log(1) = 0$ — **zero loss**
- If $f \approx 1$ (incorrectly predicts malignant): $-\log(0) \to \infty$ — **infinite loss**

Again, the loss grows as predictions diverge from the true label.

**The overall cost function** averages the loss over all training examples:

$$
J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})
$$

This cost function is **convex**, guaranteeing that gradient descent converges to the global minimum. It is derived from the statistical principle of **maximum likelihood estimation**.

---

### 6.7 Simplified Loss Function

Since $y$ is always either 0 or 1, we can write the two-case loss function as a single expression:

$$
L(f, y) = -y \log(f) - (1 - y) \log(1 - f)
$$

**Verification:**

- When $y = 1$: $L = -1 \cdot \log(f) - 0 \cdot \log(1 - f) = -\log(f)$ ✓
- When $y = 0$: $L = -0 \cdot \log(f) - 1 \cdot \log(1 - f) = -\log(1 - f)$ ✓

**The complete cost function for logistic regression:**

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) \right]
$$

This is the standard form used universally to train logistic regression models.

```python
def logistic_cost(X, y, w, b):
    """
    Compute the logistic regression cost function.
    
    Parameters:
        X: input features, shape (m, n)
        y: true labels (0 or 1), shape (m,)
        w: weights, shape (n,)
        b: bias (scalar)
    
    Returns:
        cost: the logistic cost (scalar)
    """
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)   # Predictions
    
    # Clip to avoid log(0) errors
    epsilon = 1e-7
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    
    cost = -(1/m) * np.sum(
        y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb)
    )
    return cost
```

---

### 6.8 Gradient Descent for Logistic Regression

The gradient descent algorithm for logistic regression looks **remarkably similar** to linear regression:

$$
w_j := w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

**These equations look identical to linear regression!** But they are fundamentally different because $f_{\vec{w},b}(\vec{x})$ is defined differently:

- **Linear regression:** $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
- **Logistic regression:** $f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$

The same formula produces different behaviors because the model function $f$ itself is different.

**Feature scaling** also applies to logistic regression and can speed up convergence.

```python
def gradient_descent_logistic(X, y, w, b, alpha, num_iterations):
    """
    Gradient descent for logistic regression.
    
    Parameters:
        X: input features, shape (m, n)
        y: true labels (0 or 1), shape (m,)
        w: initial weights, shape (n,)
        b: initial bias (scalar)
        alpha: learning rate
        num_iterations: number of iterations
    
    Returns:
        w, b: optimized parameters
    """
    m, n = X.shape
    
    for i in range(num_iterations):
        # Compute predictions using sigmoid
        z = np.dot(X, w) + b
        f_wb = sigmoid(z)                   # shape (m,)
        
        # Compute error
        error = f_wb - y                     # shape (m,)
        
        # Compute gradients (vectorized)
        dj_dw = (1/m) * np.dot(X.T, error)  # shape (n,)
        dj_db = (1/m) * np.sum(error)        # scalar
        
        # Simultaneous update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b
```

---

## 7. Overfitting and Regularization

### 7.1 The Problem of Overfitting

Overfitting and underfitting are two of the most important challenges in machine learning. Understanding them is critical for building models that **generalize well** to new, unseen data.

**Underfitting (High Bias)**

A model that is too simple to capture the underlying patterns in the data. It performs poorly on both training data and new data.

**Example:** Fitting a straight line to clearly curved housing price data. The model has a strong **preconception** (bias) that the relationship is linear, and it cannot overcome this even with compelling data to the contrary.

**"Just Right" (Good Generalization)**

A model that captures the true underlying pattern without memorizing noise. It performs well on both training data and new data.

**Example:** Fitting a quadratic function to housing data that naturally curves and flattens.

**Overfitting (High Variance)**

A model that is too complex and memorizes the training data, including noise and outliers. It performs **extremely well** on training data (possibly with zero error) but **poorly on new data**.

**Example:** Fitting a 4th-order polynomial that passes through every training point but creates a wiggly, unrealistic curve. A model that wiggles to fit every data point might predict that a large house is cheaper than a small one.

**Why "high variance"?** If the training set were slightly different (one house priced a little differently), the overfitting model would produce a completely different curve. The predictions are **highly variable** depending on the specific training data.

**Overfitting in classification** follows the same pattern. A simple logistic regression with linear features may underfit (straight-line decision boundary). Quadratic features give a reasonable elliptical boundary. Very high-order polynomials create an overly contorted decision boundary that perfectly classifies training data but won't generalize.

The goal is to be like Goldilocks — finding a model that is **neither too simple nor too complex**, but "just right."

---

### 7.2 Addressing Overfitting

There are three main strategies for combating overfitting:

**Strategy 1 — Get more training data**

With more examples, even a complex model will be forced to find a function that generalizes rather than memorizing specific points. This is the **#1 tool** against overfitting, but it isn't always feasible (maybe only so many houses have been sold in a given area).

**Strategy 2 — Feature selection (use fewer features)**

If you have many features (e.g., 100 features for house prediction) but relatively few training examples, reducing the feature set can help. Select only the most relevant features using domain knowledge or automated methods.

**Downside:** You might throw away useful information. Maybe all 100 features genuinely contribute to the prediction.

**Strategy 3 — Regularization**

Regularization keeps all features but **gently reduces** the impact of each by shrinking the parameter values. Instead of eliminating features (setting parameters to exactly 0), it encourages parameters to be **small but not necessarily zero**.

With small parameter values, the model can use all features but is less likely to create the extreme, wiggly curves that characterize overfitting.

---

### 7.3 Regularization Intuition

**The core idea:** If parameters $w_j$ are small, the model is simpler and smoother.

**Concrete example:** Suppose you have a 4th-order polynomial:

$$
f(x) = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b
$$

If you add a penalty for large $w_3$ and $w_4$ to the cost function:

$$
J = \frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2 + 1000 \cdot w_3^2 + 1000 \cdot w_4^2
$$

Then to minimize $J$, the algorithm must keep $w_3$ and $w_4$ close to 0. This effectively cancels out the $x^3$ and $x^4$ terms, producing a fit closer to a quadratic — which generalizes better.

**In practice,** we don't know which parameters to penalize specifically. So we penalize **all** of them.

**The regularized cost function:**

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

where:

- $\lambda$ (lambda) is the **regularization parameter** — a positive number you choose
- The first term encourages the model to **fit the data well**
- The second term (regularization term) encourages the parameters to be **small**
- $\lambda$ controls the **trade-off** between these two goals
- We divide by $2m$ so the regularization term scales with the training set size

**By convention,** we do NOT regularize the bias parameter $b$ — only $w_1$ through $w_n$.

**Effect of $\lambda$:**

- $\lambda = 0$: No regularization → high risk of overfitting
- $\lambda$ very large (e.g., $10^{10}$): All $w_j \approx 0$, so $f \approx b$ → underfitting (horizontal line)
- $\lambda$ "just right": Good balance between fitting data and keeping parameters small

---

### 7.4 Regularized Linear Regression

**The regularized cost function:**

$$
J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

**The gradient descent updates become:**

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

Notice that only the $w_j$ update changes — the $b$ update is the same as before because we don't regularize $b$.

**Alternative way to understand the $w_j$ update:**

Rearranging the update rule:

$$
w_j := w_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
$$

The first factor $\left(1 - \alpha \frac{\lambda}{m}\right)$ is a number **slightly less than 1** (e.g., 0.9998). On every iteration, $w_j$ is first multiplied by this shrinkage factor, then the usual gradient step is applied. This is why regularization **shrinks the parameters a little bit each iteration**.

**Example calculation:** With $\alpha = 0.01$, $\lambda = 1$, $m = 50$:

$$
1 - \alpha \frac{\lambda}{m} = 1 - \frac{0.01 \times 1}{50} = 1 - 0.0002 = 0.9998
$$

So each iteration multiplies $w_j$ by 0.9998 before applying the gradient update.

```python
def gradient_descent_regularized(X, y, w, b, alpha, lambda_, num_iterations):
    """
    Regularized gradient descent for linear regression.
    
    Parameters:
        X: input features, shape (m, n)
        y: target values, shape (m,)
        w: initial weights, shape (n,)
        b: initial bias (scalar)
        alpha: learning rate
        lambda_: regularization parameter
        num_iterations: number of iterations
    
    Returns:
        w, b: optimized parameters
    """
    m, n = X.shape
    
    for i in range(num_iterations):
        f_wb = np.dot(X, w) + b
        error = f_wb - y
        
        # Gradients with regularization term
        dj_dw = (1/m) * np.dot(X.T, error) + (lambda_/m) * w
        dj_db = (1/m) * np.sum(error)  # b is NOT regularized
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b
```

---

### 7.5 Regularized Logistic Regression

Logistic regression is also prone to overfitting, especially with many polynomial features or when there are many features relative to training examples.

**The regularized cost function for logistic regression:**

$$
J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)}) \log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

This is the same logistic cost function from before, plus the regularization term $\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$.

**The gradient descent updates:**

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
$$

$$
b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)
$$

These look **identical** to regularized linear regression. The crucial difference is that $f_{\vec{w},b}(\vec{x})$ is defined as the **sigmoid function** for logistic regression, not the linear function.

As with unregularized logistic regression, only the parameters $w_j$ are regularized — **not** $b$.

```python
def gradient_descent_regularized_logistic(X, y, w, b, alpha, lambda_, num_iterations):
    """
    Regularized gradient descent for logistic regression.
    
    Parameters:
        X: input features, shape (m, n)
        y: true labels (0 or 1), shape (m,)
        w: initial weights, shape (n,)
        b: initial bias (scalar)
        alpha: learning rate
        lambda_: regularization parameter
        num_iterations: number of iterations
    
    Returns:
        w, b: optimized parameters
    """
    m, n = X.shape
    
    for i in range(num_iterations):
        # Key difference: use sigmoid for predictions
        z = np.dot(X, w) + b
        f_wb = sigmoid(z)
        
        error = f_wb - y
        
        # Gradients with regularization
        dj_dw = (1/m) * np.dot(X.T, error) + (lambda_/m) * w
        dj_db = (1/m) * np.sum(error)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b
```

---

## Quick Reference: Algorithm Comparison

| Aspect | Linear Regression | Logistic Regression |
|---|---|---|
| **Task** | Regression (predict numbers) | Classification (predict categories) |
| **Model** | $f = \vec{w} \cdot \vec{x} + b$ | $f = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$ |
| **Output range** | $(-\infty, +\infty)$ | $(0, 1)$ |
| **Cost function** | Mean Squared Error | Log Loss (Cross-Entropy) |
| **Gradient formula** | Same structure | Same structure |
| **Difference** | Definition of $f$ | Definition of $f$ |
| **Cost shape** | Always convex (bowl) | Convex with log loss |
| **Regularization** | Add $\frac{\lambda}{2m}\sum w_j^2$ | Add $\frac{\lambda}{2m}\sum w_j^2$ |

---

## Key Takeaways

1. **Supervised learning** learns input-to-output mappings from labeled data, with **regression** predicting continuous values and **classification** predicting discrete categories.

2. **The cost function** measures model performance. For regression, we use the squared error cost function. For classification, we use the log loss (cross-entropy) cost function.

3. **Gradient descent** iteratively adjusts parameters to minimize the cost function. It requires choosing a learning rate $\alpha$ and uses simultaneous parameter updates.

4. **Vectorization** makes implementations both shorter and dramatically faster by leveraging parallel hardware through libraries like NumPy.

5. **Feature scaling** (normalization) is essential for efficient gradient descent, especially when features have different ranges.

6. **Feature engineering** and **polynomial regression** allow linear models to capture non-linear relationships in data.

7. **Overfitting** occurs when a model is too complex and memorizes training data rather than learning generalizable patterns. It is addressed through more data, feature selection, or regularization.

8. **Regularization** adds a penalty for large parameter values, encouraging simpler models that generalize better. The regularization parameter $\lambda$ controls the trade-off between fitting the data and keeping parameters small.

---

> **End of Course 1 Notes**
>
> These concepts — cost functions, gradient descent, sigmoid functions, and regularization — form the building blocks for understanding neural networks and deep learning, covered in Course 2 of the specialization.
