# Advanced Machine Learning: Neural Networks, Practical Advice & Tree Ensembles

> **Comprehensive Notes — Andrew Ng Machine Learning Specialization, Course 2**
> Covers neural networks, training techniques, practical system-building advice, decision trees, and ensemble methods.

---

## Table of Contents

1. [Neural Networks: Architecture & Motivation](#1-neural-networks-architecture--motivation)
2. [Forward Propagation](#2-forward-propagation)
3. [Activation Functions](#3-activation-functions)
4. [Training Neural Networks](#4-training-neural-networks)
5. [Backpropagation](#5-backpropagation)
6. [Vectorized Implementation](#6-vectorized-implementation)
7. [Practical ML Advice: Bias & Variance](#7-practical-ml-advice-bias--variance)
8. [Train / CV / Test Splits & Model Selection](#8-train--cv--test-splits--model-selection)
9. [Regularization & Debugging Strategies](#9-regularization--debugging-strategies)
10. [The Adam Optimizer](#10-the-adam-optimizer)
11. [Evaluation Metrics: Precision, Recall & F1](#11-evaluation-metrics-precision-recall--f1)
12. [Data Augmentation](#12-data-augmentation)
13. [Convolutional Layers (Intro)](#13-convolutional-layers-intro)
14. [Decision Trees](#14-decision-trees)
15. [Ensemble Methods: Random Forests & XGBoost](#15-ensemble-methods-random-forests--xgboost)
16. [When to Use Neural Networks vs. Decision Trees](#16-when-to-use-neural-networks-vs-decision-trees)

---

## 1. Neural Networks: Architecture & Motivation

### Why Neural Networks?

Traditional ML algorithms like logistic regression and linear regression hit a performance ceiling — feeding them more data beyond a certain point yields diminishing returns. Neural networks, by contrast, scale: a **larger network + more data = better performance**, with no hard ceiling observed in practice. This is the core empirical observation that ignited the deep learning revolution around 2012.

Biologically, neurons in the brain take inputs through dendrites, compute something, and fire an output signal through the axon. An **artificial neuron** is a mathematical abstraction of this: it takes a vector of numbers, applies a weighted sum followed by a non-linear function, and outputs a scalar. By stacking millions of these units, we get networks capable of remarkable feats.

### Network Architecture

A neural network consists of:

- An **input layer** (layer 0): the raw feature vector $\mathbf{x}$
- One or more **hidden layers** (layers 1 through $L-1$): learned intermediate representations
- An **output layer** (layer $L$): produces the final prediction

Each layer $l$ has $n^{[l]}$ units. Each unit $j$ in layer $l$ computes:

$$z_j^{[l]} = \mathbf{w}_j^{[l] \top} \mathbf{a}^{[l-1]} + b_j^{[l]}$$

$$a_j^{[l]} = g\!\left(z_j^{[l]}\right)$$

where $g$ is a non-linear **activation function**, $`\mathbf{w}_j^{[l]}`$ is the weight vector for neuron $j$ in layer $l$, $`b_j^{[l]}`$ is its bias, and $`\mathbf{a}^{[l-1]}`$ is the activation vector from the previous layer (or $\mathbf{x}$ if $l = 1$).

**Why do we need multiple layers?** Each layer learns a progressively more abstract representation. In image recognition, layer 1 might detect edges, layer 2 textures, layer 3 shapes, and later layers entire objects. This hierarchical feature learning is what makes deep networks so powerful.

---

## 2. Forward Propagation

Forward propagation is the algorithm for computing predictions: information flows forward through the network, layer by layer, from input to output.

### Algorithm

Given input $\mathbf{x}$, set $`\mathbf{a}^{[0]} = \mathbf{x}`$, then for each layer $l = 1, \ldots, L$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

The final output is $`\hat{y} = \mathbf{a}^{[L]}`$.

**Notation key:** $`\mathbf{W}^{[l]}`$ has shape $(n^{[l]} \times n^{[l-1]})$; each row is the weight vector for one neuron in layer $l$.

### Example: Handwritten Digit Recognition

Consider an $8 \times 8$ image (64 pixels) passed through a network with layers of 25 → 15 → 1 units:

$$\mathbf{x} \in \mathbb{R}^{64}$$

$$\mathbf{a}^{[1]} = g\!\left(\mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}\right), \quad \mathbf{a}^{[1]} \in \mathbb{R}^{25}$$

$$\mathbf{a}^{[2]} = g\!\left(\mathbf{W}^{[2]}\mathbf{a}^{[1]} + \mathbf{b}^{[2]}\right), \quad \mathbf{a}^{[2]} \in \mathbb{R}^{15}$$

$$\hat{y} = \sigma\!\left(\mathbf{w}^{[3] \top}\mathbf{a}^{[2]} + b^{[3]}\right)$$

### Python / TensorFlow Implementation

```python
import numpy as np
import tensorflow as tf

# Manual forward pass (single layer, pedagogical)
def dense(a_in, W, b, activation):
    """
    Computes the output of one dense layer.
    a_in: (n_in,)   — input activation vector
    W:    (n_in, n_out) — weight matrix
    b:    (n_out,)  — bias vector
    """
    z = np.dot(a_in, W) + b      # shape: (n_out,)
    a_out = activation(z)         # element-wise non-linearity
    return a_out

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Full forward propagation (manual, two hidden layers)
def forward_prop(x, W1, b1, W2, b2, W3, b3):
    a1 = dense(x,  W1, b1, sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    a3 = dense(a2, W3, b3, sigmoid)
    return a3

# TensorFlow equivalent — much simpler
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu', input_shape=(64,)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])
```

---

## 3. Activation Functions

Activation functions introduce **non-linearity**. Without them, a neural network with any number of layers is just a linear transformation — equivalent to a single-layer model. Non-linear activations are what allow networks to learn complex, curved decision boundaries.

### 3.1 Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Output range: $(0, 1)$. Historically used everywhere, but now mainly reserved for the **output layer of binary classification** problems, since its output can be interpreted as a probability. Suffers from the **vanishing gradient problem** in deep networks — gradients become extremely small in early layers, slowing learning.

### 3.2 ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

Output range: $[0, \infty)$. The **default choice for hidden layers**. Fast to compute, does not saturate for positive inputs (no vanishing gradient problem on the positive side), and empirically works very well. The only issue is the "dying ReLU" problem where neurons can get stuck outputting zero permanently.

### 3.3 Linear (Identity)

$$g(z) = z$$

Used in the **output layer for regression problems** where the target $y$ can be any real number (positive or negative). Using this in a hidden layer is generally pointless — it collapses the layer to a linear transformation.

### 3.4 Softmax

For multi-class classification with $K$ classes, the softmax function converts a vector of logits $\mathbf{z} \in \mathbb{R}^K$ into a probability distribution:

$$a_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}, \quad j = 1, \ldots, K$$

Note that $\sum_{j=1}^{K} a_j = 1$ and $a_j \in (0, 1)$, so the output is a valid probability distribution over $K$ classes. The associated loss is the **categorical cross-entropy**:

$$\mathcal{L} = -\sum_{j=1}^{K} y_j \log a_j$$

where $y_j = 1$ for the true class and 0 otherwise.

**Numerical stability tip:** in TensorFlow, use `from_logits=True` in the loss function and do not apply softmax in the output layer. TensorFlow will compute `softmax + cross-entropy` jointly in a numerically stable way.

```python
# Numerically stable training — preferred
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')  # logits, no softmax here
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# At inference time, apply softmax manually if you need probabilities
logits = model(x_test)
probs  = tf.nn.softmax(logits)
```

### 3.5 Other Activations

**Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, range $(-1, 1)$. Zero-centered, can be better than sigmoid for hidden layers, but still suffers from vanishing gradients.

**Leaky ReLU:** $g(z) = \max(\alpha z, z)$ with small $\alpha$ (e.g., 0.01). Fixes the dying ReLU problem by allowing a small gradient even for negative inputs.

**Swish:** $g(z) = z \cdot \sigma(z)$. A smooth, non-monotonic function that sometimes outperforms ReLU, especially in very deep networks.

### Activation Choice Summary

| Layer | Task | Recommended Activation |
|---|---|---|
| Hidden layers | Any | **ReLU** (default) |
| Output | Binary classification | Sigmoid |
| Output | Multi-class classification | Softmax |
| Output | Regression (any real number) | Linear |
| Output | Regression (non-negative) | ReLU |

---

## 4. Training Neural Networks

Training a neural network means finding the parameters $\mathbf{W}^{[l]}, \mathbf{b}^{[l]}$ for all layers $l$ that minimize a cost function $J$.

### Cost Functions

**Binary cross-entropy** (for binary classification):

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\hat{y}^{(i)} + \left(1 - y^{(i)}\right) \log\left(1 - \hat{y}^{(i)}\right) \right]$$

**Why this loss?** It's the negative log-likelihood under a Bernoulli model. Minimizing it is equivalent to maximum likelihood estimation. It penalizes confident wrong predictions exponentially harder than mild errors, which is what we want.

**Mean squared error** (for regression):

$$J = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

### Gradient Descent

The update rule for each parameter $w$:

$$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$

$$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

where $\alpha$ is the **learning rate**. All parameters are updated simultaneously. The gradient $\partial J / \partial w$ tells us how much the cost increases per unit increase in $w$ — moving in the negative gradient direction decreases cost.

### TensorFlow Training Loop

```python
# The three-step TensorFlow recipe
model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])

# Step 2: Compile — specify optimizer and loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 3: Fit — run gradient descent for `epochs` iterations
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_cv, y_cv))
```

---

## 5. Backpropagation

Backpropagation is the algorithm for efficiently computing the gradients $\partial J / \partial \mathbf{W}^{[l]}$ and $\partial J / \partial \mathbf{b}^{[l]}$ for every layer. It is an application of the **chain rule of calculus**, propagating the gradient backward from the output layer to the input.

### Why Do We Need It?

A network with millions of parameters needs a gradient with respect to every single parameter. Computing each gradient naively (by finite differences) would require a separate forward pass per parameter — computationally prohibitive. Backpropagation computes **all gradients in a single backward pass** by cleverly reusing intermediate computations.

### The Chain Rule in Action

Define the error signal at layer $l$ as:

$$\boldsymbol{\delta}^{[l]} = \frac{\partial J}{\partial \mathbf{z}^{[l]}}$$

**Output layer ($l = L$):** for binary cross-entropy with sigmoid output:

$$\delta^{[L]} = \hat{y} - y$$

**Hidden layers** (backpropagating the error):

$$\boldsymbol{\delta}^{[l]} = \left(\mathbf{W}^{[l+1] \top} \boldsymbol{\delta}^{[l+1]}\right) \odot g'^{[l]}\!\left(\mathbf{z}^{[l]}\right)$$

where $\odot$ is element-wise multiplication and $g'^{[l]}$ is the derivative of the activation function at layer $l$.

**Gradients for the parameters:**

$$\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \boldsymbol{\delta}^{[l]} \mathbf{a}^{[l-1] \top}$$

$$\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \boldsymbol{\delta}^{[l](i)}$$

### Activation Derivatives

For ReLU: $g'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$

For sigmoid: $g'(z) = \sigma(z)(1 - \sigma(z))$

### Manual Backprop (Pedagogical Example)

```python
def relu_derivative(z):
    return (z > 0).astype(float)

def backward_pass(X, Y, cache, W2, W3):
    """
    cache: dict containing z1, a1, z2, a2, z3, a3 from forward pass
    """
    m = X.shape[0]

    # Output layer gradient (sigmoid + cross-entropy combined)
    dz3 = cache['a3'] - Y                     # shape: (m, 1)

    dW3 = (1/m) * cache['a2'].T @ dz3
    db3 = (1/m) * np.sum(dz3, axis=0)

    # Hidden layer 2
    da2 = dz3 @ W3.T
    dz2 = da2 * relu_derivative(cache['z2'])

    dW2 = (1/m) * cache['a1'].T @ dz2
    db2 = (1/m) * np.sum(dz2, axis=0)

    # Hidden layer 1
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(cache['z1'])

    dW1 = (1/m) * X.T @ dz1
    db1 = (1/m) * np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2, dW3, db3
```

In practice, TensorFlow's `GradientTape` or the `model.fit()` call handles all of this automatically using automatic differentiation.

---

## 6. Vectorized Implementation

The key computational insight that makes deep learning scalable: replace scalar loops with **matrix operations**, which can be massively parallelized on GPUs.

### From Loops to Matrix Multiplication

Consider computing activations for a layer with $n^{[l]}$ neurons and a batch of $m$ training examples. The naive loop:

```python
# Slow: loop over all m examples
a_out = np.zeros((m, n_out))
for i in range(m):
    z = np.dot(a_in[i], W) + b   # (n_out,)
    a_out[i] = relu(z)
```

The vectorized version:

```python
# Fast: one matrix operation
Z = a_in @ W + b          # (m, n_in) @ (n_in, n_out) = (m, n_out)
A = relu(Z)               # element-wise, shape (m, n_out)
```

This replaces $m$ sequential dot products with a single matrix multiply, which GPU hardware executes in near-parallel. On modern hardware this can be **hundreds of times faster**.

### Matrix Shapes Reference

| Variable | Shape | Meaning |
|---|---|---|
| $`\mathbf{X}`$ | $(m, n^{[0]})$ | Training batch, $m$ examples |
| $`\mathbf{W}^{[l]}`$ | $(n^{[l-1]}, n^{[l]})$ | Weights of layer $l$ |
| $`\mathbf{b}^{[l]}`$ | $(1, n^{[l]})$ | Biases of layer $l$ (broadcast over $m$) |
| $`\mathbf{Z}^{[l]}`$ | $(m, n^{[l]})$ | Pre-activation values |
| $`\mathbf{A}^{[l]}`$ | $(m, n^{[l]})$ | Post-activation values |

### Full Vectorized Forward Pass

```python
def forward_pass_vectorized(X, params):
    """
    X: (m, n_features)
    params: dict of {'W1': ..., 'b1': ..., 'W2': ..., ...}
    Returns: output activations and a cache for backprop
    """
    cache = {}
    A = X

    L = len(params) // 2  # number of layers
    for l in range(1, L):
        W = params[f'W{l}']
        b = params[f'b{l}']
        Z = A @ W + b
        A = np.maximum(0, Z)          # ReLU for hidden layers
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    # Output layer (sigmoid for binary classification)
    W = params[f'W{L}']
    b = params[f'b{L}']
    Z = A @ W + b
    A = 1 / (1 + np.exp(-Z))         # sigmoid
    cache[f'Z{L}'] = Z
    cache[f'A{L}'] = A

    return A, cache
```

---

## 7. Practical ML Advice: Bias & Variance

Understanding bias and variance is arguably the most important meta-skill in applied machine learning. It tells you **exactly what to try next** when a model underperforms, preventing teams from wasting months on the wrong fix.

### Definitions

**Bias** is the error due to overly simplistic assumptions. A high-bias model underfits — it fails to capture the true underlying pattern in the data. Symptoms: high training error.

**Variance** is the error due to excessive sensitivity to training data fluctuations. A high-variance model overfits — it memorizes the training set but fails to generalize. Symptoms: low training error, high validation error.

Formally, for a model with expected prediction $\bar{f}(x)$ and true target $f(x)$:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\bar{f}(x) - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \bar{f}(x))^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible noise}}$$

This is the **bias-variance decomposition**. You cannot reduce irreducible noise. You can reduce bias (with a more expressive model) or variance (with regularization/more data), but classically there is a **trade-off**: reducing one tends to increase the other.

### Diagnosing Your Model

| $J_{\text{train}}$ | $J_{\text{cv}}$ | Diagnosis | Fix |
|---|---|---|---|
| High | High | High bias (underfit) | Larger model, more features, more epochs |
| Low | High | High variance (overfit) | More data, regularization, smaller model |
| Low | Low | Just right | Ship it |
| High | Much higher | Both bias AND variance | Complex model partially overfitting |

```python
# Quick diagnostic
train_error = evaluate(model, X_train, y_train)
cv_error    = evaluate(model, X_cv,    y_cv)

print(f"Train error: {train_error:.4f}")
print(f"CV error:    {cv_error:.4f}")

bias_problem     = train_error > acceptable_threshold
variance_problem = cv_error - train_error > acceptable_gap
```

### Learning Curves

Plotting $J_{\text{train}}$ and $J_{\text{cv}}$ as a function of **training set size** $m$ gives powerful diagnostics:

- **High bias:** Both curves converge to a high value — even with infinite data, the model can't do well. Adding more data won't help. Fix the model.
- **High variance:** Large gap between train and CV curves that narrows as $m$ increases — more data will help. The model is too complex relative to the available data.

```python
import matplotlib.pyplot as plt

train_sizes = [100, 500, 1000, 5000, 10000]
train_errors, cv_errors = [], []

for size in train_sizes:
    X_sub, y_sub = X_train[:size], y_train[:size]
    model.fit(X_sub, y_sub, epochs=50, verbose=0)
    train_errors.append(model.evaluate(X_sub,  y_sub,  verbose=0)[0])
    cv_errors.append(   model.evaluate(X_cv,   y_cv,   verbose=0)[0])

plt.plot(train_sizes, train_errors, label='Train error')
plt.plot(train_sizes, cv_errors,    label='CV error')
plt.xlabel('Training set size'); plt.ylabel('Cost'); plt.legend()
plt.title('Learning Curves')
plt.show()
```

---

## 8. Train / CV / Test Splits & Model Selection

### Why Three Splits?

You cannot use the test set to make **any** decisions during model development — doing so leaks information and gives you an overly optimistic estimate of generalization. The cross-validation (CV) set is used for all intermediate decisions (architecture, hyperparameters, regularization). The test set is touched only once at the very end to report an unbiased final estimate.

### Typical Split Ratios

**Large datasets** (millions of examples): 98% / 1% / 1% is fine — 1% of a million examples is plenty for reliable evaluation.

**Small/medium datasets**: 60% / 20% / 20% or 70% / 15% / 15%.

### Model Selection with the CV Set

Given candidate models (e.g., polynomial degrees 1 through 10, or neural network sizes), evaluate each on the CV set and pick the one with lowest $`J_{\text{cv}}`$. Then estimate its true generalization performance on the test set:

$$J_{\text{cv}}(w^{(d)}, b^{(d)}) = \frac{1}{2m_{\text{cv}}} \sum_{i=1}^{m_{\text{cv}}} \left( f_{w,b}\!\left(x_{\text{cv}}^{(i)}\right) - y_{\text{cv}}^{(i)} \right)^2$$

```python
from sklearn.model_selection import train_test_split

# First split: set aside test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Second split: training and cross-validation
X_train, X_cv, y_train, y_cv = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.80 = 0.20
)

print(f"Train: {len(X_train)}, CV: {len(X_cv)}, Test: {len(X_test)}")
```

---

## 9. Regularization & Debugging Strategies

### L2 Regularization (Weight Decay)

Regularization adds a penalty term to the cost function that discourages large weights, reducing overfitting:

$$J_{\text{reg}} = J + \frac{\lambda}{2m} \sum_{l=1}^{L} \|\mathbf{W}^{[l]}\|_F^2$$

where $`\|\mathbf{W}^{[l]}\|_F^2 = \sum_{j,k} (W_{jk}^{[l]})^2`$ is the Frobenius norm. The hyperparameter $\lambda \geq 0$ controls the trade-off: $\lambda = 0$ means no regularization; large $\lambda$ forces weights toward zero (high bias, low variance).

The gradient update with L2 regularization becomes:

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha\left(\frac{\partial J}{\partial \mathbf{W}^{[l]}} + \frac{\lambda}{m}\mathbf{W}^{[l]}\right) = \left(1 - \frac{\alpha \lambda}{m}\right)\mathbf{W}^{[l]} - \alpha \frac{\partial J}{\partial \mathbf{W}^{[l]}}$$

This is why L2 regularization is also called **weight decay** — the factor $(1 - \alpha\lambda/m)$ slightly shrinks weights at each step.

### Selecting $\lambda$ via Cross-Validation

```python
import numpy as np

lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]
cv_errors = []

for lam in lambdas:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(25, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(lam)),
        tf.keras.layers.Dense(15, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(lam)),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    loss = model.evaluate(X_cv, y_cv, verbose=0)
    cv_errors.append(loss)

best_lambda = lambdas[np.argmin(cv_errors)]
print(f"Best lambda: {best_lambda}")
```

### Systematic Debugging Checklist

When a model underperforms, work through this decision tree rather than guessing:

**If high training error (high bias):**
- Try a larger / deeper network
- Train for more epochs
- Add polynomial features or reduce regularization ($\lambda$)

**If large CV − train gap (high variance):**
- Collect more training data
- Increase regularization ($\lambda$)
- Use a smaller / shallower network
- Apply early stopping or dropout

**If both errors are high:**
- The problem is likely misspecified — revisit feature engineering and model architecture together.

---

## 10. The Adam Optimizer

### Motivation

Vanilla gradient descent uses a **single global learning rate** $\alpha$. This is problematic because:

- If $\alpha$ is too small, training is painfully slow.
- If $\alpha$ is too large, parameters oscillate around the minimum and may diverge.
- Different parameters may benefit from different learning rates — some gradients are consistently small, others large.

**Adam** (Adaptive Moment Estimation) solves this by maintaining a per-parameter adaptive learning rate.

### How Adam Works

Adam maintains two exponentially decaying averages:

**First moment** (mean of gradients — like momentum):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

**Second moment** (uncentered variance of gradients):

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

**Bias correction** (to avoid the estimates being too small early in training):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter update:**

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Intuition:** parameters with consistently large gradients get a smaller effective learning rate (the $\sqrt{\hat{v}_t}$ in the denominator grows large). Parameters with small or inconsistent gradients get a larger effective learning rate. This automatically scales the step size to the geometry of the loss surface.

**Default hyperparameters** (these rarely need tuning): $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

```python
# TensorFlow Adam — these are the defaults
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

model.compile(optimizer=optimizer, loss='binary_crossentropy')
```

---

## 11. Evaluation Metrics: Precision, Recall & F1

### Why Not Just Use Accuracy?

In **imbalanced datasets** (e.g., 99% negative, 1% positive for a rare disease), a model that always predicts "negative" achieves 99% accuracy — but is completely useless. We need metrics sensitive to the minority class.

### The Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

"Of all the cases we predicted as positive, how many actually were?" High precision means: when we raise an alarm, it's usually right. Useful when **false positives are costly** (e.g., unnecessary invasive treatment).

### Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

"Of all the actual positives, how many did we catch?" High recall means: we don't miss real cases. Useful when **false negatives are costly** (e.g., missing a dangerous disease).

### The Precision–Recall Trade-off

Adjusting the classification threshold (typically 0.5 for logistic regression / sigmoid output) moves along the precision–recall trade-off curve:

- **Higher threshold (e.g., 0.7):** predict positive only when very confident → higher precision, lower recall.
- **Lower threshold (e.g., 0.3):** predict positive more liberally → lower precision, higher recall.

### F1 Score

The harmonic mean of precision and recall, giving a single number that balances both:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean (rather than arithmetic mean) penalizes extreme imbalances between precision and recall — a model with precision = 1.0 and recall = 0.0 has F1 = 0, not 0.5.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

y_pred = (model.predict(X_test) >= 0.5).astype(int)

print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
```

---

## 12. Data Augmentation

### Why We Need It

More data generally leads to better generalization. But collecting and labeling real data is expensive. **Data augmentation** creates additional training examples synthetically by applying label-preserving transformations to existing data. The key principle: the transformation must not change the true label.

### Image Augmentation

Common transformations for images:

- Rotation (e.g., ±15°)
- Horizontal/vertical flipping
- Zooming and cropping
- Brightness and contrast adjustment
- Random erasing / cutout
- **Warping via a grid** (elastic distortion) — particularly effective for handwriting

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,          # rotate by up to 15 degrees
    horizontal_flip=True,       # flip left-right randomly
    zoom_range=0.1,             # zoom in/out by up to 10%
    width_shift_range=0.1,      # shift horizontally
    height_shift_range=0.1,     # shift vertically
    brightness_range=[0.8, 1.2] # vary brightness
)

# Fit to training data and use augmented batches during training
datagen.fit(X_train)
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=50,
          validation_data=(X_cv, y_cv))
```

### Audio Augmentation

For speech tasks:
- Add background noise (crowd, car, music) by mixing audio clips
- Time stretching / pitch shifting
- Adding reverb

### Important Caveat

Augmentation should add **representative** distortions — ones the model will actually encounter at test time. Adding purely random noise or unrealistic distortions may not help, and can even hurt if it misleads the model.

---

## 13. Convolutional Layers (Intro)

### Motivation: Beyond Dense Layers

In a standard **dense** (fully connected) layer, every neuron sees every input. For a $256 \times 256$ image (65,536 pixels), each neuron in the first hidden layer would have 65,536 weights — wasteful, and prone to overfitting.

A **convolutional layer** constrains each neuron to look at only a small **receptive field** (window) of the input. This offers two advantages:

1. **Parameter sharing:** every neuron in a convolutional layer uses the same weights (a learned filter), drastically reducing parameter count.
2. **Translation equivariance:** a feature detector that spots a horizontal edge in one part of the image can be reused everywhere in the image.

### How It Works

A neuron in a convolutional layer with receptive field of size $k \times k$ centered at position $(i, j)$ computes:

$$z_{i,j} = \sum_{p=0}^{k-1} \sum_{q=0}^{k-1} w_{p,q} \cdot x_{i+p,\, j+q} + b$$

The weight matrix $\mathbf{W} \in \mathbb{R}^{k \times k}$ is the **filter** (or kernel). Multiple filters are learned per layer, each detecting a different type of feature.

```python
# Convolutional neural network example
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Convolutional layers are the foundation of modern image models, and the architectural design choices (filter size, depth, number of filters) are an active research area. Transformer-based architectures have also extended similar ideas beyond vision.

---

## 14. Decision Trees

Decision trees are non-parametric models that partition the feature space with a series of axis-aligned splits, creating a tree structure. Despite being simple, they are highly interpretable and form the building blocks of the most powerful ensemble methods.

### Why Use Decision Trees?

They handle categorical features natively, require no feature scaling, are naturally interpretable ("if price > 50 AND quality = high, predict top seller"), and are computationally cheap to train. Their weakness is that a single tree tends to overfit — which tree ensembles fix.

### Building a Decision Tree: The Algorithm

At each node, we select the feature and threshold that **maximizes information gain** (or equivalently, minimizes post-split impurity). We recurse until a stopping criterion is met.

**Stopping criteria:**
- Maximum tree depth reached
- Fewer than $k$ examples in the node
- Information gain below threshold $\epsilon$

### Entropy: Measuring Impurity

Given a node where a fraction $p_1$ of examples belong to the positive class:

$$H(p_1) = -p_1 \log_2 p_1 - (1-p_1) \log_2(1-p_1)$$

Entropy is 0 when the node is pure ($p_1 = 0$ or $p_1 = 1$) and maximal (= 1 bit) when the classes are perfectly mixed ($p_1 = 0.5$). **We want to minimize entropy** — high-entropy nodes contain no useful signal.

```python
import numpy as np

def entropy(p1):
    """
    Compute entropy of a binary node.
    p1: fraction of positive examples (float in [0, 1])
    """
    if p1 == 0 or p1 == 1:
        return 0.0  # avoid log(0) — pure node has zero entropy
    p0 = 1 - p1
    return -p1 * np.log2(p1) - p0 * np.log2(p0)

# Examples
print(entropy(0.5))   # 1.0 — maximally impure
print(entropy(0.9))   # ~0.47 — mostly pure
print(entropy(1.0))   # 0.0 — perfectly pure
```

### Information Gain

Information gain measures how much a split reduces entropy:

$$IG = H(p_1^{\text{root}}) - \left( \frac{n_{\text{left}}}{n} H(p_1^{\text{left}}) + \frac{n_{\text{right}}}{n} H(p_1^{\text{right}}) \right)$$

We choose the feature that maximizes $IG$. The weighted sum accounts for the fact that larger child nodes matter more.

```python
def information_gain(y_parent, y_left, y_right):
    """
    Compute information gain of a binary split.
    y_parent, y_left, y_right: arrays of binary labels
    """
    n = len(y_parent)
    n_l, n_r = len(y_left), len(y_right)

    H_parent = entropy(y_parent.mean())
    H_left   = entropy(y_left.mean())  if n_l > 0 else 0
    H_right  = entropy(y_right.mean()) if n_r > 0 else 0

    weighted_entropy = (n_l / n) * H_left + (n_r / n) * H_right
    return H_parent - weighted_entropy

# Example: does splitting on "ear shape" help classify cats?
y_parent = np.array([1,1,1,1,0,0,0,0,0,0])  # 4 cats, 6 non-cats
y_left   = np.array([1,1,1,1,0])             # after split: 4 cats, 1 non-cat
y_right  = np.array([0,0,0,0,0])             # 0 cats, 5 non-cats

print(f"Information Gain: {information_gain(y_parent, y_left, y_right):.4f}")
```

### Handling Continuous Features

For a continuous feature (e.g., weight), try all midpoints between sorted unique values as candidate thresholds, compute information gain for each, and pick the best.

### Handling Multi-class Classification

Replace binary entropy with **multi-class entropy** (also called Shannon entropy):

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k$$

### Regression Trees

Replace entropy with **variance reduction** as the splitting criterion:

$$IG = \text{Var}(y_{\text{root}}) - \left( \frac{n_l}{n} \text{Var}(y_{\text{left}}) + \frac{n_r}{n} \text{Var}(y_{\text{right}}) \right)$$

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification tree
clf = DecisionTreeClassifier(
    max_depth=5,            # prevents overfitting
    min_samples_split=10,   # minimum examples to split a node
    min_samples_leaf=5,     # minimum examples in a leaf
    criterion='entropy'     # or 'gini'
)
clf.fit(X_train, y_train)
```

---

## 15. Ensemble Methods: Random Forests & XGBoost

A single decision tree is unstable — small changes in the training data can produce very different trees. Ensemble methods combine many trees, averaging out this instability to produce far more robust and accurate models.

### Random Forests: Bagging + Feature Randomization

**Bagging** (Bootstrap AGGregatING): For $B$ trees:

1. Draw a bootstrap sample of size $m$ **with replacement** from the training set.
2. Train a decision tree on this sample.
3. At prediction time, take a majority vote (classification) or average (regression).

Why does this help? Each bootstrap sample is different, so each tree has different errors. When you average many trees with **uncorrelated errors**, the errors cancel out. The variance of the average decreases as $1/B$, while bias remains roughly constant.

**Random forests** add one more trick: at each split, only a random subset of $\sqrt{p}$ features (for classification) is considered. This **further decorrelates the trees**, making the ensemble more powerful.

$$\hat{y} = \text{majority\_vote}\!\left( T_1(\mathbf{x}), T_2(\mathbf{x}), \ldots, T_B(\mathbf{x}) \right)$$

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # number of trees B — more is better, diminishing returns
    max_features='sqrt',     # features per split: sqrt(p) for classification
    max_depth=None,          # grow full trees — bagging controls variance
    bootstrap=True,          # with-replacement sampling
    n_jobs=-1,               # use all CPU cores
    random_state=42
)
rf.fit(X_train, y_train)

# Feature importances — useful for understanding the model
importances = rf.feature_importances_
```

### XGBoost: Gradient Boosting

Where random forests build trees **independently in parallel**, **boosting** builds them **sequentially**, each one focusing on correcting the mistakes of the previous ensemble.

**Algorithm (conceptual):**

1. Initialize prediction $F_0(\mathbf{x}) = \text{constant}$ (e.g., mean of $y$).
2. For $b = 1, \ldots, B$:
   a. Compute residuals / pseudo-residuals: what the current ensemble gets wrong.
   b. Fit a new tree $T_b$ to these residuals.
   c. Update: $F_b(\mathbf{x}) = F_{b-1}(\mathbf{x}) + \eta \cdot T_b(\mathbf{x})$

where $\eta$ is the **learning rate** (shrinkage), which slows down the boosting to prevent overfitting.

**XGBoost** (Extreme Gradient Boosting) is the most popular implementation, adding:
- Second-order (Newton) gradient approximations for faster, more accurate updates
- Built-in L1 and L2 regularization on leaf weights and tree complexity
- Intelligent handling of missing values
- Efficient parallel tree building via approximate split-finding

**The core XGBoost objective:**

$$\mathcal{L}(T) = \sum_{i=1}^{m} \ell(y_i, \hat{y}_i) + \Omega(T)$$

$$\Omega(T) = \gamma K + \frac{1}{2} \lambda \sum_{k=1}^{K} w_k^2$$

where $K$ is the number of leaves, $w_k$ is the weight of leaf $k$, $\gamma$ penalizes tree complexity (acts like pruning), and $\lambda$ is L2 regularization on leaf weights.

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# XGBoost classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=500,         # number of boosting rounds
    learning_rate=0.05,       # shrinkage factor eta — smaller = more robust
    max_depth=6,              # maximum depth per tree
    min_child_weight=1,       # minimum sum of instance weights in a leaf
    subsample=0.8,            # fraction of training examples per tree
    colsample_bytree=0.8,     # fraction of features per tree
    gamma=0.1,                # minimum loss reduction for a split
    reg_lambda=1.0,           # L2 regularization on leaf weights
    reg_alpha=0.0,            # L1 regularization on leaf weights
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Fit with early stopping (stops when CV error doesn't improve)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_cv, y_cv)],
    early_stopping_rounds=50,  # stop if no improvement for 50 rounds
    verbose=100
)

# Feature importance
xgb.plot_importance(xgb_model)
```

### Comparison: Bagging vs. Boosting

| Property | Random Forest (Bagging) | XGBoost (Boosting) |
|---|---|---|
| Tree construction | Independent, parallel | Sequential, adaptive |
| Focus of each tree | Random bootstrap sample | Residuals of previous ensemble |
| Variance reduction | Averaging reduces variance | Shrinkage + regularization |
| Risk of overfitting | Lower | Higher (need early stopping) |
| Training speed | Fast (parallelizable) | Slower but competitive |
| Hyperparameter sensitivity | Low | Higher |
| Typical use | Robust baseline | Competition-winning, best accuracy |

---

## 16. When to Use Neural Networks vs. Decision Trees

### Neural Networks Are Best When:

- Data is **unstructured** (images, audio, text) — neural networks excel at learning representations from raw signals.
- You have **very large datasets** — neural networks scale better with data than tree ensembles.
- You need **transfer learning** — pretrained neural networks provide powerful starting representations.
- The input features don't have a natural "feature importance" ordering.

### Decision Trees / Ensembles Are Best When:

- Data is **tabular / structured** — XGBoost and Random Forests are typically superior to neural networks on structured data and run much faster.
- **Interpretability** matters — a single tree is inherently explainable.
- **Training time** is constrained — tree ensembles train far faster than large neural networks.
- You need **feature importance** — tree-based feature importances are well-understood and reliable.

```python
# Rule of thumb for tabular data
if dataset_type == 'tabular':
    if interpretability_required:
        model = DecisionTreeClassifier(max_depth=5)    # single tree, readable
    elif best_accuracy_required:
        model = xgb.XGBClassifier(...)                  # usually wins on tabular
    else:
        model = RandomForestClassifier(...)             # robust, fast, solid
elif dataset_type in ['images', 'text', 'audio']:
    model = build_neural_network(...)                   # always prefer NNs here
```

---

## Summary of Key Formulas

| Concept | Formula |
|---|---|
| Neuron activation | $a_j^{[l]} = g\!\left(\mathbf{w}_j^{[l]\top}\mathbf{a}^{[l-1]} + b_j^{[l]}\right)$ |
| Binary cross-entropy | $J = -\frac{1}{m}\sum_i \left[y^{(i)}\log\hat{y}^{(i)} + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$ |
| Softmax | $a_j = e^{z_j} / \sum_k e^{z_k}$ |
| L2 regularized cost | $J_{\text{reg}} = J + \frac{\lambda}{2m}\sum_l \|\mathbf{W}^{[l]}\|_F^2$ |
| Entropy | $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ |
| Information gain | $IG = H(p_{\text{root}}) - \frac{n_l}{n}H(p_l) - \frac{n_r}{n}H(p_r)$ |
| Precision | $P = TP / (TP + FP)$ |
| Recall | $R = TP / (TP + FN)$ |
| F1 | $F_1 = 2PR / (P + R)$ |
| Adam update | $w \leftarrow w - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon}\hat{m}$ |

---

> **Key Takeaway:** The most important skill in applied ML is not knowing a lot of algorithms — it is knowing how to **diagnose** what is wrong with your current model and choosing the right fix. Bias/variance analysis, learning curves, and systematic cross-validation are the tools that separate effective ML engineers from ones who waste months on dead ends.
