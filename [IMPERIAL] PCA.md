# Mathematics for Machine Learning: Principal Component Analysis

## Complete Course Notes with Detailed Derivations

---

# Table of Contents

1. [Introduction to Dimensionality Reduction](#1-introduction-to-dimensionality-reduction)
2. [Statistical Foundations](#2-statistical-foundations)
   - 2.1 [Mean of Data Sets](#21-mean-of-data-sets)
   - 2.2 [Variance in One Dimension](#22-variance-in-one-dimension)
   - 2.3 [Covariance and the Covariance Matrix](#23-covariance-and-the-covariance-matrix)
   - 2.4 [Effects of Linear Transformations](#24-effects-of-linear-transformations)
3. [Inner Products and Geometry](#3-inner-products-and-geometry)
   - 3.1 [The Dot Product](#31-the-dot-product)
   - 3.2 [General Inner Products](#32-general-inner-products)
   - 3.3 [Lengths and Norms](#33-lengths-and-norms)
   - 3.4 [Distances Between Vectors](#34-distances-between-vectors)
   - 3.5 [Angles and Orthogonality](#35-angles-and-orthogonality)
   - 3.6 [Inner Products of Functions and Random Variables](#36-inner-products-of-functions-and-random-variables)
4. [Vector Spaces and Bases](#4-vector-spaces-and-bases)
   - 4.1 [Groups](#41-groups)
   - 4.2 [Vector Spaces](#42-vector-spaces)
   - 4.3 [Vector Subspaces](#43-vector-subspaces)
   - 4.4 [Generating Sets and Bases](#44-generating-sets-and-bases)
   - 4.5 [Orthogonal Complements](#45-orthogonal-complements)
5. [Orthogonal Projections](#5-orthogonal-projections)
   - 5.1 [Projection onto 1D Subspaces](#51-projection-onto-1d-subspaces)
   - 5.2 [Projection onto Higher-Dimensional Subspaces](#52-projection-onto-higher-dimensional-subspaces)
   - 5.3 [The Projection Matrix](#53-the-projection-matrix)
   - 5.4 [Worked Examples](#54-worked-examples)
6. [Principal Component Analysis](#6-principal-component-analysis)
   - 6.1 [Problem Setting and Motivation](#61-problem-setting-and-motivation)
   - 6.2 [The PCA Objective Function](#62-the-pca-objective-function)
   - 6.3 [Finding Optimal Coordinates](#63-finding-optimal-coordinates)
   - 6.4 [Reformulating the Loss Function](#64-reformulating-the-loss-function)
   - 6.5 [Finding the Principal Subspace](#65-finding-the-principal-subspace)
   - 6.6 [The Complete PCA Algorithm](#66-the-complete-pca-algorithm)
   - 6.7 [High-Dimensional PCA](#67-high-dimensional-pca)
   - 6.8 [Alternative Interpretations of PCA](#68-alternative-interpretations-of-pca)
7. [Appendix: Multivariate Chain Rule](#7-appendix-multivariate-chain-rule)

---

# 1. Introduction to Dimensionality Reduction

## The Challenge of High-Dimensional Data

Real-world data is often **high-dimensional**, meaning each data point has many features or attributes. Consider estimating a house price: you might use house type, size, number of bedrooms and bathrooms, neighborhood values, distance to transportation, crime rates, economic indicators, and dozens more features. Each feature adds a dimension to your data space.

A **640×480 color image** lives in a space of dimension:

$$
640 \times 480 \times 3 = 921{,}600 \text{ dimensions}
$$

where each pixel contributes three dimensions (red, green, blue channels). Working with such high-dimensional data presents several challenges:

| Challenge | Description |
|-----------|-------------|
| **Analysis difficulty** | Statistical patterns become harder to detect |
| **Visualization** | Cannot plot data in more than 3 dimensions |
| **Computational cost** | Storage and processing scale with dimensionality |
| **Curse of dimensionality** | Data becomes sparse; distances lose meaning |

## Why Dimensionality Reduction Works

High-dimensional data often possesses a crucial property: **redundancy**. Many dimensions are correlated or can be explained by combinations of other dimensions. Consider adding a grayscale channel to an RGB image. The grayscale value at each pixel is completely determined by the RGB values:

$$
\text{Gray} = 0.299R + 0.587G + 0.114B
$$

This fourth channel adds no new information—it's **redundant**. Dimensionality reduction exploits such structure and correlation to find more compact representations without losing essential information.

## The MNIST Example

The **MNIST dataset** contains 28×28 pixel images of handwritten digits. Each image is a vector in:

$$
\mathbb{R}^{784} \quad \text{(since } 28 \times 28 = 784\text{)}
$$

However, the pixels aren't randomly distributed—they're highly structured. Neighboring pixels often have similar values, and all images of "8" share common features. This structure means the **effective dimensionality** is much lower than 784.

## What is PCA?

**Principal Component Analysis (PCA)** is a linear dimensionality reduction algorithm that finds a lower-dimensional subspace capturing the maximum variance in the data. The lower-dimensional representation is called the **code** or **features**.

PCA works like compression algorithms (JPEG for images, MP3 for audio) but uses mathematical optimization rather than domain-specific heuristics. The mathematical foundations we'll develop include:

1. **Statistical representations**: Describing data with means and variances
2. **Geometry in vector spaces**: Measuring distances and angles using inner products
3. **Orthogonal projections**: Finding the closest point in a subspace
4. **Optimization**: Minimizing reconstruction error to find the best subspace

---

# 2. Statistical Foundations

## 2.1 Mean of Data Sets

### Definition and Intuition

The **mean** (or **expected value**) of a dataset describes the "average" or "center" data point. For a dataset $\mathcal{D} = \{x_1, x_2, \ldots, x_N\}$ containing $N$ data points, the mean is:

$$
\mathbb{E}[\mathcal{D}] = \mu = \frac{1}{N} \sum_{n=1}^{N} x_n
$$

The mean represents the **centroid**—the point that minimizes the sum of squared distances to all data points. It balances the "mass" of the data.

### Important Properties of the Mean

1. **Not necessarily in the dataset**: The mean $\mu$ may not be any actual data point
2. **Not necessarily achievable**: For dice rolls, the mean might be 3.5, which no single roll can produce
3. **Sensitive to outliers**: A single extreme value can dramatically shift the mean

### Worked Example: Dice Rolls

Consider rolling five dice and obtaining: $\mathcal{D}' = \{1, 2, 4, 6, 6\}$

$$
\mathbb{E}[\mathcal{D}'] = \frac{1 + 2 + 4 + 6 + 6}{5} = \frac{19}{5} = 3.8
$$

Notice that 3.8 is not achievable with a single die roll, yet it represents the "average" outcome.

### Mean of High-Dimensional Data

For **vector-valued data** where each $x_n \in \mathbb{R}^D$, the mean is computed **component-wise**:

$$
\boldsymbol{\mu} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_D \end{pmatrix}
$$

where $\mu_d = \frac{1}{N} \sum_{n=1}^{N} x_{n,d}$ is the mean of the $d$-th component.

### Visual Example: Average Digit

For images of handwritten "8"s, stacking all images as vectors and computing the mean produces a **blurry composite** that captures common features of all 8s while losing individual variations. This average "8" belongs to the vector space but isn't any specific digit from the dataset.

---

## 2.2 Variance in One Dimension

### Motivation: Spread Around the Mean

Two datasets can have identical means but very different **spreads**. Consider:

$$
\mathcal{D}_1 = \{1, 2, 4, 5\} \quad \text{and} \quad \mathcal{D}_2 = \{-1, 3, 7\}
$$

Both have mean 3, but $\mathcal{D}_2$ is more "spread out" around the mean. The **variance** quantifies this spread.

### Definition of Variance

The variance measures the **average squared distance** from the mean:

$$
\text{Var}[\mathcal{D}] = \sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (x_n - \mu)^2
$$

We square the differences because:
1. It makes all terms positive (distances are non-negative)
2. It penalizes larger deviations more heavily
3. It leads to mathematically convenient properties

### Worked Example: Computing Variance

For $\mathcal{D}_1 = \{1, 2, 4, 5\}$ with $\mu = 3$:

$$
\text{Var}[\mathcal{D}_1] = \frac{(1-3)^2 + (2-3)^2 + (4-3)^2 + (5-3)^2}{4} = \frac{4 + 1 + 1 + 4}{4} = \frac{10}{4} = 2.5
$$

For $\mathcal{D}_2 = \{-1, 3, 7\}$ with $\mu = 3$:

$$
\text{Var}[\mathcal{D}_2] = \frac{(-1-3)^2 + (3-3)^2 + (7-3)^2}{3} = \frac{16 + 0 + 16}{3} = \frac{32}{3} \approx 10.67
$$

Since $\text{Var}[\mathcal{D}_2] > \text{Var}[\mathcal{D}_1]$, we confirm $\mathcal{D}_2$ is more spread out.

### Standard Deviation

The variance has units of **squared** measurement units (if data is in meters, variance is in meters²). The **standard deviation** restores original units:

$$
\sigma = \sqrt{\text{Var}[\mathcal{D}]} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (x_n - \mu)^2}
$$

### Key Properties

1. **Non-negative**: $\text{Var}[\mathcal{D}] \geq 0$ always (sum of squares)
2. **Zero variance**: $\text{Var}[\mathcal{D}] = 0$ if and only if all data points are identical
3. **Interpretable**: Standard deviation is in same units as data

---

## 2.3 Covariance and the Covariance Matrix

### Why Variance Isn't Enough in Higher Dimensions

In 2D or higher, computing variance in each direction separately misses **relationships between dimensions**. Consider four datasets with identical means and per-dimension variances:

| Dataset | Shape | X-variance | Y-variance |
|---------|-------|------------|------------|
| A | Circular | 1 | 1 |
| B | Vertical ellipse | 1 | 1 |
| C | Horizontal ellipse | 1 | 1 |
| D | Diagonal ellipse | 1 | 1 |

These have identical marginal statistics but very different structures! Dataset D shows that when X increases, Y tends to increase too—they're **correlated**. The covariance captures this relationship.

### Definition of Covariance

The **covariance** between two variables $X$ and $Y$ measures their joint variability:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \frac{1}{N} \sum_{n=1}^{N} (x_n - \mu_X)(y_n - \mu_Y)
$$

**Interpretation**:
- **Positive covariance**: When $X$ is above its mean, $Y$ tends to be above its mean (positive correlation)
- **Negative covariance**: When $X$ is above its mean, $Y$ tends to be below its mean (negative correlation)
- **Zero covariance**: $X$ and $Y$ are **uncorrelated** (no linear relationship)

### The Covariance Matrix

For D-dimensional data, we organize all variances and covariances into a **covariance matrix** $\mathbf{S} \in \mathbb{R}^{D \times D}$:

$$
\mathbf{S} = \begin{pmatrix} \text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_D) \\ \text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_D) \\ \vdots & \vdots & \ddots & \vdots \\ \text{Cov}(X_D, X_1) & \text{Cov}(X_D, X_2) & \cdots & \text{Var}(X_D) \end{pmatrix}
$$

### Compact Matrix Form

For centered data (mean zero), if we stack data points as rows of matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$:

$$
\mathbf{X} = \begin{pmatrix} — \mathbf{x}_1^T — \\ — \mathbf{x}_2^T — \\ \vdots \\ — \mathbf{x}_N^T — \end{pmatrix}
$$

Then the covariance matrix is:

$$
\mathbf{S} = \frac{1}{N} \mathbf{X}^T \mathbf{X} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n \mathbf{x}_n^T
$$

### Properties of the Covariance Matrix

The covariance matrix is always:

1. **Symmetric**: $\mathbf{S} = \mathbf{S}^T$ (since $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$)
2. **Positive semi-definite**: $\mathbf{v}^T \mathbf{S} \mathbf{v} \geq 0$ for all vectors $\mathbf{v}$

These properties have profound implications:
- All eigenvalues are **real** (from symmetry)
- All eigenvalues are **non-negative** (from positive semi-definiteness)
- Eigenvectors for distinct eigenvalues are **orthogonal**

### Geometric Interpretation

The **eigenvectors** of $\mathbf{S}$ point in the **directions of maximum variance** in the data. The corresponding **eigenvalues** equal the variance in those directions. This is the foundation of PCA!

---

## 2.4 Effects of Linear Transformations

### Linear Transformations on Data

A **linear transformation** modifies data by:
1. **Scaling**: Multiplying by a constant $\alpha$
2. **Shifting**: Adding a constant $a$

Understanding how mean and variance transform under these operations is crucial for data preprocessing.

### Effect on the Mean

**Shifting**: Adding a constant shifts the mean by the same amount:

$$
\mathbb{E}[\mathcal{D} + a] = \mathbb{E}[\mathcal{D}] + a
$$

**Scaling**: Multiplying by a constant scales the mean:

$$
\mathbb{E}[\alpha \mathcal{D}] = \alpha \cdot \mathbb{E}[\mathcal{D}]
$$

**Combined (Affine Transformation)**:

$$
\mathbb{E}[\alpha \mathcal{D} + a] = \alpha \cdot \mathbb{E}[\mathcal{D}] + a
$$

### Effect on the Variance

**Shifting**: Adding a constant does **not** change variance:

$$
\text{Var}[\mathcal{D} + a] = \text{Var}[\mathcal{D}]
$$

This makes intuitive sense: shifting all points equally doesn't change the spread.

**Scaling**: Multiplying by $\alpha$ scales the variance by $\alpha^2$:

$$
\text{Var}[\alpha \mathcal{D}] = \alpha^2 \cdot \text{Var}[\mathcal{D}]
$$

The squared factor appears because variance involves squared differences.

### Derivation of Scaling Effect

Let $\mathcal{D}' = \alpha \mathcal{D}$. The new mean is $\mu' = \alpha \mu$. Then:

$$
\text{Var}[\mathcal{D}'] = \frac{1}{N} \sum_{n=1}^{N} (\alpha x_n - \alpha \mu)^2 = \frac{1}{N} \sum_{n=1}^{N} \alpha^2 (x_n - \mu)^2 = \alpha^2 \cdot \text{Var}[\mathcal{D}]
$$

### High-Dimensional Case

For vector-valued data $\mathbf{x} \in \mathbb{R}^D$, consider the transformation:

$$
\tilde{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{b}
$$

where $\mathbf{A} \in \mathbb{R}^{D' \times D}$ and $\mathbf{b} \in \mathbb{R}^{D'}$.

**Mean transformation**:

$$
\tilde{\boldsymbol{\mu}} = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}
$$

**Covariance transformation**:

$$
\tilde{\mathbf{S}} = \mathbf{A} \mathbf{S} \mathbf{A}^T
$$

Note that the shift $\mathbf{b}$ doesn't affect the covariance, while the linear transformation $\mathbf{A}$ transforms the covariance in a specific way involving both left and right multiplication.

---

# 3. Inner Products and Geometry

## 3.1 The Dot Product

### Definition

The **dot product** (or **scalar product**) of two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ is:

$$
\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i
$$

The result is a **scalar** (single number), not a vector.

### Worked Example

Let $\mathbf{x} = (1, 2)^T$ and $\mathbf{y} = (2, 1)^T$. Then:

$$
\mathbf{x}^T \mathbf{y} = 1 \cdot 2 + 2 \cdot 1 = 4
$$

### What the Dot Product Measures

The dot product has a beautiful geometric interpretation:

$$
\mathbf{x}^T \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos \theta
$$

where $\theta$ is the angle between the vectors. This means:
- **Positive dot product**: Acute angle ($\theta < 90°$), vectors point "roughly the same direction"
- **Zero dot product**: Right angle ($\theta = 90°$), vectors are **perpendicular/orthogonal**
- **Negative dot product**: Obtuse angle ($\theta > 90°$), vectors point "roughly opposite directions"

### Enabling Geometry

The dot product enables us to define:
1. **Length** of a vector
2. **Distance** between vectors
3. **Angle** between vectors
4. **Orthogonality** (perpendicularity)

---

## 3.2 General Inner Products

### Why Generalize?

The dot product assumes standard Euclidean geometry. Sometimes we need **non-standard** geometry where:
- Certain directions are "more important"
- Distances are measured differently
- The natural coordinates aren't orthogonal

**Inner products** generalize the dot product while preserving its key geometric properties.

### Formal Definition

An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ satisfying:

**1. Bilinearity**: Linear in both arguments

$$
\langle \lambda \mathbf{x} + \mathbf{z}, \mathbf{y} \rangle = \lambda \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{z}, \mathbf{y} \rangle
$$

$$
\langle \mathbf{x}, \lambda \mathbf{y} + \mathbf{z} \rangle = \lambda \langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{x}, \mathbf{z} \rangle
$$

**2. Symmetry**:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle
$$

**3. Positive Definiteness**:

$$
\langle \mathbf{x}, \mathbf{x} \rangle \geq 0 \quad \text{with equality iff } \mathbf{x} = \mathbf{0}
$$

### Inner Products via Positive Definite Matrices

Any **symmetric positive definite (SPD)** matrix $\mathbf{A}$ defines an inner product:

$$
\langle \mathbf{x}, \mathbf{y} \rangle_\mathbf{A} = \mathbf{x}^T \mathbf{A} \mathbf{y}
$$

When $\mathbf{A} = \mathbf{I}$ (identity matrix), this reduces to the standard dot product.

### Example: Non-Standard Inner Product

Let:

$$
\mathbf{A} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}
$$

This is SPD (check: symmetric, positive eigenvalues $\lambda = 1, 3$).

The inner product induced by $\mathbf{A}$:

$$
\langle \mathbf{x}, \mathbf{y} \rangle_\mathbf{A} = \mathbf{x}^T \mathbf{A} \mathbf{y} = 2x_1 y_1 - x_1 y_2 - x_2 y_1 + 2x_2 y_2
$$

This "weights" the $x_1$ and $x_2$ components differently than the standard dot product.

---

## 3.3 Lengths and Norms

### Definition via Inner Product

The **length** (or **norm**) of a vector is defined using the inner product:

$$
\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}
$$

For the standard dot product, this gives the familiar **Euclidean norm**:

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

### Worked Example

Let $\mathbf{x} = (1, 2)^T$.

**Standard Euclidean norm**:

$$
\|\mathbf{x}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{5}
$$

**With a different inner product**: Let:

$$
\mathbf{A} = \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix}
$$

Then:

$$
\|\mathbf{x}\|_\mathbf{A}^2 = \mathbf{x}^T \mathbf{A} \mathbf{x} = (1, 2) \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \end{pmatrix} = (1, 2) \begin{pmatrix} 0 \\ \frac{3}{2} \end{pmatrix} = 3
$$

So $\|\mathbf{x}\|_\mathbf{A} = \sqrt{3}$, different from $\sqrt{5}$!

### Properties of Norms

**1. Absolute homogeneity**:

$$
\|\lambda \mathbf{x}\| = |\lambda| \|\mathbf{x}\|
$$

**2. Triangle inequality**:

$$
\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|
$$

**3. Positive definiteness**:

$$
\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}
$$

### Cauchy-Schwarz Inequality

A fundamental result connecting norms and inner products:

$$
|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \|\mathbf{y}\|
$$

Equality holds if and only if $\mathbf{x}$ and $\mathbf{y}$ are **linearly dependent** (one is a scalar multiple of the other).

---

## 3.4 Distances Between Vectors

### Definition

The **distance** between two vectors is the length of their difference:

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|
$$

### Euclidean Distance

With the standard inner product:

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T (\mathbf{x} - \mathbf{y})} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

This is the familiar "straight-line" distance.

### Worked Example

Let $\mathbf{x} = (1, 2)^T$ and $\mathbf{y} = (2, 1)^T$. Then:

$$
\mathbf{x} - \mathbf{y} = (-1, 1)^T
$$

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{(-1)^2 + 1^2} = \sqrt{2}
$$

### Non-Euclidean Distance

With the inner product from:

$$
\mathbf{A} = \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix}
$$

We get:

$$
d_\mathbf{A}(\mathbf{x}, \mathbf{y})^2 = (\mathbf{x} - \mathbf{y})^T \mathbf{A} (\mathbf{x} - \mathbf{y}) = (-1, 1) \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix} \begin{pmatrix} -1 \\ 1 \end{pmatrix}
$$

$$
= (-1, 1) \begin{pmatrix} -\frac{3}{2} \\ \frac{3}{2} \end{pmatrix} = \frac{3}{2} + \frac{3}{2} = 3
$$

So $d_\mathbf{A}(\mathbf{x}, \mathbf{y}) = \sqrt{3}$, different from $\sqrt{2}$!

---

## 3.5 Angles and Orthogonality

### Computing Angles

The angle $\theta$ between vectors $\mathbf{x}$ and $\mathbf{y}$ satisfies:

$$
\cos \theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

This formula is well-defined because Cauchy-Schwarz guarantees the fraction is between -1 and 1.

### Worked Example

Let $\mathbf{x} = (1, 2)^T$ and $\mathbf{y} = (2, 1)^T$ with standard inner product:

$$
\cos \theta = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{4}{\sqrt{5} \cdot \sqrt{5}} = \frac{4}{5}
$$

$$
\theta = \arccos(0.8) \approx 0.64 \text{ radians} \approx 36.9°
$$

### Orthogonality

Two vectors are **orthogonal** (perpendicular) if their inner product is zero:

$$
\mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0
$$

### Example: Orthogonal Vectors

Let $\mathbf{x} = (1, 1)^T$ and $\mathbf{y} = (-1, 1)^T$ with dot product:

$$
\mathbf{x}^T \mathbf{y} = -1 + 1 = 0
$$

So $\mathbf{x} \perp \mathbf{y}$ (the angle is exactly 90°).

### Orthogonality Depends on Inner Product!

The same vectors might be orthogonal with one inner product but not another.

With:

$$
\mathbf{A} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}
$$

We get:

$$
\langle \mathbf{x}, \mathbf{y} \rangle_\mathbf{A} = \mathbf{x}^T \mathbf{A} \mathbf{y} = (1, 1) \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} -1 \\ 1 \end{pmatrix} = (1, 1) \begin{pmatrix} -2 \\ 1 \end{pmatrix} = -1
$$

So $\mathbf{x}$ and $\mathbf{y}$ are **not** orthogonal with respect to this inner product!

### Orthonormal Bases

A basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_D\}$ is **orthonormal** if:

$$
\langle \mathbf{b}_i, \mathbf{b}_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
$$

Orthonormal bases are extremely useful because:
- Coordinates are easy to compute: $x_i = \langle \mathbf{x}, \mathbf{b}_i \rangle$
- The Pythagorean theorem applies
- Projections have simple formulas

---

## 3.6 Inner Products of Functions and Random Variables

### Inner Product of Functions

The inner product concept extends to **continuous functions**. For functions $u, v : [a, b] \to \mathbb{R}$:

$$
\langle u, v \rangle = \int_a^b u(x) v(x) \, dx
$$

The sum over vector components becomes an integral over the domain.

### Orthogonal Functions

Two functions are orthogonal if their inner product is zero:

$$
\langle u, v \rangle = \int_a^b u(x) v(x) \, dx = 0
$$

**Example**: $\sin x$ and $\cos x$ on $[-\pi, \pi]$:

$$
\int_{-\pi}^{\pi} \sin x \cos x \, dx = \frac{1}{2} \int_{-\pi}^{\pi} \sin(2x) \, dx = 0
$$

The Fourier basis $\{1, \cos x, \sin x, \cos 2x, \sin 2x, \ldots\}$ consists of mutually orthogonal functions—this is the foundation of **Fourier analysis**.

### Inner Product of Random Variables

For random variables $X$ and $Y$, define:

$$
\langle X, Y \rangle = \text{Cov}(X, Y)
$$

This satisfies:
- **Symmetry**: $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- **Bilinearity**: $\text{Cov}(\lambda X + Z, Y) = \lambda \text{Cov}(X, Y) + \text{Cov}(Z, Y)$
- **Positive definiteness**: $\text{Cov}(X, X) = \text{Var}(X) \geq 0$

### Geometric Interpretation

The "length" of a random variable is its **standard deviation**:

$$
\|X\| = \sqrt{\text{Var}(X)} = \sigma_X
$$

Two random variables are **orthogonal** (uncorrelated) if:

$$
\text{Cov}(X, Y) = 0
$$

### The Pythagorean Theorem for Random Variables

If $X$ and $Y$ are **uncorrelated**, then:

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)
$$

Compare to the Pythagorean theorem: $c^2 = a^2 + b^2$ for orthogonal vectors!

This geometric perspective on probability is powerful—variance addition for independent variables is just the Pythagorean theorem in the space of random variables.

---

# 4. Vector Spaces and Bases

## 4.1 Groups

### Definition

A **group** $(G, \otimes)$ consists of a set $G$ and an operation $\otimes : G \times G \to G$ satisfying:

**1. Closure**: $\forall x, y \in G : x \otimes y \in G$

**2. Associativity**: $\forall x, y, z \in G : (x \otimes y) \otimes z = x \otimes (y \otimes z)$

**3. Identity element**: $\exists e \in G : e \otimes x = x \otimes e = x \quad \forall x \in G$

**4. Inverse element**: $\forall x \in G, \exists y \in G : x \otimes y = y \otimes x = e$

If additionally $x \otimes y = y \otimes x$ for all elements, the group is **Abelian** (commutative).

### Examples and Non-Examples

| Set & Operation | Group? | Reason |
|-----------------|--------|--------|
| $(\mathbb{Z}, +)$ | ✓ Abelian | Identity 0, inverse $-n$ |
| $(\mathbb{N}_0, +)$ | ✗ | No inverses (can't subtract) |
| $(\mathbb{Z}, \cdot)$ | ✗ | No inverse for $z \neq \pm 1$ |
| $(\mathbb{R} \setminus \{0\}, \cdot)$ | ✓ Abelian | Identity 1, inverse $1/x$ |
| $(\mathbb{R}^n, +)$ | ✓ Abelian | Component-wise addition |
| $(\mathbb{R}^{m \times n}, +)$ | ✓ Abelian | Matrix addition |

---

## 4.2 Vector Spaces

### Definition

A **real vector space** is a set $V$ with two operations:

**Inner operation (vector addition)**: $+ : V \times V \to V$

**Outer operation (scalar multiplication)**: $\cdot : \mathbb{R} \times V \to V$

satisfying:

1. $(V, +)$ is an Abelian group
2. **Distributivity**: $\lambda(\mathbf{x} + \mathbf{y}) = \lambda\mathbf{x} + \lambda\mathbf{y}$ and $(\lambda + \psi)\mathbf{x} = \lambda\mathbf{x} + \psi\mathbf{x}$
3. **Associativity of scalar multiplication**: $\lambda(\psi\mathbf{x}) = (\lambda\psi)\mathbf{x}$
4. **Identity for scalars**: $1 \cdot \mathbf{x} = \mathbf{x}$

Elements of $V$ are **vectors**; elements of $\mathbb{R}$ are **scalars**.

### Key Examples

1. $\mathbb{R}^n$ with standard addition and scalar multiplication
2. $\mathbb{R}^{m \times n}$ (matrices) with matrix addition and scalar multiplication
3. Continuous functions $C[a, b]$ with function addition and scalar multiplication
4. Polynomials of degree at most $n$

---

## 4.3 Vector Subspaces

### Definition

A **subspace** $U$ of vector space $V$ is a subset $U \subseteq V$ that is itself a vector space with the inherited operations.

### Subspace Test

To verify $U$ is a subspace, check:

1. **Non-empty**: $\mathbf{0} \in U$ (contains the zero vector)
2. **Closed under scalar multiplication**: $\forall \lambda \in \mathbb{R}, \mathbf{x} \in U : \lambda \mathbf{x} \in U$
3. **Closed under addition**: $\forall \mathbf{x}, \mathbf{y} \in U : \mathbf{x} + \mathbf{y} \in U$

### Examples in $\mathbb{R}^2$

| Set | Subspace? | Reason |
|-----|-----------|--------|
| Line through origin | ✓ | Passes all tests |
| Line not through origin | ✗ | Doesn't contain $\mathbf{0}$ |
| First quadrant $\{(x,y) : x, y \geq 0\}$ | ✗ | Not closed under scalar mult. |
| All of $\mathbb{R}^2$ | ✓ | Trivially satisfies |
| $\{\mathbf{0}\}$ | ✓ | Trivial subspace |

### Geometric Intuition

In $\mathbb{R}^3$, subspaces are:
- The origin alone (0-dimensional)
- Lines through the origin (1-dimensional)
- Planes through the origin (2-dimensional)
- All of $\mathbb{R}^3$ (3-dimensional)

Note: A plane that doesn't pass through the origin is **not** a subspace!

---

## 4.4 Generating Sets and Bases

### Generating Set (Span)

A set of vectors $\mathcal{A} = \{\mathbf{x}_1, \ldots, \mathbf{x}_k\}$ **generates** (or **spans**) a vector space $V$ if every vector in $V$ can be written as a linear combination:

$$
V = \text{span}(\mathcal{A}) = \lbrace \sum_{i=1}^{k} \lambda_i \mathbf{x}_i : \lambda_i \in \mathbb{R} \rbrace
$$

### Linear Independence

Vectors $\{\mathbf{x}_1, \ldots, \mathbf{x}_k\}$ are **linearly independent** if:

$$
\sum_{i=1}^{k} \lambda_i \mathbf{x}_i = \mathbf{0} \implies \lambda_1 = \lambda_2 = \cdots = \lambda_k = 0
$$

Otherwise, they are **linearly dependent** (some vector can be written as a combination of others).

### Basis

A **basis** of vector space $V$ is a set of vectors that is:
1. **Linearly independent**
2. **Generating** (spans $V$)

Equivalently, a basis is a **minimal generating set** or a **maximal linearly independent set**.

### Key Properties

1. Every vector space has a basis
2. **All bases have the same size** (number of elements)
3. This size is called the **dimension**: $\dim(V)$
4. Every vector has a **unique** representation as a linear combination of basis vectors

### Examples in $\mathbb{R}^3$

**Standard/Canonical basis**:

$$
\mathbf{e}_1 = (1, 0, 0)^T, \quad \mathbf{e}_2 = (0, 1, 0)^T, \quad \mathbf{e}_3 = (0, 0, 1)^T
$$

**Another valid basis**:

$$
\mathbf{b}_1 = (1, 0, 0)^T, \quad \mathbf{b}_2 = (1, 1, 0)^T, \quad \mathbf{b}_3 = (1, 1, 1)^T
$$

**Not a basis of $\mathbb{R}^4$** (linearly independent but doesn't span $\mathbb{R}^4$):

The set $\{(1, 2, 3, 4)^T, (2, -1, 0, 2)^T, (1, 1, 0, -4)^T\}$ spans only a 3D subspace of $\mathbb{R}^4$.

---

## 4.5 Orthogonal Complements

### Definition

Given a subspace $W \subseteq V$, the **orthogonal complement** $W^\perp$ consists of all vectors orthogonal to every vector in $W$:

$$
W^\perp = \{ \mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W \}
$$

### Key Properties

1. $W^\perp$ is also a subspace
2. $\dim(W) + \dim(W^\perp) = \dim(V)$
3. $(W^\perp)^\perp = W$
4. $W \cap W^\perp = \{\mathbf{0}\}$

### Orthogonal Decomposition Theorem

Every vector $\mathbf{y} \in V$ can be **uniquely** decomposed as:

$$
\mathbf{y} = \hat{\mathbf{y}} + \mathbf{z}
$$

where $\hat{\mathbf{y}} \in W$ and $\mathbf{z} \in W^\perp$.

If $\{\mathbf{u}_1, \ldots, \mathbf{u}_p\}$ is an orthogonal basis for $W$:

$$
\hat{\mathbf{y}} = \frac{\langle \mathbf{y}, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} \mathbf{u}_1 + \cdots + \frac{\langle \mathbf{y}, \mathbf{u}_p \rangle}{\langle \mathbf{u}_p, \mathbf{u}_p \rangle} \mathbf{u}_p
$$

and $\mathbf{z} = \mathbf{y} - \hat{\mathbf{y}}$.

### Example in $\mathbb{R}^3$

Let $W$ be the $xy$-plane: $W = \{(x, y, 0) : x, y \in \mathbb{R}\}$

Then $W^\perp$ is the $z$-axis: $W^\perp = \{(0, 0, z) : z \in \mathbb{R}\}$

Any vector $(a, b, c)$ decomposes as $(a, b, 0) + (0, 0, c)$.

---

# 5. Orthogonal Projections

## 5.1 Projection onto 1D Subspaces

### The Problem

Given a vector $\mathbf{x} \in \mathbb{R}^D$ and a 1-dimensional subspace $U$ spanned by vector $\mathbf{b}$, find the **closest point** in $U$ to $\mathbf{x}$.

### Two Key Insights

**Insight 1**: Since the projection $\pi_U(\mathbf{x}) \in U$, it must be a scalar multiple of $\mathbf{b}$:

$$
\pi_U(\mathbf{x}) = \lambda \mathbf{b}
$$

for some $\lambda \in \mathbb{R}$.

**Insight 2**: The closest point is where the "error vector" is **orthogonal** to $U$:

$$
\langle \mathbf{b}, \mathbf{x} - \pi_U(\mathbf{x}) \rangle = 0
$$

### Derivation

Starting from the orthogonality condition:

$$
\langle \mathbf{b}, \mathbf{x} - \lambda \mathbf{b} \rangle = 0
$$

By linearity:

$$
\langle \mathbf{b}, \mathbf{x} \rangle - \lambda \langle \mathbf{b}, \mathbf{b} \rangle = 0
$$

Solving for $\lambda$:

$$
\lambda = \frac{\langle \mathbf{b}, \mathbf{x} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} = \frac{\langle \mathbf{b}, \mathbf{x} \rangle}{\|\mathbf{b}\|^2}
$$

The **projection** is:

$$
\pi_U(\mathbf{x}) = \lambda \mathbf{b} = \frac{\langle \mathbf{b}, \mathbf{x} \rangle}{\|\mathbf{b}\|^2} \mathbf{b}
$$

### Using the Dot Product

With standard inner product:

$$
\pi_U(\mathbf{x}) = \frac{\mathbf{b}^T \mathbf{x}}{\mathbf{b}^T \mathbf{b}} \mathbf{b} = \frac{\mathbf{b}^T \mathbf{x}}{\|\mathbf{b}\|^2} \mathbf{b}
$$

Rearranging:

$$
\pi_U(\mathbf{x}) = \frac{\mathbf{b} \mathbf{b}^T}{\mathbf{b}^T \mathbf{b}} \mathbf{x}
$$

### Special Case: Unit Vector

If $\|\mathbf{b}\| = 1$:

$$
\lambda = \mathbf{b}^T \mathbf{x}
$$

$$
\pi_U(\mathbf{x}) = (\mathbf{b}^T \mathbf{x}) \mathbf{b} = \mathbf{b} \mathbf{b}^T \mathbf{x}
$$

---

## 5.2 Projection onto Higher-Dimensional Subspaces

### Setup

Given $\mathbf{x} \in \mathbb{R}^n$ and an $m$-dimensional subspace $U$ with ordered basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_m\}$, find $\pi_U(\mathbf{x})$.

### The Same Two Insights

**Insight 1**: The projection is a linear combination of basis vectors:

$$
\pi_U(\mathbf{x}) = \mathbf{p} = \sum_{i=1}^{m} \lambda_i \mathbf{b}_i = \mathbf{B} \boldsymbol{\lambda}
$$

where $\mathbf{B} = [\mathbf{b}_1 | \cdots | \mathbf{b}_m] \in \mathbb{R}^{n \times m}$ and $\boldsymbol{\lambda} = [\lambda_1, \ldots, \lambda_m]^T$.

**Insight 2**: The error is orthogonal to all basis vectors:

$$
\langle \mathbf{b}_i, \mathbf{x} - \mathbf{p} \rangle = 0 \quad \text{for } i = 1, \ldots, m
$$

### Derivation

The orthogonality conditions in matrix form:

$$
\mathbf{B}^T (\mathbf{x} - \mathbf{B} \boldsymbol{\lambda}) = \mathbf{0}
$$

Expanding:

$$
\mathbf{B}^T \mathbf{x} = \mathbf{B}^T \mathbf{B} \boldsymbol{\lambda}
$$

This is called the **normal equation**. Solving for $\boldsymbol{\lambda}$:

$$
\boldsymbol{\lambda} = (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{x}
$$

The matrix $(\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T$ is called the **pseudo-inverse** of $\mathbf{B}$.

### The Projection

$$
\mathbf{p} = \pi_U(\mathbf{x}) = \mathbf{B} \boldsymbol{\lambda} = \mathbf{B} (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T \mathbf{x}
$$

---

## 5.3 The Projection Matrix

### Definition

The **projection matrix** $\mathbf{P}_\pi$ satisfies $\pi_U(\mathbf{x}) = \mathbf{P}_\pi \mathbf{x}$ for any $\mathbf{x}$:

$$
\mathbf{P}_\pi = \mathbf{B} (\mathbf{B}^T \mathbf{B})^{-1} \mathbf{B}^T
$$

### Special Case: Orthonormal Basis

If the columns of $\mathbf{B}$ are orthonormal, then $\mathbf{B}^T \mathbf{B} = \mathbf{I}$, and:

$$
\mathbf{P}_\pi = \mathbf{B} \mathbf{B}^T
$$

This is much simpler and numerically stable—a major reason to use orthonormal bases.

### Properties of Projection Matrices

**1. Idempotent**: $\mathbf{P}_\pi^2 = \mathbf{P}_\pi$

Projecting twice is the same as projecting once (already in the subspace).

**2. Symmetric**: $\mathbf{P}_\pi = \mathbf{P}_\pi^T$

**3. Rank**: $\text{rank}(\mathbf{P}_\pi) = m$ (dimension of subspace)

**4. Eigenvalues**: Only 0 and 1

- Eigenvectors with eigenvalue 1 span $U$
- Eigenvectors with eigenvalue 0 span $U^\perp$

### Connection to 1D Case

For 1D with basis vector $\mathbf{b}$:

$$
\mathbf{P}_\pi = \frac{\mathbf{b} \mathbf{b}^T}{\mathbf{b}^T \mathbf{b}}
$$

Setting $\mathbf{B} = \mathbf{b}$ in the general formula:

$$
\mathbf{P}_\pi = \mathbf{b} (\mathbf{b}^T \mathbf{b})^{-1} \mathbf{b}^T = \frac{\mathbf{b} \mathbf{b}^T}{\mathbf{b}^T \mathbf{b}}
$$

(since $\mathbf{b}^T \mathbf{b}$ is a scalar).

---

## 5.4 Worked Examples

### Example 1: Projection onto a Line in $\mathbb{R}^2$

Project $\mathbf{x} = (1, 2)^T$ onto the line spanned by $\mathbf{b} = (2, 1)^T$.

**Step 1**: Compute $\lambda$

$$
\lambda = \frac{\mathbf{b}^T \mathbf{x}}{\mathbf{b}^T \mathbf{b}} = \frac{2 \cdot 1 + 1 \cdot 2}{2^2 + 1^2} = \frac{4}{5}
$$

**Step 2**: Compute projection

$$
\pi_U(\mathbf{x}) = \lambda \mathbf{b} = \frac{4}{5} (2, 1)^T = \left(\frac{8}{5}, \frac{4}{5}\right)^T
$$

**Verification**: Check orthogonality

$$
\mathbf{x} - \pi_U(\mathbf{x}) = \left(1 - \frac{8}{5}, 2 - \frac{4}{5}\right)^T = \left(-\frac{3}{5}, \frac{6}{5}\right)^T
$$

$$
\mathbf{b}^T (\mathbf{x} - \pi_U(\mathbf{x})) = 2 \cdot \left(-\frac{3}{5}\right) + 1 \cdot \frac{6}{5} = -\frac{6}{5} + \frac{6}{5} = 0 \checkmark
$$

### Example 2: Projection onto a Plane in $\mathbb{R}^3$

Project $\mathbf{x} = (2, 1, 1)^T$ onto the plane $U$ spanned by $\mathbf{b}_1 = (1, 2, 0)^T$ and $\mathbf{b}_2 = (1, 1, 0)^T$.

**Step 1**: Form matrix $\mathbf{B}$

$$
\mathbf{B} = \begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 0 & 0 \end{pmatrix}
$$

**Step 2**: Compute $\mathbf{B}^T \mathbf{x}$

$$
\mathbf{B}^T \mathbf{x} = \begin{pmatrix} 1 & 2 & 0 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}
$$

**Step 3**: Compute $\mathbf{B}^T \mathbf{B}$

$$
\mathbf{B}^T \mathbf{B} = \begin{pmatrix} 1 & 2 & 0 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 5 & 3 \\ 3 & 2 \end{pmatrix}
$$

**Step 4**: Solve $\mathbf{B}^T \mathbf{B} \boldsymbol{\lambda} = \mathbf{B}^T \mathbf{x}$

$$
\begin{pmatrix} 5 & 3 \\ 3 & 2 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}
$$

Using the inverse (det = $10 - 9 = 1$):

$$
(\mathbf{B}^T \mathbf{B})^{-1} = \begin{pmatrix} 2 & -3 \\ -3 & 5 \end{pmatrix}
$$

$$
\boldsymbol{\lambda} = \begin{pmatrix} 2 & -3 \\ -3 & 5 \end{pmatrix} \begin{pmatrix} 4 \\ 3 \end{pmatrix} = \begin{pmatrix} -1 \\ 3 \end{pmatrix}
$$

**Step 5**: Compute projection

$$
\pi_U(\mathbf{x}) = \mathbf{B} \boldsymbol{\lambda} = \begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} -1 \\ 3 \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \\ 0 \end{pmatrix}
$$

**Interpretation**: The projection has zero in the third component because the subspace $U$ is the $xy$-plane (both basis vectors have $z = 0$).

---

# 6. Principal Component Analysis

## 6.1 Problem Setting and Motivation

### The Goal

Given a dataset $\mathcal{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ with $\mathbf{x}_n \in \mathbb{R}^D$, find a **lower-dimensional representation** using only $M < D$ dimensions while **preserving as much information as possible**.

### Key Assumptions

1. **Centered data**: $\mathbb{E}[\mathcal{X}] = \mathbf{0}$ (subtract mean as preprocessing)
2. **Orthonormal basis**: We seek basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_D\}$ with $\mathbf{b}_i^T \mathbf{b}_j = \delta_{ij}$

### Representation of Data Points

Any $\mathbf{x}_n$ can be written as:

$$
\mathbf{x}_n = \sum_{i=1}^{D} \beta_{i,n} \mathbf{b}_i
$$

where $\beta_{i,n} = \mathbf{x}_n^T \mathbf{b}_i$ are the **coordinates** in the new basis.

### The PCA Approximation

We approximate each data point using only the first $M$ basis vectors:

$$
\tilde{\mathbf{x}}_n = \sum_{i=1}^{M} \beta_{i,n} \mathbf{b}_i
$$

The **code** (low-dimensional representation) is:

$$
\mathbf{z}_n = [\beta_{1,n}, \ldots, \beta_{M,n}]^T \in \mathbb{R}^M
$$

### What PCA Finds

PCA finds the **optimal** orthonormal basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_M\}$ that minimizes the **reconstruction error** between $\mathbf{x}_n$ and $\tilde{\mathbf{x}}_n$.

---

## 6.2 The PCA Objective Function

### Average Squared Reconstruction Error

The loss function measures how well we can reconstruct the original data:

$$
J = \frac{1}{N} \sum_{n=1}^{N} \|\mathbf{x}_n - \tilde{\mathbf{x}}_n\|^2
$$

This is the **average squared distance** between data points and their projections.

### Optimization Variables

We need to find:
1. **Coordinates** $\beta_{i,n}$ for each data point $n$ and basis vector $i$
2. **Basis vectors** $\mathbf{b}_1, \ldots, \mathbf{b}_M$ (and implicitly $\mathbf{b}_{M+1}, \ldots, \mathbf{b}_D$)

### Using the Chain Rule

The parameters enter $J$ only through $\tilde{\mathbf{x}}_n$. By the chain rule:

$$
\frac{\partial J}{\partial \beta_{i,n}} = \frac{\partial J}{\partial \tilde{\mathbf{x}}_n} \frac{\partial \tilde{\mathbf{x}}_n}{\partial \beta_{i,n}}
$$

$$
\frac{\partial J}{\partial \mathbf{b}_i} = \frac{\partial J}{\partial \tilde{\mathbf{x}}_n} \frac{\partial \tilde{\mathbf{x}}_n}{\partial \mathbf{b}_i}
$$

The first factor is:

$$
\frac{\partial J}{\partial \tilde{\mathbf{x}}_n} = -\frac{2}{N} (\mathbf{x}_n - \tilde{\mathbf{x}}_n)^T
$$

---

## 6.3 Finding Optimal Coordinates

### Derivative with Respect to $\beta_{i,n}$

Since $\tilde{\mathbf{x}}_n = \sum_{j=1}^{M} \beta_{j,n} \mathbf{b}_j$:

$$
\frac{\partial \tilde{\mathbf{x}}_n}{\partial \beta_{i,n}} = \mathbf{b}_i
$$

Therefore:

$$
\frac{\partial J}{\partial \beta_{i,n}} = -\frac{2}{N} (\mathbf{x}_n - \tilde{\mathbf{x}}_n)^T \mathbf{b}_i
$$

### Setting the Derivative to Zero

$$
(\mathbf{x}_n - \tilde{\mathbf{x}}_n)^T \mathbf{b}_i = 0
$$

Substituting $\tilde{\mathbf{x}}_n = \sum_{j=1}^{M} \beta_{j,n} \mathbf{b}_j$:

$$
\mathbf{x}_n^T \mathbf{b}_i - \sum_{j=1}^{M} \beta_{j,n} \mathbf{b}_j^T \mathbf{b}_i = 0
$$

Since the basis is orthonormal ($\mathbf{b}_j^T \mathbf{b}_i = \delta_{ji}$):

$$
\mathbf{x}_n^T \mathbf{b}_i - \beta_{i,n} = 0
$$

### The Optimal Coordinates

$$
\boxed{\beta_{i,n} = \mathbf{x}_n^T \mathbf{b}_i}
$$

**Interpretation**: The optimal coordinate is the **orthogonal projection** of $\mathbf{x}_n$ onto the direction $\mathbf{b}_i$. This confirms that the best low-dimensional representation is obtained by **projecting** the data onto the principal subspace.

---

## 6.4 Reformulating the Loss Function

### Expressing the Reconstruction

With optimal coordinates:

$$
\tilde{\mathbf{x}}_n = \sum_{j=1}^{M} (\mathbf{x}_n^T \mathbf{b}_j) \mathbf{b}_j = \sum_{j=1}^{M} \mathbf{b}_j \mathbf{b}_j^T \mathbf{x}_n = \left( \sum_{j=1}^{M} \mathbf{b}_j \mathbf{b}_j^T \right) \mathbf{x}_n
$$

This is the **projection** of $\mathbf{x}_n$ onto the principal subspace spanned by $\{\mathbf{b}_1, \ldots, \mathbf{b}_M\}$.

### Orthogonal Decomposition

Using the full orthonormal basis, any $\mathbf{x}_n$ can be written as:

$$
\mathbf{x}_n = \underbrace{\sum_{j=1}^{M} (\mathbf{x}_n^T \mathbf{b}_j) \mathbf{b}_j}_{\text{principal subspace}} + \underbrace{\sum_{j=M+1}^{D} (\mathbf{x}_n^T \mathbf{b}_j) \mathbf{b}_j}_{\text{orthogonal complement}}
$$

### The Displacement Vector

The difference is:

$$
\mathbf{x}_n - \tilde{\mathbf{x}}_n = \sum_{j=M+1}^{D} (\mathbf{x}_n^T \mathbf{b}_j) \mathbf{b}_j
$$

This lies entirely in the **orthogonal complement** of the principal subspace.

### Rewriting the Loss

$$
J = \frac{1}{N} \sum_{n=1}^{N} \left\| \sum_{j=M+1}^{D} (\mathbf{x}_n^T \mathbf{b}_j) \mathbf{b}_j \right\|^2
$$

Since $\{\mathbf{b}_j\}$ is orthonormal:

$$
J = \frac{1}{N} \sum_{n=1}^{N} \sum_{j=M+1}^{D} (\mathbf{x}_n^T \mathbf{b}_j)^2
$$

### Connecting to the Covariance Matrix

Rearranging sums:

$$
J = \sum_{j=M+1}^{D} \frac{1}{N} \sum_{n=1}^{N} (\mathbf{x}_n^T \mathbf{b}_j)^2 = \sum_{j=M+1}^{D} \mathbf{b}_j^T \left( \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n \mathbf{x}_n^T \right) \mathbf{b}_j
$$

The term in parentheses is the **data covariance matrix** $\mathbf{S}$ (for centered data):

$$
\boxed{J = \sum_{j=M+1}^{D} \mathbf{b}_j^T \mathbf{S} \mathbf{b}_j}
$$

### The Key Insight

The loss equals the **variance of the data projected onto the ignored subspace**. Minimizing this is equivalent to **maximizing variance in the principal subspace**.

---

## 6.5 Finding the Principal Subspace

### Setting Up the Optimization

We want to minimize:

$$
J = \sum_{j=M+1}^{D} \mathbf{b}_j^T \mathbf{S} \mathbf{b}_j
$$

subject to orthonormality: $\mathbf{b}_i^T \mathbf{b}_j = \delta_{ij}$

### The 1D Case First

For $D = 2, M = 1$, we minimize:

$$
J = \mathbf{b}_2^T \mathbf{S} \mathbf{b}_2 \quad \text{subject to} \quad \|\mathbf{b}_2\| = 1
$$

Using **Lagrange multipliers**:

$$
\mathcal{L} = \mathbf{b}_2^T \mathbf{S} \mathbf{b}_2 + \lambda (1 - \mathbf{b}_2^T \mathbf{b}_2)
$$

### Finding Critical Points

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_2} = 2 \mathbf{S} \mathbf{b}_2 - 2 \lambda \mathbf{b}_2 = \mathbf{0}
$$

This gives:

$$
\boxed{\mathbf{S} \mathbf{b}_2 = \lambda \mathbf{b}_2}
$$

This is an **eigenvalue equation**! The optimal $\mathbf{b}_2$ is an **eigenvector** of $\mathbf{S}$.

### Which Eigenvector?

Substituting back into the loss:

$$
J = \mathbf{b}_2^T \mathbf{S} \mathbf{b}_2 = \mathbf{b}_2^T (\lambda \mathbf{b}_2) = \lambda \|\mathbf{b}_2\|^2 = \lambda
$$

The loss equals the eigenvalue! To **minimize** $J$, choose $\mathbf{b}_2$ as the eigenvector with the **smallest** eigenvalue.

### The Principal Direction

Since $\mathbf{b}_1 \perp \mathbf{b}_2$ and eigenvectors of symmetric matrices are orthogonal:

$$
\mathbf{b}_1 = \text{eigenvector of } \mathbf{S} \text{ with largest eigenvalue}
$$

### General Case

For an $M$-dimensional principal subspace:

$$
J = \sum_{j=M+1}^{D} \lambda_j
$$

where $\lambda_j$ are eigenvalues with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$.

To minimize $J$: choose $\mathbf{b}_{M+1}, \ldots, \mathbf{b}_D$ as eigenvectors of the **smallest** $D - M$ eigenvalues.

Equivalently, the **principal subspace** is spanned by eigenvectors of the **largest** $M$ eigenvalues.

### Summary: The PCA Solution

$$
\boxed{\text{Principal components} = \text{eigenvectors of } \mathbf{S} \text{ with largest eigenvalues}}
$$

The eigenvalues equal the variance in each principal direction.

---

## 6.6 The Complete PCA Algorithm

### Preprocessing

**Step 1: Center the data**

$$
\bar{\mathbf{x}} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n \quad \Rightarrow \quad \mathbf{x}_n \leftarrow \mathbf{x}_n - \bar{\mathbf{x}}
$$

**Step 2: (Optional but recommended) Standardize**

$$
\sigma_d = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x_{n,d}^2} \quad \Rightarrow \quad x_{n,d} \leftarrow \frac{x_{n,d}}{\sigma_d}
$$

This makes features comparable when they have different units.

### Core Algorithm

**Step 3: Compute covariance matrix**

$$
\mathbf{S} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n \mathbf{x}_n^T = \frac{1}{N} \mathbf{X}^T \mathbf{X}
$$

**Step 4: Eigendecomposition**

Compute eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$ and eigenvectors $\mathbf{b}_1, \ldots, \mathbf{b}_D$.

**Step 5: Select principal components**

Form $\mathbf{B} = [\mathbf{b}_1 | \cdots | \mathbf{b}_M]$ from the top $M$ eigenvectors.

### Projection

**To encode** (project to low dimension):

$$
\mathbf{z} = \mathbf{B}^T \mathbf{x} \in \mathbb{R}^M
$$

**To decode** (reconstruct):

$$
\tilde{\mathbf{x}} = \mathbf{B} \mathbf{z} = \mathbf{B} \mathbf{B}^T \mathbf{x} \in \mathbb{R}^D
$$

### Projecting New Data Points

For a new point $\mathbf{x}^*$:

1. Normalize: $\mathbf{x}^* \leftarrow \frac{\mathbf{x}^* - \bar{\mathbf{x}}}{\boldsymbol{\sigma}}$ (using training mean and std)
2. Project: $\mathbf{z}^* = \mathbf{B}^T \mathbf{x}^*$
3. Reconstruct: $\tilde{\mathbf{x}}^* = \mathbf{B} \mathbf{z}^*$

### Choosing $M$: Explained Variance

The **variance explained** by the first $M$ components:

$$
\text{Explained variance ratio} = \frac{\sum_{i=1}^{M} \lambda_i}{\sum_{i=1}^{D} \lambda_i}
$$

Common choices:
- Keep 95% or 99% of variance
- Use elbow method on scree plot
- Choose $M$ based on downstream task performance

---

## 6.7 High-Dimensional PCA

### The Computational Challenge

For $D$-dimensional data, the covariance matrix $\mathbf{S} \in \mathbb{R}^{D \times D}$.

Eigendecomposition costs $O(D^3)$ operations.

For images with $D = 10^6$ pixels, this is infeasible!

### The Key Insight

When $N \ll D$ (fewer samples than dimensions):

$$
\text{rank}(\mathbf{S}) = \text{rank}\left(\frac{1}{N} \mathbf{X}^T \mathbf{X}\right) \leq \min(N, D) = N
$$

So $\mathbf{S}$ has at most $N$ non-zero eigenvalues. The other $D - N$ eigenvalues are exactly zero.

### The Trick

Instead of the $D \times D$ matrix $\mathbf{X}^T \mathbf{X}$, work with the $N \times N$ matrix:

$$
\mathbf{M} = \frac{1}{N} \mathbf{X} \mathbf{X}^T \in \mathbb{R}^{N \times N}
$$

### Relationship Between Eigenvectors

If $\mathbf{c}$ is an eigenvector of $\mathbf{M}$ with eigenvalue $\lambda$:

$$
\mathbf{M} \mathbf{c} = \lambda \mathbf{c}
$$

$$
\frac{1}{N} \mathbf{X} \mathbf{X}^T \mathbf{c} = \lambda \mathbf{c}
$$

Multiply both sides by $\mathbf{X}^T$:

$$
\frac{1}{N} \mathbf{X}^T \mathbf{X} (\mathbf{X}^T \mathbf{c}) = \lambda (\mathbf{X}^T \mathbf{c})
$$

$$
\mathbf{S} (\mathbf{X}^T \mathbf{c}) = \lambda (\mathbf{X}^T \mathbf{c})
$$

So $\mathbf{X}^T \mathbf{c}$ is an eigenvector of $\mathbf{S}$ with the **same eigenvalue** $\lambda$.

### The Algorithm

1. Compute $\mathbf{M} = \frac{1}{N} \mathbf{X} \mathbf{X}^T$ ($N \times N$ matrix)
2. Find eigenvectors $\mathbf{c}_1, \ldots, \mathbf{c}_N$ of $\mathbf{M}$
3. Recover eigenvectors of $\mathbf{S}$: $\mathbf{b}_i = \mathbf{X}^T \mathbf{c}_i$ (then normalize)

Complexity: $O(N^3 + N^2 D)$ instead of $O(D^3)$ — much faster when $N \ll D$.

---

## 6.8 Alternative Interpretations of PCA

### 1. Maximum Variance Perspective

PCA finds directions that **maximize variance** in the projected data.

First principal component:

$$
\mathbf{b}_1 = \arg\max_{\|\mathbf{b}\|=1} \mathbf{b}^T \mathbf{S} \mathbf{b} = \arg\max_{\|\mathbf{b}\|=1} \text{Var}[\mathbf{b}^T \mathbf{x}]
$$

Subsequent components maximize variance orthogonal to previous components.

### 2. Minimum Reconstruction Error

As we derived, PCA minimizes:

$$
J = \frac{1}{N} \sum_{n=1}^{N} \|\mathbf{x}_n - \tilde{\mathbf{x}}_n\|^2
$$

### 3. Linear Autoencoder

An **autoencoder** learns:
- **Encoder**: $\mathbf{z} = f(\mathbf{x})$ (compress)
- **Decoder**: $\tilde{\mathbf{x}} = g(\mathbf{z})$ (reconstruct)

With **linear** encoder/decoder:

$$
\mathbf{z} = \mathbf{W}^T \mathbf{x}, \quad \tilde{\mathbf{x}} = \mathbf{W} \mathbf{z}
$$

Minimizing reconstruction error yields PCA solution: $\mathbf{W} = \mathbf{B}$ (top eigenvectors).

**Deep autoencoders** replace linear mappings with neural networks for nonlinear dimensionality reduction.

### 4. Information Theory: Maximum Mutual Information

The **mutual information** $I(\mathbf{x}; \mathbf{z})$ measures how much knowing $\mathbf{z}$ tells us about $\mathbf{x}$.

Under Gaussian assumptions, maximizing $I(\mathbf{x}; \mathbf{z})$ gives PCA.

### 5. Probabilistic PCA

Model data as generated from low-dimensional latent variable:

$$
\mathbf{x} = \mathbf{B} \mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}
$$

where:
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (latent variable)
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ (noise)

The **marginal likelihood**:

$$
p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \mathbf{B}\mathbf{B}^T + \sigma^2 \mathbf{I})
$$

**Maximum likelihood estimation** recovers:
- $\boldsymbol{\mu} = $ sample mean
- Columns of $\mathbf{B} = $ top eigenvectors (scaled by $\sqrt{\lambda_i - \sigma^2}$)

Advantages of probabilistic PCA:
- Handles missing data
- Provides uncertainty estimates
- Enables Bayesian extensions

---

# 7. Appendix: Multivariate Chain Rule

## Differentiation Rules for Vectors

For $\mathbf{x} \in \mathbb{R}^n$, the standard rules apply with matrix dimensions:

**Product Rule**:

$$
\frac{\partial}{\partial \mathbf{x}} [f(\mathbf{x}) g(\mathbf{x})] = \frac{\partial f}{\partial \mathbf{x}} g(\mathbf{x}) + f(\mathbf{x}) \frac{\partial g}{\partial \mathbf{x}}
$$

**Sum Rule**:

$$
\frac{\partial}{\partial \mathbf{x}} [f(\mathbf{x}) + g(\mathbf{x})] = \frac{\partial f}{\partial \mathbf{x}} + \frac{\partial g}{\partial \mathbf{x}}
$$

**Chain Rule**:

$$
\frac{\partial}{\partial \mathbf{x}} [g(f(\mathbf{x}))] = \frac{\partial g}{\partial f} \frac{\partial f}{\partial \mathbf{x}}
$$

## Matrix-Vector Calculus

For linear function $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$:

$$
\frac{\partial f}{\partial \mathbf{x}} = \mathbf{A}
$$

For quadratic form $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ (symmetric $\mathbf{A}$):

$$
\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{x}^T \mathbf{A} \quad \text{(row vector)}
$$

## Example: Gradient of Linear Model Loss

Consider:

$$
L(\boldsymbol{\theta}) = \|\mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\theta}\|^2 = (\mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\theta})^T(\mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\theta})
$$

Let $\mathbf{e} = \mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\theta}$. Then $L = \mathbf{e}^T \mathbf{e}$.

**Step 1**: $\frac{\partial L}{\partial \mathbf{e}} = 2\mathbf{e}^T$

**Step 2**: $\frac{\partial \mathbf{e}}{\partial \boldsymbol{\theta}} = -\boldsymbol{\Phi}$

**Step 3**: Chain rule:

$$
\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial \mathbf{e}} \frac{\partial \mathbf{e}}{\partial \boldsymbol{\theta}} = 2\mathbf{e}^T (-\boldsymbol{\Phi}) = -2(\mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\theta})^T \boldsymbol{\Phi}
$$

Setting to zero and solving gives the normal equation:

$$
\boldsymbol{\Phi}^T \boldsymbol{\Phi} \boldsymbol{\theta} = \boldsymbol{\Phi}^T \mathbf{y}
$$

---

# Summary: Key Formulas

## Statistics

| Quantity | Formula |
|----------|---------|
| Mean | $\boldsymbol{\mu} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n$ |
| Variance (1D) | $\sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (x_n - \mu)^2$ |
| Covariance matrix | $\mathbf{S} = \frac{1}{N} \sum_{n=1}^{N} (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T$ |
| Linear transform of mean | $\mathbb{E}[\mathbf{A}\mathbf{x} + \mathbf{b}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}$ |
| Linear transform of covariance | $\text{Cov}[\mathbf{A}\mathbf{x}] = \mathbf{A}\mathbf{S}\mathbf{A}^T$ |

## Inner Products and Geometry

| Quantity | Formula |
|----------|---------|
| Length | $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ |
| Distance | $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$ |
| Angle | $\cos \theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}$ |
| Orthogonality | $\mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0$ |

## Projections

| Quantity | Formula |
|----------|---------|
| Projection onto line (span of $\mathbf{b}$) | $\pi_U(\mathbf{x}) = \frac{\mathbf{b}^T \mathbf{x}}{\mathbf{b}^T \mathbf{b}} \mathbf{b}$ |
| Projection matrix (general) | $\mathbf{P}_\pi = \mathbf{B}(\mathbf{B}^T\mathbf{B})^{-1}\mathbf{B}^T$ |
| Projection matrix (orthonormal basis) | $\mathbf{P}_\pi = \mathbf{B}\mathbf{B}^T$ |

## PCA

| Quantity | Formula |
|----------|---------|
| Covariance matrix | $\mathbf{S} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$ (centered data) |
| Principal components | Eigenvectors of $\mathbf{S}$ with largest eigenvalues |
| Variance explained | $\frac{\sum_{i=1}^{M} \lambda_i}{\sum_{i=1}^{D} \lambda_i}$ |
| Encoding | $\mathbf{z} = \mathbf{B}^T \mathbf{x}$ |
| Decoding | $\tilde{\mathbf{x}} = \mathbf{B}\mathbf{z}$ |
| Reconstruction error | $J = \sum_{j=M+1}^{D} \lambda_j$ |

---

*These notes cover the mathematical foundations of Principal Component Analysis as presented in the Imperial College London course "Mathematics for Machine Learning: PCA".*
