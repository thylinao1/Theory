# Mathematics for Machine Learning: Linear Algebra

## Course Overview

Linear algebra provides the mathematical foundation for machine learning by offering a set of notational conventions and operations that allow us to manipulate large systems of equations conveniently. This course focuses on building intuition about vectors, matrices, and transformations rather than just mechanical calculations.

---

## Chapter 1: Introduction and Motivation

### 1.1 Why Linear Algebra for Machine Learning?

Linear algebra is essential for machine learning because it provides tools to represent, interpret, and control complex systems. While open-source libraries allow applying ML methods without deep mathematical understanding, problems inevitably arise—and without knowledge of the underlying mathematics, debugging becomes nearly impossible.

**Key Applications:**
- Solving simultaneous equations (price discovery, parameter estimation)
- Fitting equations to data (optimization problems)
- Neural networks and data transformations

### 1.2 The Apples and Bananas Problem

A motivating example: discovering individual prices from total bills.

**Problem Setup:**
- Trip 1: 2 apples + 3 bananas = 8 euros
- Trip 2: 10 apples + 1 banana = 13 euros

**Mathematical Formulation:**

$$2a + 3b = 8$$

$$10a + b = 13$$

This can be written as a matrix equation:

$$\begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} 8 \\ 13 \end{pmatrix}$$

**Why it's useful:** This formulation allows us to solve systems with many variables and equations using computer algorithms, which would be impractical by hand.

### 1.3 The Optimization Problem

Fitting a function (like a Gaussian distribution) to data involves finding optimal parameter values.

**Gaussian Distribution:**

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Where:
- $\mu$ = center/mean of distribution
- $\sigma$ = width/standard deviation

**Optimization Approach:**
- Parameters form a vector in "parameter space"
- We search for the minimum of a "goodness of fit" function
- Moves in parameter space are vectors
- Finding the steepest descent requires calculus on vectors

**Why it's useful:** Understanding vectors and calculus allows us to optimize fitting parameters efficiently, which is fundamental to training machine learning models.

---

## Chapter 2: Vectors

### 2.1 What is a Vector?

A vector is a mathematical object that can represent:
1. **Geometric interpretation:** Something that moves us about space (direction + magnitude)
2. **Data science interpretation:** A list of attributes describing an object

**Example - House as a Vector:**

$$\vec{h} = \begin{pmatrix} 120 \text{ m}^2 \\ 2 \text{ bedrooms} \\ 1 \text{ bathroom} \\ 150{,}000 \text{ euros} \end{pmatrix}$$

**Why it's useful:** Vectors generalize the concept of "moving in space" to include any collection of related quantities, enabling us to apply geometric intuition to abstract data problems.

### 2.2 Vector Operations

Vectors are defined by two fundamental operations:

#### Vector Addition

Adding vectors means placing them end-to-end.

**Component-wise addition:**

$$\begin{pmatrix} r_1 \\ r_2 \end{pmatrix} + \begin{pmatrix} s_1 \\ s_2 \end{pmatrix} = \begin{pmatrix} r_1 + s_1 \\ r_2 + s_2 \end{pmatrix}$$

**Properties:**
- Commutative: $\vec{r} + \vec{s} = \vec{s} + \vec{r}$
- Associative: $(\vec{r} + \vec{s}) + \vec{t} = \vec{r} + (\vec{s} + \vec{t})$

#### Scalar Multiplication

Scaling a vector by a number changes its length:

$$a\vec{r} = \begin{pmatrix} ar_1 \\ ar_2 \end{pmatrix}$$

**Special cases:**
- $-\vec{r}$ = vector pointing in opposite direction
- $\vec{r} + (-\vec{r}) = \vec{0}$

**Why it's useful:** These operations allow us to combine and scale data, forming the basis for all linear algebra computations.

### 2.3 Coordinate Systems and Basis Vectors

A coordinate system is defined by basis vectors (typically denoted $\hat{e}_1, \hat{e}_2$ or $\hat{i}, \hat{j}$).

**Standard Basis in 2D:**

$$\hat{i} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \hat{j} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

**Vector in terms of basis:**

$$\vec{r} = r_1\hat{e}_1 + r_2\hat{e}_2$$

**Why it's useful:** The coordinate system is arbitrary—the same vector can be described with different numbers depending on the chosen basis. This flexibility is crucial for simplifying problems.

---

## Chapter 3: The Dot Product (Inner/Scalar/Projection Product)

### 3.1 Definition of the Dot Product

The dot product multiplies corresponding components and sums them:

$$\vec{r} \cdot \vec{s} = r_1s_1 + r_2s_2 + \ldots + r_ns_n = \sum_{i=1}^{n} r_is_i$$

**Example:**

$$\begin{pmatrix} 3 \\ 2 \end{pmatrix} \cdot \begin{pmatrix} -1 \\ 2 \end{pmatrix} = 3(-1) + 2(2) = -3 + 4 = 1$$

**Why it's useful:** The dot product provides a single number that captures the relationship between two vectors—how much they "agree" in direction.

### 3.2 Properties of the Dot Product

1. **Commutative:** $\vec{r} \cdot \vec{s} = \vec{s} \cdot \vec{r}$

2. **Distributive over addition:** $\vec{r} \cdot (\vec{s} + \vec{t}) = \vec{r} \cdot \vec{s} + \vec{r} \cdot \vec{t}$

3. **Associative with scalar multiplication:** $\vec{r} \cdot (a\vec{s}) = a(\vec{r} \cdot \vec{s})$

**Why it's useful:** These properties allow algebraic manipulation of dot products, essential for derivations and proofs.

### 3.3 Vector Length (Magnitude/Modulus)

The length of a vector is found using the dot product with itself:

$$|\vec{r}| = \sqrt{\vec{r} \cdot \vec{r}} = \sqrt{r_1^2 + r_2^2 + \ldots + r_n^2}$$

This follows from the Pythagorean theorem.

**Why it's useful:** Computing distances and normalizing vectors (making them unit length) are fundamental operations in ML.

### 3.4 Geometric Interpretation: Angle Between Vectors

From the cosine rule, we derive:

$$\vec{r} \cdot \vec{s} = |\vec{r}||\vec{s}|\cos\theta$$

**Implications:**
- If $\theta = 0°$ (same direction): $\vec{r} \cdot \vec{s} = |\vec{r}||\vec{s}|$ (maximum positive)
- If $\theta = 90°$ (perpendicular/orthogonal): $\vec{r} \cdot \vec{s} = 0$
- If $\theta = 180°$ (opposite directions): $\vec{r} \cdot \vec{s} = -|\vec{r}||\vec{s}|$ (maximum negative)

**Why it's useful:** The dot product immediately tells us whether vectors point in similar directions (positive), are perpendicular (zero), or oppose each other (negative).

### 3.5 Projection

#### Scalar Projection

The "shadow" of $\vec{s}$ onto $\vec{r}$:

$$\text{Scalar projection of } \vec{s} \text{ onto } \vec{r} = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|} = |\vec{s}|\cos\theta$$

#### Vector Projection

The projection as a vector in the direction of $\vec{r}$:

$$\text{proj}_{\vec{r}}\vec{s} = \frac{\vec{r} \cdot \vec{s}}{\vec{r} \cdot \vec{r}}\vec{r} = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|^2}\vec{r}$$

**Why it's useful:** Projection decomposes a vector into components along different directions—essential for changing coordinate systems and dimensionality reduction.

---

## Chapter 4: Changing Basis

### 4.1 Basis Vectors and Linear Independence

**Definition:** A basis is a set of $n$ vectors that:
1. Are linearly independent (none can be written as a combination of others)
2. Span the space (any vector in the space can be expressed as their combination)

**Linear Independence Test:**

Vectors $\{b_1, b_2, \ldots, b_n\}$ are linearly independent if there exist no scalars $a_i$ (not all zero) such that:

$$a_1b_1 + a_2b_2 + \ldots + a_nb_n = 0$$

**Why it's useful:** The number of linearly independent basis vectors determines the dimensionality of the space. Choosing good basis vectors simplifies problems dramatically.

### 4.2 Changing Basis Using Projection

When the new basis vectors are **orthogonal** to each other, we can use the dot product to change coordinates.

**Given:** Vector $\vec{r}$ in basis $\{e_1, e_2\}$, new basis $\{b_1, b_2\}$ where $b_1 \perp b_2$

**To find $\vec{r}$ in the new basis:**

$$r_{b_1} = \frac{\vec{r} \cdot \vec{b}_1}{|\vec{b}_1|^2}, \quad r_{b_2} = \frac{\vec{r} \cdot \vec{b}_2}{|\vec{b}_2|^2}$$

**Orthogonality Check:**

$$\vec{b}_1 \cdot \vec{b}_2 = 0 \implies \text{orthogonal}$$

**Why it's useful:** Transforming data to a better coordinate system can reveal structure (e.g., principal components) or simplify computations.

### 4.3 Dimensionality and Data

In data science, we often want to map high-dimensional data to lower dimensions, find directions that capture the most "information", and discard "noise" dimensions.

**Example:** Points lying roughly on a line in 2D can be described by distance along the line (signal) and distance from the line (noise).

**Why it's useful:** Dimensionality reduction is fundamental to handling high-dimensional data in machine learning.

---

## Chapter 5: Introduction to Matrices

### 5.1 What is a Matrix?

A matrix is a rectangular array of numbers that represents a linear transformation—an operation that transforms vectors while preserving the grid structure of space.

**Matrix Notation:**

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

**Matrix-Vector Multiplication:**

$$A\vec{r} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} ax + by \\ cx + dy \end{pmatrix}$$

**Why it's useful:** Matrices encode transformations compactly and allow us to apply the same operation to many vectors efficiently.

### 5.2 Matrices as Transformations of Basis Vectors

The columns of a transformation matrix tell us where the basis vectors go:

$$A = \begin{pmatrix} | & | \\ \vec{e}_1' & \vec{e}_2' \\ | & | \end{pmatrix}$$

**Example:**

$$A = \begin{pmatrix} 2 & 3 \\ 10 & 1 \end{pmatrix}$$

- $\hat{e}_1 = (1, 0)$ transforms to $(2, 10)$
- $\hat{e}_2 = (0, 1)$ transforms to $(3, 1)$

**Why it's useful:** Understanding matrices as basis transformations provides geometric intuition for abstract operations.

### 5.3 Properties of Linear Transformations

Linear transformations preserve:
- Grid lines remain parallel and evenly spaced
- The origin stays fixed
- Linear combinations: $A(n\vec{r}) = nA\vec{r}$ and $A(\vec{r}+\vec{s}) = A\vec{r} + A\vec{s}$

**Why it's useful:** These properties ensure that vector operations remain valid after transformation.

---

## Chapter 6: Types of Matrices and Transformations

### 6.1 Identity Matrix

The matrix that does nothing:

$$I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

For any vector: $I\vec{v} = \vec{v}$

### 6.2 Scaling Matrices

Diagonal matrices scale each axis independently:

$$\begin{pmatrix} a & 0 \\ 0 & d \end{pmatrix}$$

This scales the x-axis by factor $a$ and the y-axis by factor $d$. If $a$ or $d$ is a fraction, it compresses that axis.

### 6.3 Reflection/Mirror Matrices

**Reflection about y-axis:**

$$\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Reflection about x-axis:**

$$\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Inversion (both axes):**

$$\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Mirror at 45°:**

$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### 6.4 Shear Matrices

Shear keeps one axis fixed while tilting the other.

**Horizontal shear:**

$$\begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$

**Vertical shear:**

$$\begin{pmatrix} 1 & 0 \\ k & 1 \end{pmatrix}$$

### 6.5 Rotation Matrices

**Rotation by angle $\theta$ (counterclockwise):**

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Why these are useful:** Any shape change can be decomposed into combinations of these fundamental transformations.

---

## Chapter 7: Matrix Composition and Multiplication

### 7.1 Composing Transformations

Applying transformation $A_1$ followed by $A_2$ is equivalent to multiplying by $A_2A_1$:

$$A_2(A_1\vec{r}) = (A_2A_1)\vec{r}$$

**Important:** Order matters! The rightmost matrix is applied first.

### 7.2 Matrix Multiplication

**Rule:** Multiply rows of the left matrix by columns of the right matrix.

$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}\begin{pmatrix} e & f \\ g & h \end{pmatrix} = \begin{pmatrix} ae+bg & af+bh \\ ce+dg & cf+dh \end{pmatrix}$$

**Properties:**
- **Associative:** $(AB)C = A(BC)$
- **NOT Commutative:** $AB \neq BA$ in general
- **Distributive:** $A(B + C) = AB + AC$

**Why it's useful:** Complex transformations can be pre-computed as a single matrix, making repeated applications efficient.

### 7.3 Einstein Summation Convention

A compact notation for matrix operations:

$$(AB)_{ik} = \sum_j A_{ij}B_{jk}$$

The repeated index $j$ implies summation.

**Why it's useful:** Simplifies coding matrix operations—just loop over indices.

### 7.4 Non-Square Matrices

Matrices don't have to be square. An $(m \times n)$ matrix times an $(n \times p)$ matrix yields an $(m \times p)$ matrix:

$$(m \times n) \times (n \times p) = (m \times p)$$

**Requirement:** The inner dimensions must match.

---

## Chapter 8: Solving Systems of Linear Equations

### 8.1 Matrix Formulation

A system of equations:

$$\begin{aligned}
a_1x + b_1y + c_1z &= d_1 \\
a_2x + b_2y + c_2z &= d_2 \\
a_3x + b_3y + c_3z &= d_3
\end{aligned}$$

Can be written as:

$$A\vec{x} = \vec{d}$$

### 8.2 The Inverse Matrix

If $A^{-1}$ exists such that $A^{-1}A = AA^{-1} = I$, then:

$$\vec{x} = A^{-1}\vec{d}$$

**Why it's useful:** The inverse lets us "undo" a transformation and solve for unknown vectors.

### 8.3 Gaussian Elimination (Row Echelon Form)

**Method:**
1. **Elimination:** Subtract multiples of rows to create zeros below the diagonal
2. **Back-substitution:** Solve from bottom up

**Example:**

$$\begin{pmatrix} 1 & 1 & 3 \\ 1 & 2 & 4 \\ 1 & 1 & 2 \end{pmatrix}\begin{pmatrix} a \\ b \\ c \end{pmatrix} = \begin{pmatrix} 15 \\ 21 \\ 13 \end{pmatrix}$$

After elimination (triangular form):

$$\begin{pmatrix} 1 & 1 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}\begin{pmatrix} a \\ b \\ c \end{pmatrix} = \begin{pmatrix} 15 \\ 6 \\ 2 \end{pmatrix}$$

Back-substitution gives: $c = 2$, then $b = 4$, then $a = 5$.

**Why it's useful:** This is computationally efficient and works for any size system.

### 8.4 Finding the Inverse Matrix

To find $A^{-1}$, augment $A$ with $I$ and reduce $A$ to $I$:

$$[A | I] \rightarrow [I | A^{-1}]$$

**2×2 Inverse Formula:**

$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \implies A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**Why it's useful:** The inverse provides a general solution valid for any right-hand side vector.

---

## Chapter 9: Determinants

### 9.1 Geometric Meaning

The determinant measures how much a transformation **scales area** (2D) or **volume** (3D).

**For 2×2:**

$$\det(A) = \det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**Properties:**
- $|\det(A)| > 1$: Space expanded
- $|\det(A)| < 1$: Space compressed
- $\det(A) < 0$: Orientation flipped (handedness changed)
- $\det(A) = 0$: Space collapsed to lower dimension

### 9.2 Determinant and Invertibility

$$\det(A) = 0 \iff A \text{ has no inverse}$$

**Why:** If the determinant is zero, the transformation collapses space (loses information), which cannot be undone.

### 9.3 Linear Dependence

If $\det(A) = 0$, the columns of $A$ are **linearly dependent**—one can be written as a combination of the others.

**Why it's useful:** The determinant provides a quick check for whether a system of equations has a unique solution.

---

## Chapter 10: Changing Basis with Matrices

### 10.1 Basis Transformation Matrices

To transform a vector from one basis to another:

**Bear's basis vectors in my coordinates:**

$$B = \begin{pmatrix} | & | \\ \vec{b}_1 & \vec{b}_2 \\ | & | \end{pmatrix}$$

**Transformation:**
- $B \cdot \vec{v}_{\text{bear}} = \vec{v}_{\text{mine}}$ (Bear's vector to my coordinates)
- $B^{-1} \cdot \vec{v}_{\text{mine}} = \vec{v}_{\text{bear}}$ (My vector to Bear's coordinates)

### 10.2 Performing Transformations in Different Bases

To apply transformation $T$ (defined in standard basis) to vectors in basis $B$:

$$T_B = B^{-1}TB$$

**Process:**
1. Convert to standard basis: $B\vec{v}$
2. Apply transformation: $T(B\vec{v})$
3. Convert back: $B^{-1}T(B\vec{v})$

**Why it's useful:** This allows us to work in whichever coordinate system makes the problem easiest.

---

## Chapter 11: Orthogonal Matrices and the Gram-Schmidt Process

### 11.1 Transpose of a Matrix

The transpose swaps rows and columns:

$$(A^T)_{ij} = A_{ji}$$

**Example:**

$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$$

### 11.2 Orthonormal Basis

A set of vectors is **orthonormal** if:
- All vectors are mutually orthogonal: $\vec{e}_i \cdot \vec{e}_j = 0$ for $i \neq j$
- All vectors have unit length: $|\vec{e}_i| = 1$

### 11.3 Orthogonal Matrices

A matrix $A$ is **orthogonal** if its columns form an orthonormal basis.

**Key Property:**

$$A^TA = AA^T = I$$

Therefore:

$$A^{-1} = A^T$$

**Properties:**
- $|\det(A)| = 1$ (preserves area/volume)
- Represents pure rotation and/or reflection

**Why it's useful:** Inverting orthogonal matrices is trivial (just transpose), and projections become simple dot products.

### 11.4 The Gram-Schmidt Process

**Purpose:** Convert any set of linearly independent vectors into an orthonormal basis.

**Algorithm:**

Given vectors $\{v_1, v_2, \ldots, v_n\}$:

**Step 1 - First basis vector:**

$$e_1 = \frac{v_1}{|v_1|}$$

**Step 2 - Second basis vector:**

$$u_2 = v_2 - (v_2 \cdot e_1)e_1, \quad e_2 = \frac{u_2}{|u_2|}$$

**Step 3 - Third basis vector:**

$$u_3 = v_3 - (v_3 \cdot e_1)e_1 - (v_3 \cdot e_2)e_2, \quad e_3 = \frac{u_3}{|u_3|}$$

Continue for remaining vectors...

**Why it's useful:** Many algorithms require orthonormal bases. Gram-Schmidt systematically creates one from any starting vectors.

### 11.5 Example: Reflection in an Arbitrary Plane

**Problem:** Reflect a point through a plane defined by vectors $v_1$ and $v_2$.

**Solution:**
1. Use Gram-Schmidt to find orthonormal basis: $e_1, e_2$ (in plane), $e_3$ (normal to plane)
2. Build transformation matrix $E = [e_1 | e_2 | e_3]$
3. Define reflection in this basis:

$$T_E = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$

4. Transform back: $T = E \cdot T_E \cdot E^T$

---

## Chapter 12: Eigenvalues and Eigenvectors

### 12.1 The Eigen-Problem

**Eigenvectors** are vectors that remain on their original span after a transformation (they may stretch or flip, but don't change direction).

**Eigenvalues** are the scaling factors applied to eigenvectors.

**Definition:**

$$A\vec{x} = \lambda\vec{x}$$

Where:
- $A$ = transformation matrix
- $\vec{x}$ = eigenvector
- $\lambda$ = eigenvalue

**Why it's useful:** Eigenvectors reveal the "natural" directions of a transformation—directions where the transformation acts simply as scaling.

### 12.2 Geometric Interpretation

**Visualizing eigenvectors:**
- Draw a unit square
- Apply the transformation
- Vectors that stay on their original line (span) are eigenvectors
- How much they stretch = eigenvalue

**Examples:**

| Transformation | Eigenvectors | Eigenvalues |
|----------------|--------------|-------------|
| Vertical scaling by 2 | Horizontal, Vertical | 1, 2 |
| Horizontal shear | Horizontal only | 1 |
| Rotation (not 180°) | None (real) | Complex |
| 180° rotation | All vectors | -1 |
| 3D rotation | Axis of rotation | 1 |

### 12.3 Calculating Eigenvalues

**Method:** Solve the characteristic equation.

Rearranging $A\vec{x} = \lambda\vec{x}$:

$$(A - \lambda I)\vec{x} = 0$$

For non-trivial solutions, the matrix must be singular:

$$\det(A - \lambda I) = 0$$

**For 2×2 matrices:**

$$\det\begin{pmatrix} a-\lambda & b \\ c & d-\lambda \end{pmatrix} = 0$$

This gives the **characteristic polynomial**:

$$\lambda^2 - (a+d)\lambda + (ad-bc) = 0$$

Its roots are the eigenvalues.

### 12.4 Calculating Eigenvectors

Once eigenvalues are known, substitute back:

$$(A - \lambda I)\vec{x} = 0$$

Solve for $\vec{x}$.

**Example:** For $A = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$:

- $\lambda_1 = 1$: eigenvector $\vec{x} = t\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ (any horizontal vector)
- $\lambda_2 = 2$: eigenvector $\vec{x} = t\begin{pmatrix} 0 \\ 1 \end{pmatrix}$ (any vertical vector)

### 12.5 Special Cases

**No real eigenvectors:** Rotation matrices (except 180°) have only complex eigenvalues.

**All vectors are eigenvectors:** Uniform scaling and 180° rotation.

**Why it's useful:** Eigenanalysis simplifies understanding complex transformations by finding their fundamental directions.

---

## Chapter 13: Diagonalization

### 13.1 The Power of Diagonal Matrices

For diagonal matrices, powers are trivial:

$$D = \begin{pmatrix} a & 0 & 0 \\ 0 & b & 0 \\ 0 & 0 & c \end{pmatrix} \implies D^n = \begin{pmatrix} a^n & 0 & 0 \\ 0 & b^n & 0 \\ 0 & 0 & c^n \end{pmatrix}$$

### 13.2 Diagonalization Process

If matrix $T$ has $n$ linearly independent eigenvectors, it can be diagonalized:

$$T = CDC^{-1}$$

Where:
- $C$ = matrix whose columns are eigenvectors of $T$
- $D$ = diagonal matrix of corresponding eigenvalues

### 13.3 Computing Powers of Matrices

**Key insight:**

$$T^n = CD^nC^{-1}$$

**Proof:**

$$T^2 = (CDC^{-1})(CDC^{-1}) = CD(C^{-1}C)DC^{-1} = CD^2C^{-1}$$

**Why it's useful:** Computing $T^{1000000}$ directly requires millions of matrix multiplications. With diagonalization, we only need to raise diagonal elements to the power and do two matrix multiplications.

### 13.4 Example

For the matrix:

$$T = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$$

**Eigenvectors:** $(1, 0)$ with $\lambda = 1$, and $(1, 1)$ with $\lambda = 2$

$$C = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad D = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$$

Then:

$$T^2 = CD^2C^{-1} = \begin{pmatrix} 1 & 3 \\ 0 & 4 \end{pmatrix}$$

---

## Chapter 14: PageRank Algorithm

### 14.1 The Problem

**Goal:** Rank web pages by importance based on link structure.

**Assumption:** Important pages are linked to by other important pages.

### 14.2 The Link Matrix

**Construction:**
1. For each page, list which pages it links to
2. Normalize by total number of outgoing links
3. Use these as columns of link matrix $L$

**Example for 4 pages {A, B, C, D}:**

If page A links to B, C, D (3 links total):

$$L_A = \begin{pmatrix} 0 \\ 1/3 \\ 1/3 \\ 1/3 \end{pmatrix}$$

### 14.3 The Rank Equation

The rank of page $i$ depends on the ranks of all pages linking to it:

$$r_i = \sum_{j=1}^{n} L_{ij} \cdot r_j$$

In matrix form:

$$\vec{r} = L\vec{r}$$

**This is an eigenproblem!** We seek the eigenvector with eigenvalue 1.

### 14.4 Power Method (Iterative Solution)

**Step 1 - Initialize:**

$$\vec{r}_0 = \frac{1}{n}(1, 1, \ldots, 1)^T$$

**Step 2 - Iterate:**

$$\vec{r}_{i+1} = L\vec{r}_i$$

**Step 3 - Converge:** When $\vec{r}$ stops changing, we've found the eigenvector.

### 14.5 Damping Factor

To improve convergence and handle edge cases:

$$\vec{r}_{i+1} = d \cdot L\vec{r}_i + \frac{1-d}{n}\vec{1}$$

Where $d \approx 0.85$ represents probability of following a link (vs. randomly jumping).

**Why it's useful:** PageRank demonstrated that eigenvector analysis could be applied to massive real-world networks, powering Google's search engine.

---

## Summary of Key Formulas

### Vectors

| Concept | Formula |
|---------|---------|
| Dot product | $\vec{r} \cdot \vec{s} = \sum_i r_is_i = \|\vec{r}\|\|\vec{s}\|\cos\theta$ |
| Vector length | $\|\vec{r}\| = \sqrt{\vec{r} \cdot \vec{r}}$ |
| Scalar projection | $\frac{\vec{r} \cdot \vec{s}}{\|\vec{r}\|}$ |
| Vector projection | $\frac{\vec{r} \cdot \vec{s}}{\|\vec{r}\|^2}\vec{r}$ |

### Matrices

| Concept | Formula |
|---------|---------|
| 2×2 Determinant | $\det(A) = ad - bc$ |
| 2×2 Inverse | $A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$ |
| Orthogonal matrix | $A^{-1} = A^T$ |

### Eigenproblems

| Concept | Formula |
|---------|---------|
| Eigenvector equation | $A\vec{x} = \lambda\vec{x}$ |
| Characteristic equation | $\det(A - \lambda I) = 0$ |
| Diagonalization | $A = CDC^{-1}$ |
| Matrix power | $A^n = CD^nC^{-1}$ |

---

## Key Takeaways

1. **Vectors** are lists that can represent both geometric directions and data attributes.

2. **The dot product** measures how much two vectors "agree" and enables projection operations.

3. **Matrices** are linear transformations; their columns show where basis vectors go.

4. **The determinant** measures volume scaling and indicates whether a transformation is invertible.

5. **Basis change** lets us work in coordinate systems that simplify our problems.

6. **Orthonormal bases** make everything easier—inverses become transposes, and projections become dot products.

7. **Eigenvectors** are the "natural directions" of a transformation; **eigenvalues** are their scaling factors.

8. **Diagonalization** transforms repeated matrix operations from expensive multiplications into simple power operations.

9. **Real applications** like PageRank show how these abstract concepts power systems handling billions of data points.
