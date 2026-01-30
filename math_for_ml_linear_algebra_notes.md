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

```
2a + 3b = 8
10a + 1b = 13
```

This can be written as a matrix equation:

```
┌        ┐ ┌   ┐   ┌    ┐
│  2   3 │ │ a │   │  8 │
│ 10   1 │ │ b │ = │ 13 │
└        ┘ └   ┘   └    ┘
```

**Why it's useful:** This formulation allows us to solve systems with many variables and equations using computer algorithms, which would be impractical by hand.

### 1.3 The Optimization Problem

Fitting a function (like a Gaussian distribution) to data involves finding optimal parameter values.

**Gaussian Distribution:**

```
f(x) = (1 / (σ√(2π))) × exp(-(x - μ)² / (2σ²))
```

Where:
- μ = center/mean of distribution
- σ = width/standard deviation

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

```
        ┌ 120 m²        ┐
  h⃗  =  │ 2 bedrooms    │
        │ 1 bathroom    │
        └ 150,000 euros ┘
```

**Why it's useful:** Vectors generalize the concept of "moving in space" to include any collection of related quantities, enabling us to apply geometric intuition to abstract data problems.

### 2.2 Vector Operations

Vectors are defined by two fundamental operations:

#### Vector Addition

Adding vectors means placing them end-to-end:

```
r⃗ + s⃗ = s⃗ + r⃗
```

**Component-wise addition:**

```
┌ r₁ ┐   ┌ s₁ ┐   ┌ r₁ + s₁ ┐
│    │ + │    │ = │         │
└ r₂ ┘   └ s₂ ┘   └ r₂ + s₂ ┘
```

**Properties:**
- Commutative: r⃗ + s⃗ = s⃗ + r⃗
- Associative: (r⃗ + s⃗) + t⃗ = r⃗ + (s⃗ + t⃗)

#### Scalar Multiplication

Scaling a vector by a number changes its length:

```
      ┌ ar₁ ┐
ar⃗ = │     │
      └ ar₂ ┘
```

**Special cases:**
- −r⃗ = vector pointing in opposite direction
- r⃗ + (−r⃗) = 0⃗

**Why it's useful:** These operations allow us to combine and scale data, forming the basis for all linear algebra computations.

### 2.3 Coordinate Systems and Basis Vectors

A coordinate system is defined by basis vectors (typically denoted ê₁, ê₂ or î, ĵ).

**Standard Basis in 2D:**

```
      ┌ 1 ┐         ┌ 0 ┐
î  =  │   │ ,  ĵ =  │   │
      └ 0 ┘         └ 1 ┘
```

**Vector in terms of basis:**

```
r⃗ = r₁ê₁ + r₂ê₂ = (r₁, r₂)
```

**Why it's useful:** The coordinate system is arbitrary—the same vector can be described with different numbers depending on the chosen basis. This flexibility is crucial for simplifying problems.

---

## Chapter 3: The Dot Product (Inner/Scalar/Projection Product)

### 3.1 Definition of the Dot Product

The dot product multiplies corresponding components and sums them:

```
r⃗ · s⃗ = r₁s₁ + r₂s₂ + ... + rₙsₙ = Σᵢ rᵢsᵢ
```

**Example:**

```
┌  3 ┐   ┌ -1 ┐
│    │ · │    │ = 3(-1) + 2(2) = -3 + 4 = 1
└  2 ┘   └  2 ┘
```

**Why it's useful:** The dot product provides a single number that captures the relationship between two vectors—how much they "agree" in direction.

### 3.2 Properties of the Dot Product

1. **Commutative:** r⃗ · s⃗ = s⃗ · r⃗

2. **Distributive over addition:** r⃗ · (s⃗ + t⃗) = r⃗ · s⃗ + r⃗ · t⃗

3. **Associative with scalar multiplication:** r⃗ · (as⃗) = a(r⃗ · s⃗)

**Why it's useful:** These properties allow algebraic manipulation of dot products, essential for derivations and proofs.

### 3.3 Vector Length (Magnitude/Modulus)

The length of a vector is found using the dot product with itself:

```
|r⃗| = √(r⃗ · r⃗) = √(r₁² + r₂² + ... + rₙ²)
```

This follows from the Pythagorean theorem.

**Why it's useful:** Computing distances and normalizing vectors (making them unit length) are fundamental operations in ML.

### 3.4 Geometric Interpretation: Angle Between Vectors

From the cosine rule, we derive:

```
r⃗ · s⃗ = |r⃗| |s⃗| cos(θ)
```

**Implications:**
- If θ = 0° (same direction): r⃗ · s⃗ = |r⃗||s⃗| (maximum positive)
- If θ = 90° (perpendicular/orthogonal): r⃗ · s⃗ = 0
- If θ = 180° (opposite directions): r⃗ · s⃗ = −|r⃗||s⃗| (maximum negative)

**Why it's useful:** The dot product immediately tells us whether vectors point in similar directions (positive), are perpendicular (zero), or oppose each other (negative).

### 3.5 Projection

#### Scalar Projection

The "shadow" of s⃗ onto r⃗:

```
Scalar projection of s⃗ onto r⃗ = (r⃗ · s⃗) / |r⃗| = |s⃗| cos(θ)
```

#### Vector Projection

The projection as a vector in the direction of r⃗:

```
Vector projection of s⃗ onto r⃗ = [(r⃗ · s⃗) / (r⃗ · r⃗)] × r⃗
```

**Why it's useful:** Projection decomposes a vector into components along different directions—essential for changing coordinate systems and dimensionality reduction.

---

## Chapter 4: Changing Basis

### 4.1 Basis Vectors and Linear Independence

**Definition:** A basis is a set of n vectors that:
1. Are linearly independent (none can be written as a combination of others)
2. Span the space (any vector in the space can be expressed as their combination)

**Linear Independence Test:**

Vectors {b₁, b₂, ..., bₙ} are linearly independent if no bᵢ can be written as:

```
bᵢ = a₁b₁ + a₂b₂ + ... + aᵢ₋₁bᵢ₋₁ + aᵢ₊₁bᵢ₊₁ + ... + aₙbₙ
```

**Why it's useful:** The number of linearly independent basis vectors determines the dimensionality of the space. Choosing good basis vectors simplifies problems dramatically.

### 4.2 Changing Basis Using Projection

When the new basis vectors are **orthogonal** to each other, we can use the dot product to change coordinates.

**Given:** Vector r⃗ in basis {e₁, e₂}, new basis {b₁, b₂} where b₁ ⊥ b₂

**To find r⃗ in the new basis:**

```
r_b₁ = (r⃗ₑ · b⃗₁) / |b⃗₁|²

r_b₂ = (r⃗ₑ · b⃗₂) / |b⃗₂|²
```

**Orthogonality Check:**

```
b⃗₁ · b⃗₂ = 0  ⟹  orthogonal
```

**Why it's useful:** Transforming data to a better coordinate system can reveal structure (e.g., principal components) or simplify computations.

### 4.3 Dimensionality and Data

In data science, we often want to:
- Map high-dimensional data to lower dimensions
- Find directions that capture the most "information"
- Discard "noise" dimensions

**Example:** Points lying roughly on a line in 2D can be described by:
1. Distance along the line (signal)
2. Distance from the line (noise)

**Why it's useful:** Dimensionality reduction is fundamental to handling high-dimensional data in machine learning.

---

## Chapter 5: Introduction to Matrices

### 5.1 What is a Matrix?

A matrix is a rectangular array of numbers that represents a linear transformation—an operation that transforms vectors while preserving the grid structure of space.

**Matrix Notation:**

```
      ┌       ┐
A  =  │ a   b │
      │ c   d │
      └       ┘
```

**Matrix-Vector Multiplication:**

```
      ┌       ┐ ┌   ┐   ┌ ax + by ┐
Ar⃗ = │ a   b │ │ x │ = │         │
      │ c   d │ │ y │   │ cx + dy │
      └       ┘ └   ┘   └         ┘
```

**Why it's useful:** Matrices encode transformations compactly and allow us to apply the same operation to many vectors efficiently.

### 5.2 Matrices as Transformations of Basis Vectors

The columns of a transformation matrix tell us where the basis vectors go:

```
      ┌  |    |  ┐
A  =  │ e⃗₁'  e⃗₂' │
      └  |    |  ┘
```

**Example:**

```
      ┌        ┐
A  =  │  2   3 │
      │ 10   1 │
      └        ┘
```

- ê₁ = (1, 0) transforms to (2, 10)
- ê₂ = (0, 1) transforms to (3, 1)

**Why it's useful:** Understanding matrices as basis transformations provides geometric intuition for abstract operations.

### 5.3 Properties of Linear Transformations

Linear transformations preserve:
- Grid lines remain parallel and evenly spaced
- The origin stays fixed
- Linear combinations: A(nr⃗) = nAr⃗ and A(r⃗ + s⃗) = Ar⃗ + As⃗

**Why it's useful:** These properties ensure that vector operations remain valid after transformation.

---

## Chapter 6: Types of Matrices and Transformations

### 6.1 Identity Matrix

The matrix that does nothing:

```
      ┌       ┐
I  =  │ 1   0 │
      │ 0   1 │
      └       ┘
```

For any vector: Iv⃗ = v⃗

### 6.2 Scaling Matrices

Diagonal matrices scale each axis independently:

```
┌       ┐
│ a   0 │
│ 0   d │
└       ┘
```

- Scales x-axis by factor a
- Scales y-axis by factor d
- If a or d is a fraction, it compresses that axis

### 6.3 Reflection/Mirror Matrices

**Reflection about y-axis:**

```
┌        ┐
│ -1   0 │
│  0   1 │
└        ┘
```

**Reflection about x-axis:**

```
┌        ┐
│  1   0 │
│  0  -1 │
└        ┘
```

**Inversion (both axes):**

```
┌        ┐
│ -1   0 │
│  0  -1 │
└        ┘
```

**Mirror at 45°:**

```
┌       ┐
│ 0   1 │
│ 1   0 │
└       ┘
```

### 6.4 Shear Matrices

Shear keeps one axis fixed while tilting the other:

**Horizontal shear:**

```
┌       ┐
│ 1   k │
│ 0   1 │
└       ┘
```

**Vertical shear:**

```
┌       ┐
│ 1   0 │
│ k   1 │
└       ┘
```

### 6.5 Rotation Matrices

**Rotation by angle θ (counterclockwise):**

```
          ┌                   ┐
R(θ)  =   │ cos(θ)   -sin(θ) │
          │ sin(θ)    cos(θ) │
          └                   ┘
```

**Why these are useful:** Any shape change can be decomposed into combinations of these fundamental transformations.

---

## Chapter 7: Matrix Composition and Multiplication

### 7.1 Composing Transformations

Applying transformation A₁ followed by A₂ is equivalent to multiplying by A₂A₁:

```
A₂(A₁r⃗) = (A₂A₁)r⃗
```

**Important:** Order matters! The rightmost matrix is applied first.

### 7.2 Matrix Multiplication

**Rule:** Multiply rows of the left matrix by columns of the right matrix.

```
┌       ┐ ┌       ┐   ┌               ┐
│ a   b │ │ e   f │   │ ae+bg   af+bh │
│ c   d │ │ g   h │ = │ ce+dg   cf+dh │
└       ┘ └       ┘   └               ┘
```

**Properties:**
- **Associative:** (AB)C = A(BC)
- **NOT Commutative:** AB ≠ BA in general
- **Distributive:** A(B + C) = AB + AC

**Why it's useful:** Complex transformations can be pre-computed as a single matrix, making repeated applications efficient.

### 7.3 Einstein Summation Convention

A compact notation for matrix operations:

```
(AB)ᵢₖ = Σⱼ Aᵢⱼ Bⱼₖ = Aᵢⱼ Bⱼₖ
```

(Repeated index implies summation)

**Why it's useful:** Simplifies coding matrix operations—just loop over indices.

### 7.4 Non-Square Matrices

Matrices don't have to be square. An (m × n) matrix times an (n × p) matrix yields an (m × p) matrix:

```
(m × n) × (n × p) = (m × p)
```

**Requirement:** The inner dimensions must match.

---

## Chapter 8: Solving Systems of Linear Equations

### 8.1 Matrix Formulation

A system of equations:

```
a₁x + b₁y + c₁z = d₁
a₂x + b₂y + c₂z = d₂
a₃x + b₃y + c₃z = d₃
```

Can be written as:

```
Ax⃗ = d⃗
```

### 8.2 The Inverse Matrix

If A⁻¹ exists such that A⁻¹A = AA⁻¹ = I, then:

```
x⃗ = A⁻¹d⃗
```

**Why it's useful:** The inverse lets us "undo" a transformation and solve for unknown vectors.

### 8.3 Gaussian Elimination (Row Echelon Form)

**Method:**
1. **Elimination:** Subtract multiples of rows to create zeros below the diagonal
2. **Back-substitution:** Solve from bottom up

**Example:**

```
┌           ┐ ┌   ┐   ┌    ┐
│ 1   1   3 │ │ a │   │ 15 │
│ 1   2   4 │ │ b │ = │ 21 │
│ 1   1   2 │ │ c │   │ 13 │
└           ┘ └   ┘   └    ┘
```

After elimination:

```
┌           ┐ ┌   ┐   ┌    ┐
│ 1   1   3 │ │ a │   │ 15 │
│ 0   1   1 │ │ b │ = │  6 │
│ 0   0   1 │ │ c │   │  2 │
└           ┘ └   ┘   └    ┘
```

**Triangular form** allows easy back-substitution: c = 2, then b = 4, then a = 5.

**Why it's useful:** This is computationally efficient and works for any size system.

### 8.4 Finding the Inverse Matrix

To find A⁻¹, augment A with I and reduce A to I:

```
[A | I] → [I | A⁻¹]
```

**2×2 Inverse Formula:**

```
      ┌       ┐                    1       ┌        ┐
A  =  │ a   b │   ⟹   A⁻¹  =  ─────────  │  d  -b │
      │ c   d │               (ad - bc)   │ -c   a │
      └       ┘                           └        ┘
```

**Why it's useful:** The inverse provides a general solution valid for any right-hand side vector.

---

## Chapter 9: Determinants

### 9.1 Geometric Meaning

The determinant measures how much a transformation **scales area** (2D) or **volume** (3D).

**For 2×2:**

```
         ┌       ┐
det(A) = │ a   b │ = ad - bc
         │ c   d │
         └       ┘
```

**Properties:**
- |det(A)| > 1: Space expanded
- |det(A)| < 1: Space compressed
- det(A) < 0: Orientation flipped (handedness changed)
- det(A) = 0: Space collapsed to lower dimension

### 9.2 Determinant and Invertibility

```
det(A) = 0  ⟺  A has no inverse
```

**Why:** If the determinant is zero, the transformation collapses space (loses information), which cannot be undone.

### 9.3 Linear Dependence

If det(A) = 0, the columns of A are **linearly dependent**—one can be written as a combination of the others.

**Why it's useful:** The determinant provides a quick check for whether a system of equations has a unique solution.

---

## Chapter 10: Changing Basis with Matrices

### 10.1 Basis Transformation Matrices

To transform a vector from one basis to another:

**Bear's basis vectors in my coordinates:**

```
      ┌  |    |  ┐
B  =  │ b⃗₁   b⃗₂ │
      └  |    |  ┘
```

**Transformation:**
- B · v⃗_bear = v⃗_mine (Bear's vector to my coordinates)
- B⁻¹ · v⃗_mine = v⃗_bear (My vector to Bear's coordinates)

### 10.2 Performing Transformations in Different Bases

To apply transformation T (defined in standard basis) to vectors in basis B:

```
T_B = B⁻¹ T B
```

**Process:**
1. Convert to standard basis: Bv⃗
2. Apply transformation: T(Bv⃗)
3. Convert back: B⁻¹T(Bv⃗)

**Why it's useful:** This allows us to work in whichever coordinate system makes the problem easiest.

---

## Chapter 11: Orthogonal Matrices and the Gram-Schmidt Process

### 11.1 Transpose of a Matrix

The transpose swaps rows and columns:

```
(Aᵀ)ᵢⱼ = Aⱼᵢ
```

**Example:**

```
┌       ┐ᵀ    ┌       ┐
│ 1   2 │     │ 1   3 │
│ 3   4 │  =  │ 2   4 │
└       ┘     └       ┘
```

### 11.2 Orthonormal Basis

A set of vectors is **orthonormal** if:
- All vectors are mutually orthogonal: e⃗ᵢ · e⃗ⱼ = 0 for i ≠ j
- All vectors have unit length: |e⃗ᵢ| = 1

### 11.3 Orthogonal Matrices

A matrix A is **orthogonal** if its columns form an orthonormal basis.

**Key Property:**

```
AᵀA = AAᵀ = I
```

Therefore:

```
A⁻¹ = Aᵀ
```

**Properties:**
- |det(A)| = 1 (preserves area/volume)
- Represents pure rotation and/or reflection

**Why it's useful:** Inverting orthogonal matrices is trivial (just transpose), and projections become simple dot products.

### 11.4 The Gram-Schmidt Process

**Purpose:** Convert any set of linearly independent vectors into an orthonormal basis.

**Algorithm:**

Given vectors {v₁, v₂, ..., vₙ}:

**Step 1 - First basis vector:**

```
e₁ = v₁ / |v₁|
```

**Step 2 - Second basis vector:**

```
u₂ = v₂ - (v₂ · e₁)e₁
e₂ = u₂ / |u₂|
```

**Step 3 - Third basis vector:**

```
u₃ = v₃ - (v₃ · e₁)e₁ - (v₃ · e₂)e₂
e₃ = u₃ / |u₃|
```

**Continue for remaining vectors...**

**Why it's useful:** Many algorithms require orthonormal bases. Gram-Schmidt systematically creates one from any starting vectors.

### 11.5 Example: Reflection in an Arbitrary Plane

**Problem:** Reflect a point through a plane defined by vectors v₁ and v₂.

**Solution:**
1. Use Gram-Schmidt to find orthonormal basis: e₁, e₂ (in plane), e₃ (normal to plane)
2. Build transformation matrix E = [e₁ | e₂ | e₃]
3. Define reflection in this basis:

```
       ┌           ┐
T_E =  │ 1   0   0 │
       │ 0   1   0 │
       │ 0   0  -1 │
       └           ┘
```

4. Transform back: T = E · T_E · Eᵀ

---

## Chapter 12: Eigenvalues and Eigenvectors

### 12.1 The Eigen-Problem

**Eigenvectors** are vectors that remain on their original span after a transformation (they may stretch or flip, but don't change direction).

**Eigenvalues** are the scaling factors applied to eigenvectors.

**Definition:**

```
Ax⃗ = λx⃗
```

Where:
- A = transformation matrix
- x⃗ = eigenvector
- λ = eigenvalue

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

Rearranging Ax⃗ = λx⃗:

```
(A - λI)x⃗ = 0
```

For non-trivial solutions, the matrix must be singular:

```
det(A - λI) = 0
```

**For 2×2 matrices:**

```
     ┌               ┐
det  │ a - λ     b   │ = 0
     │   c     d - λ │
     └               ┘
```

This gives the **characteristic polynomial**:

```
λ² - (a + d)λ + (ad - bc) = 0
```

Its roots are the eigenvalues.

### 12.4 Calculating Eigenvectors

Once eigenvalues are known, substitute back:

```
(A - λI)x⃗ = 0
```

Solve for x⃗.

**Example:** For A with diagonal entries 1 and 2:

```
      ┌       ┐
A  =  │ 1   0 │
      │ 0   2 │
      └       ┘
```

- λ₁ = 1: eigenvector x⃗ = t(1, 0) (any horizontal vector)
- λ₂ = 2: eigenvector x⃗ = t(0, 1) (any vertical vector)

### 12.5 Special Cases

**No real eigenvectors:** Rotation matrices (except 180°) have only complex eigenvalues.

**All vectors are eigenvectors:** Uniform scaling and 180° rotation.

**Why it's useful:** Eigenanalysis simplifies understanding complex transformations by finding their fundamental directions.

---

## Chapter 13: Diagonalization

### 13.1 The Power of Diagonal Matrices

For diagonal matrices, powers are trivial:

```
      ┌           ┐            ┌              ┐
D  =  │ a   0   0 │     Dⁿ =   │ aⁿ   0    0  │
      │ 0   b   0 │   ⟹       │  0   bⁿ   0  │
      │ 0   0   c │            │  0    0   cⁿ │
      └           ┘            └              ┘
```

### 13.2 Diagonalization Process

If matrix T has n linearly independent eigenvectors, it can be diagonalized:

```
T = C D C⁻¹
```

Where:
- C = matrix whose columns are eigenvectors of T
- D = diagonal matrix of corresponding eigenvalues

### 13.3 Computing Powers of Matrices

**Key insight:**

```
Tⁿ = C Dⁿ C⁻¹
```

**Proof:**

```
T² = (CDC⁻¹)(CDC⁻¹) = CD(C⁻¹C)DC⁻¹ = CD²C⁻¹
```

**Why it's useful:** Computing T¹⁰⁰⁰⁰⁰⁰ directly requires millions of matrix multiplications. With diagonalization, we only need to raise diagonal elements to the power and do two matrix multiplications.

### 13.4 Example

For the matrix:

```
      ┌       ┐
T  =  │ 1   1 │
      │ 0   2 │
      └       ┘
```

**Eigenvectors:** (1, 0) with λ = 1, and (1, 1) with λ = 2

```
      ┌       ┐         ┌       ┐
C  =  │ 1   1 │ ,  D =  │ 1   0 │
      │ 0   1 │         │ 0   2 │
      └       ┘         └       ┘
```

Then:

```
T² = CDC⁻¹ = ... = ┌       ┐
                   │ 1   3 │
                   │ 0   4 │
                   └       ┘
```

---

## Chapter 14: PageRank Algorithm

### 14.1 The Problem

**Goal:** Rank web pages by importance based on link structure.

**Assumption:** Important pages are linked to by other important pages.

### 14.2 The Link Matrix

**Construction:**
1. For each page, list which pages it links to
2. Normalize by total number of outgoing links
3. Use these as columns of link matrix L

**Example for 4 pages {A, B, C, D}:**

If page A links to B, C, D (3 links total):

```
        ┌   0   ┐
L_A =   │  1/3  │
        │  1/3  │
        └  1/3  ┘
```

### 14.3 The Rank Equation

The rank of page i depends on the ranks of all pages linking to it:

```
rᵢ = Σⱼ Lᵢⱼ × rⱼ
```

In matrix form:

```
r⃗ = Lr⃗
```

**This is an eigenproblem!** We seek the eigenvector with eigenvalue 1.

### 14.4 Power Method (Iterative Solution)

**Step 1 - Initialize:**

```
r⃗₀ = (1/n)(1, 1, ..., 1)ᵀ
```

**Step 2 - Iterate:**

```
r⃗ᵢ₊₁ = L r⃗ᵢ
```

**Step 3 - Converge:** When r⃗ stops changing, we've found the eigenvector.

### 14.5 Damping Factor

To improve convergence and handle edge cases:

```
r⃗ᵢ₊₁ = d × Lr⃗ᵢ + [(1-d)/n] × 1⃗
```

Where d ≈ 0.85 represents probability of following a link (vs. randomly jumping).

**Why it's useful:** PageRank demonstrated that eigenvector analysis could be applied to massive real-world networks, powering Google's search engine.

---

## Summary of Key Formulas

### Vectors

| Concept | Formula |
|---------|---------|
| Dot product | r⃗ · s⃗ = Σᵢ rᵢsᵢ = \|r⃗\| \|s⃗\| cos(θ) |
| Vector length | \|r⃗\| = √(r⃗ · r⃗) |
| Scalar projection | (r⃗ · s⃗) / \|r⃗\| |
| Vector projection | [(r⃗ · s⃗) / (r⃗ · r⃗)] × r⃗ |

### Matrices

| Concept | Formula |
|---------|---------|
| 2×2 Determinant | det = ad - bc |
| 2×2 Inverse | A⁻¹ = [1/(ad-bc)] × [d, -b; -c, a] |
| Orthogonal matrix | A⁻¹ = Aᵀ |

### Eigenproblems

| Concept | Formula |
|---------|---------|
| Eigenvector equation | Ax⃗ = λx⃗ |
| Characteristic equation | det(A - λI) = 0 |
| Diagonalization | A = CDC⁻¹ |
| Matrix power | Aⁿ = CDⁿC⁻¹ |

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
