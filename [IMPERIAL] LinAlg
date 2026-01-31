# Mathematics for Machine Learning: Linear Algebra

## Course Overview

Linear algebra provides the mathematical foundation for machine learning by offering a set of notational conventions and operations that allow us to manipulate large systems of equations conveniently. This course focuses on building intuition about vectors, matrices, and transformations rather than just mechanical calculations.

The power of linear algebra lies in its ability to generalize. Once you understand how to solve a system of two equations with two unknowns, the same techniques scale to millions of equations with millions of unknowns—the kind of problems that arise routinely in machine learning. A neural network with millions of parameters, a recommendation system analyzing millions of users, or a search engine ranking billions of web pages all rely on the same fundamental operations we'll explore here.

---

## Chapter 1: Introduction and Motivation

### 1.1 Why Linear Algebra for Machine Learning?

Linear algebra is essential for machine learning because it provides tools to represent, interpret, and control complex systems. While open-source libraries allow applying ML methods without deep mathematical understanding, problems inevitably arise—and without knowledge of the underlying mathematics, debugging becomes nearly impossible.

Consider what happens when you train a neural network. At its core, the network performs thousands of matrix multiplications, applies transformations to high-dimensional vectors, and adjusts its parameters by following gradients through a multidimensional space. When something goes wrong—the model doesn't converge, produces strange outputs, or fails to generalize—understanding the linear algebra underneath is often the key to diagnosing and fixing the problem.

Beyond debugging, understanding linear algebra gives you the ability to design better models, implement more efficient algorithms, and develop intuitions about why certain approaches work while others fail. It transforms you from someone who uses tools to someone who truly understands them.

### 1.2 The Apples and Bananas Problem

Let's start with a concrete problem that motivates the entire field. Imagine you go shopping twice and record only the totals:

**Problem Setup:**
- Trip 1: You buy 2 apples and 3 bananas and pay 8 euros total
- Trip 2: You buy 10 apples and 1 banana and pay 13 euros total

What is the price of a single apple? What about a single banana?

**Traditional Approach:**

You could solve this with algebra you learned in school. Let $a$ be the price of an apple and $b$ be the price of a banana. Then:

$$2a + 3b = 8$$

$$10a + b = 13$$

From the second equation, $b = 13 - 10a$. Substituting into the first: $2a + 3(13-10a) = 8$, which gives $2a + 39 - 30a = 8$, so $-28a = -31$, meaning $a = \frac{31}{28} \approx 1.11$ euros. Then $b = 13 - 10(1.11) \approx 1.89$ euros.

This works, but it's clumsy. What if you had 100 different items and 100 different shopping trips? The substitution method becomes impossibly complex.

**The Linear Algebra Approach:**

Linear algebra lets us write this system compactly as:

$$\begin{bmatrix} 2 & 3 \\\ 10 & 1 \end{bmatrix} \begin{bmatrix} a \\\ b \end{bmatrix} = \begin{bmatrix} 8 \\\ 13 \end{bmatrix}$$

This notation, which we write as $A\vec{x} = \vec{b}$, encapsulates the entire problem in a single equation. The solution is simply $\vec{x} = A^{-1}\vec{b}$—multiply both sides by the inverse of $A$. The same formula works whether you have 2 unknowns or 2 million unknowns. This is the power of abstraction: we develop general tools that work regardless of the problem size.

### 1.3 The Optimization Problem

Machine learning problems are fundamentally about fitting functions to data. Consider the problem of fitting a Gaussian (bell curve) distribution to a set of measurements. The Gaussian distribution is described by:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Here, $\mu$ determines where the center of the bell curve sits (the mean), and $\sigma$ determines how wide or narrow the curve is (the standard deviation). Given a dataset, we want to find the values of $\mu$ and $\sigma$ that make the curve best fit our data.

We can think of $\mu$ and $\sigma$ as coordinates in a two-dimensional "parameter space." Every point in this space represents a different Gaussian curve. Somewhere in this space is the best curve—the one that fits our data most closely.

To find this optimal point, we define a "loss function" that measures how poorly a given curve fits the data. We then search for the point where this loss is minimized. The search process involves moving through parameter space, and each move is a vector. To find the direction of steepest descent, we need calculus on vectors—gradient descent. Understanding vectors, their directions, and how functions change as we move through space is essential to this entire process.

This is why linear algebra and calculus on vectors form the mathematical backbone of machine learning optimization.

### 1.4 Neural Networks and Beyond

In a neural network, data flows through layers of transformations. Each layer performs an operation that can be written as:

$$\vec{y} = \sigma(W\vec{x} + \vec{b})$$

Here, $W$ is a matrix of weights, $\vec{x}$ is the input vector, $\vec{b}$ is a bias vector, and $\sigma$ is a nonlinear activation function applied element-wise. The matrix multiplication $W\vec{x}$ is a linear transformation—it rotates, scales, and projects the input. Understanding what these transformations do geometrically helps you understand what the network is learning and why certain architectures work better than others.

Principal Component Analysis (PCA), a fundamental dimensionality reduction technique, is entirely based on eigenvalues and eigenvectors. Singular Value Decomposition (SVD), used in recommendation systems and image compression, extends these ideas. PageRank, the algorithm that powered Google's original search engine, is an eigenvector problem. Wherever you look in machine learning, linear algebra is there.

---

## Chapter 2: Vectors

### 2.1 What is a Vector?

A vector is one of the most fundamental objects in mathematics, and it admits several complementary interpretations.

**The Geometric Interpretation:** In physics and geometry, a vector represents something that has both magnitude (size) and direction. Think of a vector as an arrow pointing from one point to another. The velocity of a car is a vector: it tells you both how fast the car is moving and in which direction. The same velocity can be described from any starting point—it's the arrow itself that matters, not where you draw it.

Crucially, a vector is defined by its direction and magnitude, not by its position. Two arrows pointing the same way with the same length represent the same vector, regardless of where they're drawn. This is why vectors are perfect for representing change, movement, or displacement.

**The Data Science Interpretation:** In data science and machine learning, we use vectors to represent collections of related numbers. A house might be described by its area, number of bedrooms, number of bathrooms, and price:

$$\vec{h} = \begin{bmatrix} 120 \\\ 2 \\\ 1 \\\ 150000 \end{bmatrix}$$

This four-dimensional vector captures everything we know about the house in a single mathematical object. Each component represents a different feature, and together they define a point in a four-dimensional "feature space." While we can't visualize four dimensions, the mathematics works exactly the same as in two or three dimensions.

This dual interpretation—geometric arrow and list of numbers—is what makes vectors so powerful. We can apply geometric intuition (angles, projections, distances) to abstract data problems, and we can use computational tools (component-wise operations) to solve geometric problems.

### 2.2 Vector Addition

Vectors are defined by two fundamental operations: addition and scalar multiplication. Everything else builds from these.

**Geometric Picture:** When you add two vectors, you place them end-to-end. If $\vec{r}$ represents "walk 3 steps east" and $\vec{s}$ represents "walk 4 steps north," then $\vec{r} + \vec{s}$ represents the combined journey—ending up 3 steps east and 4 steps north of where you started.

**Component-wise Computation:**

$$\begin{bmatrix} r_1 \\\ r_2 \end{bmatrix} + \begin{bmatrix} s_1 \\\ s_2 \end{bmatrix} = \begin{bmatrix} r_1 + s_1 \\\ r_2 + s_2 \end{bmatrix}$$

Each component adds independently. If $\vec{r} = (3, 4)$ and $\vec{s} = (1, -2)$, then $\vec{r} + \vec{s} = (4, 2)$.

**Properties:** Vector addition is commutative ($\vec{r} + \vec{s} = \vec{s} + \vec{r}$) and associative (($\vec{r} + \vec{s}) + \vec{t} = \vec{r} + (\vec{s} + \vec{t})$). These properties match our geometric intuition: it doesn't matter which order you chain the arrows together; you'll end up at the same place.

There's also a zero vector $\vec{0}$ (all components are zero) that serves as an identity: $\vec{r} + \vec{0} = \vec{r}$. And every vector has an additive inverse: $\vec{r} + (-\vec{r}) = \vec{0}$.

### 2.3 Scalar Multiplication

A scalar is just a regular number (as opposed to a vector). Multiplying a vector by a scalar scales its length:

$$a\vec{r} = \begin{bmatrix} ar_1 \\\ ar_2 \end{bmatrix}$$

If $a > 1$, the vector gets longer. If $0 < a < 1$, it gets shorter. If $a = -1$, the vector flips to point in the opposite direction while keeping the same length. If $a = 0$, you get the zero vector.

**Geometric Picture:** Scalar multiplication stretches or compresses the arrow, and negative scalars flip its direction. The vector $2\vec{r}$ is twice as long as $\vec{r}$ and points the same way. The vector $-\vec{r}$ has the same length as $\vec{r}$ but points opposite.

**Properties:** Scalar multiplication distributes over vector addition: $a(\vec{r} + \vec{s}) = a\vec{r} + a\vec{s}$. It also distributes over scalar addition: $(a + b)\vec{r} = a\vec{r} + b\vec{r}$. These properties, combined with vector addition, make the set of vectors into a structure called a "vector space"—a rich algebraic framework with deep consequences.

### 2.4 Coordinate Systems and Basis Vectors

To work with vectors numerically, we need a coordinate system. This requires choosing basis vectors—a set of reference directions against which all other vectors are measured.

**Standard Basis in 2D:** The most common choice is the standard basis, consisting of unit vectors along each axis:

$$\hat{i} = \begin{bmatrix} 1 \\\ 0 \end{bmatrix}, \quad \hat{j} = \begin{bmatrix} 0 \\\ 1 \end{bmatrix}$$

These are often written as $\hat{e}_1$ and $\hat{e}_2$ in more general contexts.

**Expressing Vectors in a Basis:** Any vector in the plane can be written as a combination of $\hat{i}$ and $\hat{j}$:

$$\vec{r} = r_1\hat{i} + r_2\hat{j} = r_1\begin{bmatrix} 1 \\\ 0 \end{bmatrix} + r_2\begin{bmatrix} 0 \\\ 1 \end{bmatrix} = \begin{bmatrix} r_1 \\\ r_2 \end{bmatrix}$$

The numbers $r_1$ and $r_2$ are called the components or coordinates of $\vec{r}$ in this basis.

**The Arbitrariness of Coordinates:** Here's a crucial insight: the vector itself is a geometric object that exists independently of any coordinate system. The numbers we use to describe it depend on our choice of basis. If we chose different basis vectors—say, vectors pointing northeast and northwest—the same vector $\vec{r}$ would have different numerical components.

This might seem like a nuisance, but it's actually a superpower. By choosing the right basis for a problem, we can simplify calculations dramatically. Much of linear algebra is about understanding how to change bases and what stays the same when we do.

---

## Chapter 3: The Dot Product

### 3.1 Definition and Computation

The dot product (also called inner product or scalar product) takes two vectors and produces a single number. It's defined as:

$$\vec{r} \cdot \vec{s} = r_1s_1 + r_2s_2 + \ldots + r_ns_n = \sum_{i=1}^{n} r_is_i$$

You multiply corresponding components and add up the results.

**Example:**

$$\begin{bmatrix} 3 \\\ 2 \end{bmatrix} \cdot \begin{bmatrix} -1 \\\ 2 \end{bmatrix} = 3 \times (-1) + 2 \times 2 = -3 + 4 = 1$$

The computation is simple, but the dot product's significance goes far beyond this formula.

### 3.2 Algebraic Properties

The dot product satisfies several important properties:

**Commutative:** $\vec{r} \cdot \vec{s} = \vec{s} \cdot \vec{r}$. The order doesn't matter.

**Distributive:** $\vec{r} \cdot (\vec{s} + \vec{t}) = \vec{r} \cdot \vec{s} + \vec{r} \cdot \vec{t}$. You can distribute the dot product over addition.

**Scalar Association:** $\vec{r} \cdot (a\vec{s}) = a(\vec{r} \cdot \vec{s})$. You can pull scalars out.

**Positive Definiteness:** $\vec{r} \cdot \vec{r} \geq 0$, with equality only when $\vec{r} = \vec{0}$. A vector dotted with itself is always non-negative.

These properties allow algebraic manipulation of expressions involving dot products, which is essential for proofs and derivations.

### 3.3 The Length of a Vector

The dot product gives us a beautiful formula for a vector's length (magnitude):

$$|\vec{r}| = \sqrt{\vec{r} \cdot \vec{r}} = \sqrt{r_1^2 + r_2^2 + \ldots + r_n^2}$$

This is just the Pythagorean theorem generalized to any number of dimensions. In 2D, if $\vec{r} = (3, 4)$, then $|\vec{r}| = \sqrt{9 + 16} = \sqrt{25} = 5$. The vector forms a 3-4-5 right triangle with the axes.

A unit vector is a vector with length 1. To create a unit vector pointing in the same direction as $\vec{r}$, divide by its length:

$$\hat{r} = \frac{\vec{r}}{|\vec{r}|}$$

This process is called normalization. Unit vectors are useful because they represent pure direction without any magnitude information.

### 3.4 The Geometric Meaning: Angles

The dot product encodes information about the angle between vectors. This is captured by a fundamental formula:

$$\vec{r} \cdot \vec{s} = |\vec{r}||\vec{s}|\cos\theta$$

where $\theta$ is the angle between $\vec{r}$ and $\vec{s}$.

**Derivation:** Consider two vectors $\vec{r}$ and $\vec{s}$ with angle $\theta$ between them. The vector $\vec{r} - \vec{s}$ forms the third side of a triangle. By the law of cosines:

$$|\vec{r} - \vec{s}|^2 = |\vec{r}|^2 + |\vec{s}|^2 - 2|\vec{r}||\vec{s}|\cos\theta$$

Expanding the left side using the dot product:

$$(\vec{r} - \vec{s}) \cdot (\vec{r} - \vec{s}) = \vec{r} \cdot \vec{r} - 2\vec{r} \cdot \vec{s} + \vec{s} \cdot \vec{s} = |\vec{r}|^2 - 2\vec{r} \cdot \vec{s} + |\vec{s}|^2$$

Comparing the two expressions gives us $\vec{r} \cdot \vec{s} = |\vec{r}||\vec{s}|\cos\theta$.

**Implications of the Formula:**

When $\theta = 0°$ (vectors point the same direction), $\cos\theta = 1$, so $\vec{r} \cdot \vec{s} = |\vec{r}||\vec{s}|$—the maximum possible value.

When $\theta = 90°$ (vectors are perpendicular), $\cos\theta = 0$, so $\vec{r} \cdot \vec{s} = 0$. This gives us a quick test for perpendicularity: two vectors are orthogonal if and only if their dot product is zero.

When $\theta = 180°$ (vectors point opposite directions), $\cos\theta = -1$, so $\vec{r} \cdot \vec{s} = -|\vec{r}||\vec{s}|$—the minimum possible value.

In general, the sign of the dot product tells you whether vectors point roughly the same direction (positive), are perpendicular (zero), or point roughly opposite (negative).

### 3.5 Projection

Projection is one of the most useful applications of the dot product. It answers the question: "How much of vector $\vec{s}$ points in the direction of $\vec{r}$?"

**Scalar Projection:** The scalar projection of $\vec{s}$ onto $\vec{r}$ is the length of the "shadow" cast by $\vec{s}$ onto the line defined by $\vec{r}$:

$$\text{Scalar projection} = |\vec{s}|\cos\theta = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|}$$

This can be positive (if $\vec{s}$ has a component in the direction of $\vec{r}$) or negative (if $\vec{s}$ points somewhat opposite to $\vec{r}$).

**Vector Projection:** The vector projection is the actual vector component of $\vec{s}$ in the direction of $\vec{r}$:

$$\text{proj}_{\vec{r}}\vec{s} = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|^2}\vec{r} = \frac{\vec{r} \cdot \vec{s}}{\vec{r} \cdot \vec{r}}\vec{r}$$

This formula takes the scalar projection, divides by $|\vec{r}|$ to get a coefficient, then multiplies by $\vec{r}$ to get a vector pointing in the direction of $\vec{r}$.

**Decomposition:** Any vector $\vec{s}$ can be decomposed into two parts: one parallel to $\vec{r}$ (the projection) and one perpendicular to $\vec{r}$:

$$\vec{s} = \text{proj}_{\vec{r}}\vec{s} + \vec{s}_\perp$$

where $\vec{s}_\perp = \vec{s} - \text{proj}_{\vec{r}}\vec{s}$ is perpendicular to $\vec{r}$.

Projection is fundamental to changing coordinate systems, least-squares regression, and dimensionality reduction.

---

## Chapter 4: Changing Basis

### 4.1 Linear Independence

Before we can change bases, we need to understand what makes a valid basis. A set of vectors $\{v_1, v_2, \ldots, v_n\}$ is linearly independent if none of them can be written as a combination of the others.

**Formal Definition:** The vectors are linearly independent if the only solution to

$$a_1v_1 + a_2v_2 + \ldots + a_nv_n = \vec{0}$$

is $a_1 = a_2 = \ldots = a_n = 0$ (the trivial solution).

**Geometric Intuition:** In 2D, two vectors are linearly independent if they don't point along the same line. In 3D, three vectors are linearly independent if they don't all lie in the same plane. Independent vectors "point in genuinely different directions."

**Why It Matters:** If vectors are linearly dependent, they contain redundant information. You can't use them as a basis because they don't span the full space uniquely. A basis must have exactly as many linearly independent vectors as the dimension of the space.

### 4.2 What is a Basis?

A basis for a vector space is a set of vectors that:
1. Are linearly independent
2. Span the space (any vector can be expressed as their linear combination)

In $\mathbb{R}^n$, a basis consists of exactly $n$ linearly independent vectors. There are infinitely many possible bases for any vector space, but they all have the same number of vectors.

**Example:** In $\mathbb{R}^2$, the standard basis is $\{\hat{i}, \hat{j}\}$. But $\{(1, 1), (1, -1)\}$ is also a valid basis—these two vectors are linearly independent and span the plane. The vector $(3, 1)$ can be written as $2(1, 1) + 1(1, -1)$ in this new basis, so its coordinates are $(2, 1)$ in this system (compared to $(3, 1)$ in the standard basis).

### 4.3 Changing Basis Using Projection

When the basis vectors are orthogonal (perpendicular to each other), changing basis is straightforward using projection.

**Setup:** Suppose we have a vector $\vec{r}$ expressed in the standard basis, and we want to find its components in a new orthogonal basis $\{b_1, b_2\}$.

**Formula:** The component along $b_1$ is:

$$r_{b_1} = \frac{\vec{r} \cdot \vec{b}_1}{|\vec{b}_1|^2}$$

And similarly for $b_2$. We're finding how much of $\vec{r}$ projects onto each basis direction.

**Example:** Let $\vec{r} = (3, 1)$ and suppose our new basis is $b_1 = (1, 1)$ and $b_2 = (1, -1)$. First, verify orthogonality: $b_1 \cdot b_2 = 1 - 1 = 0$. Good.

Now compute:
- $r_{b_1} = \frac{(3, 1) \cdot (1, 1)}{|(1, 1)|^2} = \frac{4}{2} = 2$
- $r_{b_2} = \frac{(3, 1) \cdot (1, -1)}{|(1, -1)|^2} = \frac{2}{2} = 1$

So in the new basis, $\vec{r}$ has coordinates $(2, 1)$. Check: $2(1, 1) + 1(1, -1) = (2, 2) + (1, -1) = (3, 1)$. ✓

### 4.4 Why Change Basis?

Changing basis is a powerful technique for simplifying problems. In data science, we often want to find a basis aligned with the "natural directions" of our data.

**Principal Component Analysis (PCA):** Imagine a cloud of data points that's elongated along some direction. The standard $(x, y)$ coordinates might not be the most informative way to describe these points. PCA finds a new basis where the first axis points along the direction of maximum variance (the "principal component"). In this new coordinate system, the data's structure becomes clearer, and we might be able to discard low-variance directions as noise.

**Simplifying Transformations:** Some matrices become diagonal (only nonzero entries on the main diagonal) in the right basis. Diagonal matrices are trivial to work with—raising them to powers, for instance, just means raising the diagonal entries to that power. The technique of diagonalization (Chapter 13) relies on finding the right basis to simplify a matrix.

---

## Chapter 5: Introduction to Matrices

### 5.1 What is a Matrix?

A matrix is a rectangular array of numbers, but this description misses its deeper significance. A matrix represents a linear transformation—a function that takes vectors as input and produces vectors as output while preserving the structure of vector addition and scalar multiplication.

**Notation:** An $m \times n$ matrix has $m$ rows and $n$ columns:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\\ a_{21} & a_{22} & \cdots & a_{2n} \\\ \vdots & \vdots & \ddots & \vdots \\\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

The entry in row $i$ and column $j$ is written $a_{ij}$ or $A_{ij}$.

### 5.2 Matrix-Vector Multiplication

When a matrix $A$ multiplies a vector $\vec{x}$, it transforms $\vec{x}$ into a new vector $\vec{y} = A\vec{x}$.

**Computation:**

$$A\vec{x} = \begin{bmatrix} a & b \\\ c & d \end{bmatrix}\begin{bmatrix} x \\\ y \end{bmatrix} = \begin{bmatrix} ax + by \\\ cx + dy \end{bmatrix}$$

Each component of the output is the dot product of a row of $A$ with the input vector.

**Alternative View:** We can also see this as a linear combination of the columns of $A$:

$$A\vec{x} = x \begin{bmatrix} a \\\ c \end{bmatrix} + y \begin{bmatrix} b \\\ d \end{bmatrix}$$

The output is $x$ times the first column plus $y$ times the second column. This column view is often more geometrically insightful.

### 5.3 Matrices as Transformations of Basis Vectors

Here's the key insight that makes matrices intuitive: the columns of a matrix tell you where the basis vectors go under the transformation.

If $A = \begin{bmatrix} a & b \\\ c & d \end{bmatrix}$, then:
- The first column $(a, c)^T$ is where $\hat{i} = (1, 0)^T$ lands
- The second column $(b, d)^T$ is where $\hat{j} = (0, 1)^T$ lands

**Verification:** Apply $A$ to $\hat{i}$:

$$A\hat{i} = \begin{bmatrix} a & b \\\ c & d \end{bmatrix}\begin{bmatrix} 1 \\\ 0 \end{bmatrix} = \begin{bmatrix} a \\\ c \end{bmatrix}$$

The basis vector $\hat{i}$ gets mapped to the first column. Similarly, $\hat{j}$ gets mapped to the second column.

**Building Intuition:** To understand any matrix, ask yourself: where do the basis vectors go? If the first column is $(2, 0)^T$, the transformation stretches $\hat{i}$ by a factor of 2. If the second column is $(0, -1)^T$, it flips $\hat{j}$ to point downward. Once you know where the basis vectors land, any other vector's destination follows by linearity.

### 5.4 Properties of Linear Transformations

A transformation $T$ is linear if it satisfies:
1. $T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})$ — it respects addition
2. $T(c\vec{v}) = cT(\vec{v})$ — it respects scalar multiplication

**Geometric Consequences:** Linear transformations preserve:
- The origin (it always maps to itself)
- Straight lines (they remain straight)
- Parallel lines (they stay parallel)
- Ratios of distances along lines

What they can change:
- Lengths (stretching/compressing)
- Angles (shearing)
- Orientation (flipping/reflecting)

**Non-example:** Translation (shifting all points by a fixed vector) is NOT a linear transformation because it moves the origin.

---

## Chapter 6: Types of Matrix Transformations

### 6.1 Identity Matrix

The identity matrix does nothing—it maps every vector to itself:

$$I = \begin{bmatrix} 1 & 0 \\\ 0 & 1 \end{bmatrix}$$

Each basis vector maps to itself: $\hat{i} \to (1, 0)^T = \hat{i}$ and $\hat{j} \to (0, 1)^T = \hat{j}$.

For any vector $\vec{v}$: $I\vec{v} = \vec{v}$.

For any matrix $A$: $AI = IA = A$.

### 6.2 Scaling Matrices

Diagonal matrices scale each axis independently:

$$S = \begin{bmatrix} s_x & 0 \\\ 0 & s_y \end{bmatrix}$$

This stretches the x-axis by factor $s_x$ and the y-axis by factor $s_y$.

**Example:** $\begin{bmatrix} 2 & 0 \\\ 0 & 3 \end{bmatrix}$ doubles x-coordinates and triples y-coordinates. A unit square becomes a $2 \times 3$ rectangle.

**Uniform Scaling:** If $s_x = s_y = k$, everything scales uniformly by factor $k$. Shapes grow or shrink but maintain their proportions.

### 6.3 Reflection Matrices

Reflections flip space across a line (in 2D) or plane (in 3D).

**Reflection about the y-axis:**

$$\begin{bmatrix} -1 & 0 \\\ 0 & 1 \end{bmatrix}$$

The x-coordinate flips sign; y stays the same. Points on the right move to the left and vice versa.

**Reflection about the x-axis:**

$$\begin{bmatrix} 1 & 0 \\\ 0 & -1 \end{bmatrix}$$

**Reflection about the line y = x:**

$$\begin{bmatrix} 0 & 1 \\\ 1 & 0 \end{bmatrix}$$

This swaps x and y coordinates. The basis vector $\hat{i}$ lands at $(0, 1) = \hat{j}$, and $\hat{j}$ lands at $(1, 0) = \hat{i}$.

**Inversion through the origin:**

$$\begin{bmatrix} -1 & 0 \\\ 0 & -1 \end{bmatrix}$$

Every point maps to its opposite. This is equivalent to a 180° rotation.

### 6.4 Rotation Matrices

A rotation by angle $\theta$ counterclockwise about the origin is:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\\ \sin\theta & \cos\theta \end{bmatrix}$$

**Derivation:** Where does $\hat{i} = (1, 0)^T$ go when rotated by angle $\theta$? It lands at $(\cos\theta, \sin\theta)^T$—that's the definition of cosine and sine.

Where does $\hat{j} = (0, 1)^T$ go? It was perpendicular to $\hat{i}$ (90° ahead), and after rotating everything by $\theta$, it's at angle $90° + \theta$. So it lands at $(\cos(90°+\theta), \sin(90°+\theta))^T = (-\sin\theta, \cos\theta)^T$.

These two columns give us the rotation matrix.

**Example:** Rotation by 90° counterclockwise:

$$R(90°) = \begin{bmatrix} 0 & -1 \\\ 1 & 0 \end{bmatrix}$$

The point $(1, 0)$ maps to $(0, 1)$; the point $(0, 1)$ maps to $(-1, 0)$. Everything rotates a quarter turn.

### 6.5 Shear Matrices

Shear transformations "tilt" space while keeping one axis fixed.

**Horizontal shear:**

$$\begin{bmatrix} 1 & k \\\ 0 & 1 \end{bmatrix}$$

This keeps the x-axis fixed ($\hat{i} \to \hat{i}$) but tilts the y-axis ($\hat{j} \to (k, 1)^T$). Vertical lines become slanted.

**Vertical shear:**

$$\begin{bmatrix} 1 & 0 \\\ k & 1 \end{bmatrix}$$

This keeps the y-axis fixed but tilts the x-axis.

Shears are useful for understanding how transformations can change angles while preserving areas (if $k$ is finite).

---

## Chapter 7: Matrix Composition and Multiplication

### 7.1 Composing Transformations

When you apply one transformation followed by another, the combined effect is also a linear transformation. If $A$ is applied first and then $B$, the composition is $BA$ (note the order!).

**Why the Reversed Order?** We write $B(A\vec{v})$ for "apply $A$ to $\vec{v}$, then apply $B$ to the result." Mathematically, this equals $(BA)\vec{v}$. The rightmost matrix acts first.

**Example:** Rotate by 90° then reflect across the y-axis.

- Rotation: $R = \begin{bmatrix} 0 & -1 \\\ 1 & 0 \end{bmatrix}$
- Reflection: $F = \begin{bmatrix} -1 & 0 \\\ 0 & 1 \end{bmatrix}$
- Combined: $FR = \begin{bmatrix} -1 & 0 \\\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 & -1 \\\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\\ 1 & 0 \end{bmatrix}$

The composition turns out to be reflection across the line $y = x$.

### 7.2 Matrix Multiplication

Matrix multiplication is defined so that $(AB)\vec{v} = A(B\vec{v})$—it implements composition of transformations.

**Formula:**

$$\begin{bmatrix} a & b \\\ c & d \end{bmatrix}\begin{bmatrix} e & f \\\ g & h \end{bmatrix} = \begin{bmatrix} ae+bg & af+bh \\\ ce+dg & cf+dh \end{bmatrix}$$

Each entry $(i, j)$ of the product is the dot product of row $i$ of the first matrix with column $j$ of the second matrix.

**Properties:**
- **Associative:** $(AB)C = A(BC)$
- **NOT Commutative:** $AB \neq BA$ in general. Rotation then reflection is different from reflection then rotation.
- **Distributive:** $A(B + C) = AB + AC$

**Dimension Requirements:** To multiply $A$ (size $m \times n$) by $B$ (size $n \times p$), the inner dimensions must match. The result is $m \times p$.

### 7.3 Geometric Interpretation of Multiplication

Think of $AB$ as "where do the columns of $B$ land under transformation $A$?"

Column $j$ of $AB$ is $A$ times column $j$ of $B$. This is because when we apply $AB$ to the basis vector $\hat{e}_j$, we get the $j$-th column of $AB$, which is $A$ applied to the $j$-th column of $B$.

---

## Chapter 8: Solving Linear Systems

### 8.1 The Matrix Equation

A system of linear equations can be written as $A\vec{x} = \vec{b}$:

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} \\\ a_{21} & a_{22} & a_{23} \\\ a_{31} & a_{32} & a_{33} \end{bmatrix}\begin{bmatrix} x_1 \\\ x_2 \\\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\\ b_2 \\\ b_3 \end{bmatrix}$$

We're looking for the vector $\vec{x}$ that, when transformed by $A$, gives $\vec{b}$.

**Geometric View:** We're asking: what linear combination of the columns of $A$ produces $\vec{b}$? The coefficients of that combination are the entries of $\vec{x}$.

### 8.2 The Inverse Matrix

If $A$ has an inverse $A^{-1}$, meaning $A^{-1}A = AA^{-1} = I$, then the solution is simply:

$$\vec{x} = A^{-1}\vec{b}$$

We "undo" the transformation $A$ by applying $A^{-1}$.

**When Does the Inverse Exist?** A square matrix has an inverse if and only if its determinant is nonzero. Geometrically, this means the transformation doesn't collapse space to a lower dimension.

### 8.3 Gaussian Elimination

Even when we know the inverse exists, computing it directly for large matrices is inefficient. Gaussian elimination is a more practical method.

**The Process:**
1. Write the augmented matrix $[A | \vec{b}]$
2. Use row operations to reduce $A$ to upper triangular form (zeros below the diagonal)
3. Back-substitute to find the solution

**Row Operations (preserve solutions):**
- Swap two rows
- Multiply a row by a nonzero constant
- Add a multiple of one row to another

**Example:** Solve the system:
- $x + y + 3z = 15$
- $x + 2y + 4z = 21$
- $x + y + 2z = 13$

Augmented matrix:

$$\begin{bmatrix} 1 & 1 & 3 & | & 15 \\\ 1 & 2 & 4 & | & 21 \\\ 1 & 1 & 2 & | & 13 \end{bmatrix}$$

Subtract row 1 from rows 2 and 3:

$$\begin{bmatrix} 1 & 1 & 3 & | & 15 \\\ 0 & 1 & 1 & | & 6 \\\ 0 & 0 & -1 & | & -2 \end{bmatrix}$$

Now back-substitute: From row 3, $-z = -2$, so $z = 2$. From row 2, $y + z = 6$, so $y = 4$. From row 1, $x + y + 3z = 15$, so $x = 15 - 4 - 6 = 5$.

Solution: $(x, y, z) = (5, 4, 2)$.

### 8.4 The 2×2 Inverse Formula

For $2 \times 2$ matrices, there's a direct formula:

$$A = \begin{bmatrix} a & b \\\ c & d \end{bmatrix} \implies A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\\ -c & a \end{bmatrix}$$

The quantity $ad - bc$ is the determinant. If it's zero, no inverse exists.

**Memory Aid:** Swap the diagonal entries, negate the off-diagonal entries, divide by the determinant.

---

## Chapter 9: Determinants

### 9.1 Definition and Computation

The determinant is a single number associated with a square matrix that captures essential information about the transformation.

**For 2×2:**

$$\det\begin{bmatrix} a & b \\\ c & d \end{bmatrix} = ad - bc$$

**For 3×3 (cofactor expansion):**

$$\det\begin{bmatrix} a & b & c \\\ d & e & f \\\ g & h & i \end{bmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

### 9.2 Geometric Interpretation

The determinant measures how the transformation scales areas (2D) or volumes (3D).

**Magnitude:** $|\det(A)|$ is the factor by which areas get multiplied. If $\det(A) = 2$, a unit square transforms into a shape with area 2.

**Sign:** The sign indicates orientation. If $\det(A) > 0$, orientation is preserved (clockwise stays clockwise). If $\det(A) < 0$, orientation is reversed (clockwise becomes counterclockwise).

**Zero Determinant:** If $\det(A) = 0$, the transformation collapses space to a lower dimension—a 2D plane squashed to a line, or a 3D volume squashed to a plane or line. No inverse exists because information is lost.

### 9.3 Properties

- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^{-1}) = 1/\det(A)$
- $\det(A^T) = \det(A)$
- Swapping two rows negates the determinant
- Multiplying a row by $k$ multiplies the determinant by $k$
- Adding a multiple of one row to another doesn't change the determinant

### 9.4 Determinant and Invertibility

$$\det(A) \neq 0 \iff A \text{ is invertible}$$

This is one of the most important facts in linear algebra. A nonzero determinant means the transformation preserves dimensionality, so it can be reversed.

---

## Chapter 10: Changing Basis with Matrices

### 10.1 The Change of Basis Matrix

Suppose you have a vector described in one coordinate system and want to express it in another.

Let $B$ be a matrix whose columns are the new basis vectors expressed in the original coordinates:

$$B = \begin{bmatrix} | & | \\\ \vec{b}_1 & \vec{b}_2 \\\ | & | \end{bmatrix}$$

Then:
- $B\vec{v}_{\text{new}} = \vec{v}_{\text{original}}$ — converts from new basis coordinates to original
- $B^{-1}\vec{v}_{\text{original}} = \vec{v}_{\text{new}}$ — converts from original to new basis

### 10.2 Transformations in Different Bases

Suppose transformation $T$ is defined in the standard basis. To apply it to vectors described in basis $B$:

$$T_B = B^{-1}TB$$

**The Process:**
1. $B$ converts from basis $B$ to standard
2. $T$ applies the transformation in standard coordinates
3. $B^{-1}$ converts back to basis $B$

This "sandwich" formula $B^{-1}TB$ is called a similarity transformation. The matrices $T$ and $T_B$ represent the same geometric transformation, just described in different coordinate systems.

### 10.3 Why This Matters

Choosing the right basis can dramatically simplify a transformation. The ideal case is when $T_B$ becomes diagonal—then the transformation just scales along each basis direction independently. Finding such a basis is the goal of diagonalization (Chapter 13), and the basis vectors turn out to be eigenvectors.

---

## Chapter 11: Orthogonal Matrices and Gram-Schmidt

### 11.1 Matrix Transpose

The transpose $A^T$ flips a matrix across its diagonal:

$$(A^T)_{ij} = A_{ji}$$

Rows become columns and vice versa.

**Example:**

$$\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}^T = \begin{bmatrix} 1 & 3 \\\ 2 & 4 \end{bmatrix}$$

**Properties:**
- $(AB)^T = B^T A^T$
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$

### 11.2 Orthonormal Bases

An orthonormal basis consists of vectors that are:
1. **Orthogonal:** Every pair is perpendicular ($\vec{e}_i \cdot \vec{e}_j = 0$ for $i \neq j$)
2. **Normal:** Each has unit length ($|\vec{e}_i| = 1$)

The standard basis $\{\hat{i}, \hat{j}, \hat{k}\}$ is orthonormal. But there are infinitely many other orthonormal bases—any rotation of the standard basis, for instance.

**Why Orthonormal Bases are Special:**
- Projections become simple dot products
- No matrix inversion needed for coordinate changes
- The formulas for changing basis simplify dramatically

### 11.3 Orthogonal Matrices

A matrix $Q$ is orthogonal if its columns form an orthonormal set.

**Key Property:**

$$Q^T Q = Q Q^T = I$$

Therefore:

$$Q^{-1} = Q^T$$

The inverse of an orthogonal matrix is just its transpose—no computation needed!

**Geometric Meaning:** Orthogonal matrices represent rotations and reflections—transformations that preserve lengths and angles. They're the "rigid motions" of linear algebra.

**Properties:**
- $|\det(Q)| = 1$ (areas/volumes preserved)
- $|Q\vec{v}| = |\vec{v}|$ (lengths preserved)
- $(Q\vec{u}) \cdot (Q\vec{v}) = \vec{u} \cdot \vec{v}$ (dot products preserved)

### 11.4 The Gram-Schmidt Process

Given any set of linearly independent vectors, Gram-Schmidt constructs an orthonormal basis spanning the same space.

**Algorithm:**

Given $\{v_1, v_2, \ldots, v_n\}$:

**Step 1:** Normalize $v_1$:
$$e_1 = \frac{v_1}{|v_1|}$$

**Step 2:** Remove from $v_2$ its component along $e_1$, then normalize:
$$u_2 = v_2 - (v_2 \cdot e_1)e_1$$
$$e_2 = \frac{u_2}{|u_2|}$$

**Step 3:** Remove from $v_3$ its components along $e_1$ and $e_2$, then normalize:
$$u_3 = v_3 - (v_3 \cdot e_1)e_1 - (v_3 \cdot e_2)e_2$$
$$e_3 = \frac{u_3}{|u_3|}$$

**Continue** for remaining vectors.

Each step subtracts out all the components along previously computed basis vectors, leaving only the "new" direction, which is then normalized.

**Example:** Orthonormalize $\{(1, 1), (1, 0)\}$.

$e_1 = (1, 1)/\sqrt{2} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$

$u_2 = (1, 0) - [(1, 0) \cdot e_1]e_1 = (1, 0) - \frac{1}{\sqrt{2}} \cdot (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}) = (1, 0) - (\frac{1}{2}, \frac{1}{2}) = (\frac{1}{2}, -\frac{1}{2})$

$e_2 = (\frac{1}{2}, -\frac{1}{2})/|(\frac{1}{2}, -\frac{1}{2})| = (\frac{1}{2}, -\frac{1}{2})/\frac{1}{\sqrt{2}} = (\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}})$

Check: $e_1 \cdot e_2 = \frac{1}{2} - \frac{1}{2} = 0$. ✓

---

## Chapter 12: Eigenvalues and Eigenvectors

### 12.1 The Concept

Most vectors change direction when a matrix transformation is applied. But some special vectors only get stretched or compressed—they stay on their original line. These are eigenvectors.

**Definition:** A nonzero vector $\vec{x}$ is an eigenvector of matrix $A$ if:

$$A\vec{x} = \lambda\vec{x}$$

The scalar $\lambda$ is the corresponding eigenvalue.

**Geometric Interpretation:** When $A$ is applied to an eigenvector, the result is just a scaled version of the same vector. The eigenvector "points along" a direction that the transformation finds natural.

### 12.2 Why Eigenvectors Matter

Eigenvectors reveal the fundamental structure of a transformation:

**Diagonal Scaling:** In the basis of eigenvectors (if one exists), the transformation is just scaling along each axis. This is the simplest possible form.

**Stability Analysis:** In dynamical systems, eigenvalues determine stability. If all eigenvalues have magnitude less than 1, the system converges to equilibrium.

**Principal Components:** In PCA, eigenvectors of the covariance matrix are the principal components—the directions of maximum variance.

**PageRank:** Google's original algorithm finds the dominant eigenvector of the web's link matrix.

### 12.3 Finding Eigenvalues

Starting from $A\vec{x} = \lambda\vec{x}$, rearrange:

$$(A - \lambda I)\vec{x} = \vec{0}$$

For a nonzero solution $\vec{x}$ to exist, the matrix $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0$$

This is the characteristic equation. For an $n \times n$ matrix, it's a polynomial of degree $n$ in $\lambda$.

**Example:** Find eigenvalues of $A = \begin{bmatrix} 4 & 2 \\\ 1 & 3 \end{bmatrix}$.

$$\det\begin{bmatrix} 4-\lambda & 2 \\\ 1 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0$$

Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$.

### 12.4 Finding Eigenvectors

For each eigenvalue, solve $(A - \lambda I)\vec{x} = \vec{0}$.

**For $\lambda_1 = 5$:**

$$\begin{bmatrix} -1 & 2 \\\ 1 & -2 \end{bmatrix}\begin{bmatrix} x \\\ y \end{bmatrix} = \begin{bmatrix} 0 \\\ 0 \end{bmatrix}$$

Both rows give $-x + 2y = 0$, so $x = 2y$. Eigenvector: $\vec{v}_1 = t\begin{bmatrix} 2 \\\ 1 \end{bmatrix}$ for any $t \neq 0$.

**For $\lambda_2 = 2$:**

$$\begin{bmatrix} 2 & 2 \\\ 1 & 1 \end{bmatrix}\begin{bmatrix} x \\\ y \end{bmatrix} = \begin{bmatrix} 0 \\\ 0 \end{bmatrix}$$

Both rows give $x + y = 0$. Eigenvector: $\vec{v}_2 = t\begin{bmatrix} 1 \\\ -1 \end{bmatrix}$.

### 12.5 Special Cases

**Repeated Eigenvalues:** The characteristic polynomial may have repeated roots. The corresponding eigenvectors may or may not span the expected dimension.

**Complex Eigenvalues:** Rotation matrices have no real eigenvectors—no real vector stays on its line under rotation (except 180°). The eigenvalues are complex numbers. For $R(\theta)$, they are $e^{\pm i\theta}$.

**Symmetric Matrices:** Real symmetric matrices always have real eigenvalues and orthogonal eigenvectors. This is why they're so important in data science (covariance matrices are symmetric).

---

## Chapter 13: Diagonalization

### 13.1 Diagonal Matrices

Diagonal matrices are trivial to work with:

$$D = \begin{bmatrix} d_1 & 0 & 0 \\\ 0 & d_2 & 0 \\\ 0 & 0 & d_3 \end{bmatrix}$$

Powers are easy:

$$D^n = \begin{bmatrix} d_1^n & 0 & 0 \\\ 0 & d_2^n & 0 \\\ 0 & 0 & d_3^n \end{bmatrix}$$

The inverse (if all $d_i \neq 0$):

$$D^{-1} = \begin{bmatrix} 1/d_1 & 0 & 0 \\\ 0 & 1/d_2 & 0 \\\ 0 & 0 & 1/d_3 \end{bmatrix}$$

### 13.2 The Diagonalization Theorem

If an $n \times n$ matrix $A$ has $n$ linearly independent eigenvectors, it can be diagonalized:

$$A = PDP^{-1}$$

where:
- $P$ is the matrix whose columns are eigenvectors of $A$
- $D$ is the diagonal matrix of corresponding eigenvalues

**Interpretation:** In the basis of eigenvectors, the transformation is just diagonal scaling. $P$ converts to that basis, $D$ applies the simple scaling, and $P^{-1}$ converts back.

### 13.3 Computing Powers

The diagonalization makes computing powers trivial:

$$A^n = PD^nP^{-1}$$

**Proof:** $A^2 = (PDP^{-1})(PDP^{-1}) = PD(P^{-1}P)DP^{-1} = PD^2P^{-1}$. By induction, $A^n = PD^nP^{-1}$.

**Why This Matters:** Computing $A^{1000}$ directly requires 999 matrix multiplications. With diagonalization, you compute $D^{1000}$ (just raise diagonal entries to the 1000th power), then do two matrix multiplications. For large matrices and large powers, this saves enormous computation.

### 13.4 Example

Diagonalize $A = \begin{bmatrix} 4 & 2 \\\ 1 & 3 \end{bmatrix}$.

We found eigenvalues $\lambda_1 = 5$, $\lambda_2 = 2$ and eigenvectors $\vec{v}_1 = (2, 1)^T$, $\vec{v}_2 = (1, -1)^T$.

$$P = \begin{bmatrix} 2 & 1 \\\ 1 & -1 \end{bmatrix}, \quad D = \begin{bmatrix} 5 & 0 \\\ 0 & 2 \end{bmatrix}$$

$$P^{-1} = \frac{1}{-3}\begin{bmatrix} -1 & -1 \\\ -1 & 2 \end{bmatrix} = \begin{bmatrix} 1/3 & 1/3 \\\ 1/3 & -2/3 \end{bmatrix}$$

Verify: $PDP^{-1} = A$. ✓

Now $A^{10}$ is easy:

$$A^{10} = P\begin{bmatrix} 5^{10} & 0 \\\ 0 & 2^{10} \end{bmatrix}P^{-1} = P\begin{bmatrix} 9765625 & 0 \\\ 0 & 1024 \end{bmatrix}P^{-1}$$

---

## Chapter 14: PageRank Algorithm

### 14.1 The Problem

In 1998, Larry Page and Sergey Brin faced a problem: how do you rank billions of web pages by importance? Previous search engines ranked pages by keyword matching, but this was easily manipulated.

Their insight: a page is important if important pages link to it. This recursive definition seems circular, but linear algebra makes it precise.

### 14.2 The Link Matrix

Construct a matrix $L$ where $L_{ij}$ represents the probability of moving from page $j$ to page $i$ by following a random link.

**Construction:**
1. If page $j$ has $n_j$ outgoing links, and one goes to page $i$, then $L_{ij} = 1/n_j$
2. If page $j$ doesn't link to page $i$, then $L_{ij} = 0$

Each column of $L$ sums to 1 (it's a probability distribution).

**Example:** Four pages A, B, C, D with links:
- A links to B, C, D
- B links to A, D
- C links to A
- D links to B, C

$$L = \begin{bmatrix} 0 & 1/2 & 1 & 0 \\\ 1/3 & 0 & 0 & 1/2 \\\ 1/3 & 0 & 0 & 1/2 \\\ 1/3 & 1/2 & 0 & 0 \end{bmatrix}$$

### 14.3 Rank as an Eigenvector

Let $\vec{r}$ be the vector of page ranks (where $r_i$ is the importance of page $i$). The rank of page $i$ should be proportional to the sum of ranks of pages linking to it, weighted by link probability:

$$r_i = \sum_j L_{ij} r_j$$

In matrix form:

$$\vec{r} = L\vec{r}$$

This is an eigenvector equation with $\lambda = 1$! The PageRank vector is the eigenvector of $L$ corresponding to eigenvalue 1.

### 14.4 The Power Method

Finding eigenvectors for billion-page matrices requires iterative methods.

**Power Iteration:**
1. Start with initial guess $\vec{r}_0 = (1/n, 1/n, \ldots, 1/n)^T$ (uniform distribution)
2. Iterate: $\vec{r}_{k+1} = L\vec{r}_k$
3. Stop when $\vec{r}$ converges

Under certain conditions, this converges to the dominant eigenvector—the one with eigenvalue 1.

### 14.5 Damping Factor

Real web graphs have problems: some pages have no outgoing links (dead ends), and some clusters of pages only link to each other (spider traps).

**Solution:** Introduce a damping factor $d \approx 0.85$. At each step, with probability $d$ you follow a random link, and with probability $1-d$ you jump to a random page:

$$\vec{r}_{k+1} = d \cdot L\vec{r}_k + \frac{1-d}{n}\vec{1}$$

This guarantees convergence and handles edge cases.

### 14.6 Scale and Efficiency

The power method is beautifully suited to sparse matrices. Most web pages link to only a few others, so $L$ is mostly zeros. Each iteration just requires multiplying by a sparse matrix—much faster than dense matrix operations.

PageRank demonstrated that eigenvector analysis could scale to billions of dimensions, revolutionizing both search engines and our understanding of network analysis.

---

## Summary: Key Formulas

### Vector Operations

The dot product: $\vec{r} \cdot \vec{s} = \sum_i r_is_i = |\vec{r}||\vec{s}|\cos\theta$

Vector length: $|\vec{r}| = \sqrt{\vec{r} \cdot \vec{r}}$

Vector projection: $\text{proj}_{\vec{r}}\vec{s} = \frac{\vec{r} \cdot \vec{s}}{|\vec{r}|^2}\vec{r}$

### Matrix Operations

Matrix multiplication $(AB)_{ij} = \sum_k A_{ik}B_{kj}$

2×2 determinant: $\det\begin{bmatrix} a & b \\\ c & d \end{bmatrix} = ad - bc$

2×2 inverse: $\begin{bmatrix} a & b \\\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\\ -c & a \end{bmatrix}$

### Eigenproblems

Eigenvalue equation: $A\vec{x} = \lambda\vec{x}$

Characteristic equation: $\det(A - \lambda I) = 0$

Diagonalization: $A = PDP^{-1}$

Matrix powers: $A^n = PD^nP^{-1}$

---

## Key Takeaways

Linear algebra is fundamentally about structure and transformation. Vectors are the objects, matrices are the transformations, and everything else builds from there.

The dot product connects geometry (angles, lengths, projections) to algebra (sums of products). Orthogonality—when the dot product is zero—simplifies almost everything.

Matrices are linear transformations. Their columns show where basis vectors go. This geometric view makes abstract algebra concrete.

The determinant measures volume scaling and indicates invertibility. Zero determinant means information loss.

Changing basis lets you choose the most convenient coordinate system. The right choice can transform a complex problem into a trivial one.

Eigenvectors are the natural directions of a transformation—the directions it treats simply. Eigenvalues tell you the scaling along each direction.

Diagonalization expresses a matrix in its simplest form. Many practical applications (computing powers, solving differential equations, understanding stability) become trivial once a matrix is diagonalized.

These ideas extend to infinite dimensions (functional analysis), curved spaces (differential geometry), and quantum mechanics. The linear algebra you've learned here is the foundation for vast areas of mathematics, physics, and computer science.
