# Intermediate Deep Learning with PyTorch - Mathematical Deep Dive

## Table of Contents
1. [PyTorch and Object-Oriented Programming](#1-pytorch-and-object-oriented-programming)
2. [Optimizers, Training, and Evaluation](#2-optimizers-training-and-evaluation)
3. [Vanishing and Exploding Gradients](#3-vanishing-and-exploding-gradients)
4. [Handling Images with PyTorch](#4-handling-images-with-pytorch)
5. [Convolutional Neural Networks (Deep Dive)](#5-convolutional-neural-networks-deep-dive)
6. [Training and Evaluating Image Classifiers](#6-training-and-evaluating-image-classifiers)
7. [Handling Sequences with PyTorch](#7-handling-sequences-with-pytorch)
8. [Recurrent Neural Networks (Deep Dive)](#8-recurrent-neural-networks-deep-dive)
9. [LSTM Networks (Deep Dive)](#9-lstm-networks-deep-dive)
10. [GRU Networks (Deep Dive)](#10-gru-networks-deep-dive)
11. [Training and Evaluating RNNs](#11-training-and-evaluating-rnns)
12. [Multi-Input and Multi-Output Models](#12-multi-input-and-multi-output-models)

---

## 1. PyTorch and Object-Oriented Programming

### 1.1 Key Definitions

**Object-Oriented Programming (OOP)** is a programming paradigm centered around "objects" that encapsulate both data (attributes) and behavior (methods). In PyTorch, OOP enables flexible definition of custom Datasets and neural network architectures.

**Class**: A blueprint defining the structure and behavior of objects. When instantiated, a class produces an object with the defined attributes and methods.

**`__init__` method**: The constructor method, automatically invoked upon object creation. It initializes instance attributes and sets up the object's initial state.

**`self`**: A reference to the current instance, allowing access to instance-specific attributes and methods within class definitions.

### 1.2 PyTorch Dataset Structure

A PyTorch Dataset must inherit from `torch.utils.data.Dataset` and implement three essential methods:

```python
import pandas as pd
from torch.utils.data import Dataset

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()  # Initialize parent class behavior
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()
    
    def __len__(self):
        # Returns N, the total number of samples
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Returns tuple (x_i, y_i) for sample index i
        features = self.data[idx, :-1]  # x_i ∈ ℝ^d
        label = self.data[idx, -1]       # y_i ∈ {0, 1}
        return features, label
```

### 1.3 DataLoader and Batching

The DataLoader wraps a Dataset to provide mini-batch iteration. Given a dataset of N samples, it produces batches of size B:

**Number of batches per epoch**: ⌈N/B⌉

```python
from torch.utils.data import DataLoader

dataloader_train = DataLoader(
    dataset_train,
    batch_size=32,    # B = 32
    shuffle=True,     # Randomize order each epoch
)
```

### 1.4 Neural Network as a Class

The forward pass of a neural network with L layers can be expressed mathematically as:

**Layer l computation**:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

where:
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix for layer l
- $b^{(l)} \in \mathbb{R}^{n_l}$ is the bias vector
- $\sigma(\cdot)$ is the activation function
- $a^{(0)} = x$ (input)

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # W^(1) ∈ ℝ^{16×9}, b^(1) ∈ ℝ^16
        self.fc1 = nn.Linear(9, 16)
        # W^(2) ∈ ℝ^{8×16}, b^(2) ∈ ℝ^8
        self.fc2 = nn.Linear(16, 8)
        # W^(3) ∈ ℝ^{1×8}, b^(3) ∈ ℝ^1
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        # a^(1) = ReLU(W^(1)x + b^(1))
        x = nn.functional.relu(self.fc1(x))
        # a^(2) = ReLU(W^(2)a^(1) + b^(2))
        x = nn.functional.relu(self.fc2(x))
        # a^(3) = σ(W^(3)a^(2) + b^(3))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
```

---

## 2. Optimizers, Training, and Evaluation

### 2.1 The Optimization Problem

Neural network training is an optimization problem seeking to minimize an objective function:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)$$

where:
- $\theta$ represents all learnable parameters (weights and biases)
- $\mathcal{L}(\theta)$ is the empirical loss over the training set
- $\ell(\hat{y}, y)$ is the per-sample loss function
- $f_\theta(x)$ is the model's prediction

### 2.2 Gradient Descent

The fundamental update rule for gradient descent is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta > 0$ is the learning rate (step size).

**Stochastic Gradient Descent (SGD)** approximates the full gradient using a mini-batch B:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

### 2.3 Loss Functions

**Binary Cross-Entropy (BCE)** for binary classification ($y \in \{0, 1\}$):

$$\ell_{BCE}(\hat{y}, y) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Categorical Cross-Entropy** for multi-class classification ($y \in \{1, ..., K\}$):

$$\ell_{CE}(\hat{y}, y) = -\sum_{k=1}^{K} \mathbb{1}_{[y=k]} \log(\hat{y}_k) = -\log(\hat{y}_y)$$

where $\hat{y}_k$ is the predicted probability for class k.

**Mean Squared Error (MSE)** for regression:

$$\ell_{MSE}(\hat{y}, y) = (\hat{y} - y)^2$$

### 2.4 Optimizer Algorithms

**Stochastic Gradient Descent (SGD)**:
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ is the gradient at time t.

**SGD with Momentum**: Accumulates velocity to accelerate convergence:
$$v_t = \gamma v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

where $\gamma \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Adagrad (Adaptive Gradient)**: Adapts learning rate per parameter based on historical gradients:
$$G_t = G_{t-1} + g_t \odot g_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

where $\odot$ denotes element-wise multiplication and $\epsilon \approx 10^{-8}$ prevents division by zero.

**Problem with Adagrad**: $G_t$ grows monotonically, causing learning rate to decay to zero.

**RMSprop (Root Mean Square Propagation)**: Uses exponential moving average instead:
$$E[g^2]_t = \rho \cdot E[g^2]_{t-1} + (1-\rho) \cdot g_t \odot g_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t$$

where $\rho$ is the decay rate (typically 0.9).

**Adam (Adaptive Moment Estimation)**: Combines momentum with RMSprop:

First moment estimate (momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

Second moment estimate (RMSprop-like):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t$$

Bias correction (crucial for early iterations):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update rule:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

```python
import torch.optim as optim

# Different optimizers
optimizer_sgd = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### 2.5 Backpropagation

Backpropagation computes gradients efficiently using the chain rule. For a loss $\mathcal{L}$ and layer output $a^{(l)} = \sigma(z^{(l)})$ where $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$:

**Error signal at layer l**:
$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \odot \sigma'(z^{(l)})$$

**Recursive relationship**:
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

**Gradients for parameters**:
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

### 2.6 Training Loop Implementation

```python
net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader_train:
        # Forward pass: compute ŷ = f_θ(x)
        y_pred = net(x_batch)
        
        # Compute loss: L = ℓ(ŷ, y)
        loss = criterion(y_pred, y_batch.view(-1, 1))
        
        # Reset gradients: ∇θ ← 0
        optimizer.zero_grad()
        
        # Backward pass: compute ∇θL via backpropagation
        loss.backward()
        
        # Parameter update: θ ← θ - η·∇θL (via optimizer algorithm)
        optimizer.step()
```

---

## 3. Vanishing and Exploding Gradients

### 3.1 The Gradient Flow Problem

Consider a deep network with L layers. During backpropagation, the gradient flows backward through each layer. For layer l:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \cdot \prod_{k=l+1}^{L} \frac{\partial a^{(k)}}{\partial a^{(k-1)}} \cdot \frac{\partial a^{(l)}}{\partial W^{(l)}}$$

The product term $\prod_{k=l+1}^{L} \frac{\partial a^{(k)}}{\partial a^{(k-1)}}$ involves multiplication of many Jacobian matrices.

**Vanishing Gradients**: If $\|\frac{\partial a^{(k)}}{\partial a^{(k-1)}}\| < 1$ consistently, the product shrinks exponentially:
$$\left\|\prod_{k=l+1}^{L}\right\| \approx \gamma^{L-l} \to 0 \text{ as } L-l \to \infty$$

**Exploding Gradients**: If $\|\frac{\partial a^{(k)}}{\partial a^{(k-1)}}\| > 1$ consistently:
$$\left\|\prod_{k=l+1}^{L}\right\| \to \infty$$

### 3.2 Activation Functions and Their Derivatives

**Sigmoid**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Maximum derivative: $\sigma'(0) = 0.25$. Since $\sigma'(z) \leq 0.25$ always, gradients shrink by at least 75% at each layer, causing vanishing gradients in deep networks.

**Tanh**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$\tanh'(z) = 1 - \tanh^2(z)$$

Maximum derivative: $\tanh'(0) = 1$, but $\tanh'(z) < 1$ for $z \neq 0$, still prone to vanishing gradients.

**ReLU (Rectified Linear Unit)**:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

Advantage: Gradient is exactly 1 for positive inputs (no vanishing). Problem: **Dying ReLU** — neurons with negative input have zero gradient and stop learning.

**Leaky ReLU**:
$$\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

where $\alpha$ is a small constant (e.g., 0.01). Allows small gradient for negative inputs.

**ELU (Exponential Linear Unit)**:
$$\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$$
$$\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z = \text{ELU}(z) + \alpha & \text{if } z \leq 0 \end{cases}$$

Advantages: Non-zero gradient for negative inputs; outputs have mean closer to zero (reduces bias shift).

### 3.3 Weight Initialization

**Goal**: Initialize weights such that the variance of activations and gradients is preserved across layers.

For a layer computing $z = Wx$ where $x$ has variance $\text{Var}(x)$ and weights $W_{ij}$ are i.i.d.:

$$\text{Var}(z_j) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)$$

To maintain $\text{Var}(z) = \text{Var}(x)$, we need $\text{Var}(W) = \frac{1}{n_{in}}$.

**Xavier/Glorot Initialization** (for tanh, sigmoid):
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

or equivalently $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$

**He/Kaiming Initialization** (for ReLU family):

ReLU zeros out ~50% of activations, so we need to compensate:
$$\text{Var}(W) = \frac{2}{n_{in}}$$
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$$

```python
import torch.nn.init as init

# He initialization for ReLU layers
init.kaiming_uniform_(layer.weight, nonlinearity='relu')

# For sigmoid output layer
init.kaiming_uniform_(layer.weight, nonlinearity='sigmoid')
```

### 3.4 Batch Normalization

**Batch Normalization** normalizes layer inputs across the mini-batch, then applies learnable scale and shift.

For a mini-batch $B = \{x_1, ..., x_m\}$:

**Step 1 - Compute batch statistics**:
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

**Step 2 - Normalize**:
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Step 3 - Scale and shift** (learnable parameters $\gamma$, $\beta$):
$$y_i = \gamma \hat{x}_i + \beta = \text{BN}_{\gamma,\beta}(x_i)$$

**Why it works**:
1. Reduces internal covariate shift (distribution of inputs to each layer stays stable)
2. Allows higher learning rates without divergence
3. Acts as regularization (each sample is normalized with batch statistics, adding noise)
4. Model learns optimal input distribution via $\gamma$ and $\beta$

**During inference**: Use running averages of $\mu$ and $\sigma^2$ computed during training.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)  # BatchNorm after linear, before activation
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        
        # He initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')
    
    def forward(self, x):
        # Linear → BatchNorm → Activation
        x = nn.functional.elu(self.bn1(self.fc1(x)))
        x = nn.functional.elu(self.bn2(self.fc2(x)))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
```

---

## 4. Handling Images with PyTorch

### 4.1 Digital Image Representation

A digital image is a discrete 2D signal represented as a matrix of pixel values.

**Grayscale image**: $I \in \mathbb{R}^{H \times W}$ where $I(i,j) \in [0, 255]$ represents intensity at pixel $(i,j)$.

**Color (RGB) image**: $I \in \mathbb{R}^{H \times W \times 3}$ where each pixel has three channels:
$$I(i,j) = [R(i,j), G(i,j), B(i,j)]^T$$

**PyTorch tensor format**: $(N, C, H, W)$ where N is batch size, C is channels (1 for grayscale, 3 for RGB).

### 4.2 Image Loading and Transforms

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),         # PIL Image → Tensor, scales [0,255] → [0,1]
    transforms.Resize((128, 128)), # Resize to H×W = 128×128
])

dataset_train = ImageFolder("clouds_train", transform=train_transforms)
```

### 4.3 Data Augmentation

Data augmentation applies random transformations to increase effective dataset size and improve generalization.

**Mathematical perspective**: If we have N training samples and apply K random augmentations per sample, we effectively have up to N×K training examples, though correlated.

**Common augmentations**:

- **Horizontal flip**: $I'(i,j) = I(i, W-j)$
- **Rotation by angle θ**: Apply rotation matrix $R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$
- **Random crop**: Extract random H'×W' subregion
- **Color jittering**: Adjust brightness, contrast, saturation

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 50% probability
    transforms.RandomRotation(45),            # Random angle ∈ [-45°, 45°]
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])
```

---

## 5. Convolutional Neural Networks (Deep Dive)

### 5.1 Motivation: Why Convolution?

For a 256×256 RGB image, a fully connected layer with 1000 neurons requires:
$$256 \times 256 \times 3 \times 1000 = 196,608,000 \text{ parameters}$$

**Problems with fully connected layers for images**:
1. Massive parameter count → overfitting, slow training
2. No spatial invariance — same object at different positions requires relearning
3. Ignores local spatial structure of images

**Convolutional layers address these via**:
1. **Local connectivity**: Each neuron connects only to a local region
2. **Parameter sharing**: Same filter applied across all spatial positions
3. **Translation equivariance**: Shifting input shifts output correspondingly

### 5.2 The Convolution Operation

**Discrete 2D convolution** of input $I$ with kernel $K$:

$$(I * K)(i,j) = \sum_{m}\sum_{n} I(i+m, j+n) \cdot K(m,n)$$

For a kernel of size $k \times k$ centered at position $(i,j)$:

$$(I * K)(i,j) = \sum_{m=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor}\sum_{n=-\lfloor k/2 \rfloor}^{\lfloor k/2 \rfloor} I(i+m, j+n) \cdot K(m,n)$$

**Example**: 3×3 kernel on 5×5 input

Input patch at position (1,1):
$$\begin{pmatrix} I_{0,0} & I_{0,1} & I_{0,2} \\ I_{1,0} & I_{1,1} & I_{1,2} \\ I_{2,0} & I_{2,1} & I_{2,2} \end{pmatrix}$$

Kernel:
$$K = \begin{pmatrix} K_{0,0} & K_{0,1} & K_{0,2} \\ K_{1,0} & K_{1,1} & K_{1,2} \\ K_{2,0} & K_{2,1} & K_{2,2} \end{pmatrix}$$

Output at (1,1):
$$(I * K)_{1,1} = \sum_{m=0}^{2}\sum_{n=0}^{2} I_{m,n} \cdot K_{m,n}$$

### 5.3 Convolution Output Dimensions

For input of size $H_{in} \times W_{in}$, kernel size $k$, padding $p$, and stride $s$:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

**Common configurations**:
- "Same" convolution: $p = \lfloor k/2 \rfloor$, $s=1$ → $H_{out} = H_{in}$
- "Valid" convolution: $p = 0$ → $H_{out} = H_{in} - k + 1$

### 5.4 Multi-Channel Convolution

For input with $C_{in}$ channels and producing $C_{out}$ feature maps:

**Input**: $X \in \mathbb{R}^{C_{in} \times H \times W}$

**Filters**: $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$

**Bias**: $b \in \mathbb{R}^{C_{out}}$

**Output**: $Y \in \mathbb{R}^{C_{out} \times H' \times W'}$

For output channel $c_{out}$:
$$Y_{c_{out}}(i,j) = b_{c_{out}} + \sum_{c_{in}=1}^{C_{in}} (X_{c_{in}} * W_{c_{out}, c_{in}})(i,j)$$

**Total parameters**: $C_{out} \times C_{in} \times k \times k + C_{out}$

### 5.5 Padding and Its Importance

**Zero-padding** adds zeros around the input border before convolution.

**Mathematical effect**: Extends input from $H \times W$ to $(H + 2p) \times (W + 2p)$

**Benefits**:
1. **Preserve spatial dimensions**: With $p = \lfloor k/2 \rfloor$ and $s=1$, output size equals input size
2. **Equal treatment of border pixels**: Without padding, corner pixels are covered by kernel only once; center pixels are covered $k^2$ times
3. **Prevent information loss at edges**

### 5.6 Pooling Operations

**Max Pooling** with window size $k$ and stride $s$:

$$Y(i,j) = \max_{0 \leq m,n < k} X(i \cdot s + m, j \cdot s + n)$$

**Average Pooling**:

$$Y(i,j) = \frac{1}{k^2} \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} X(i \cdot s + m, j \cdot s + n)$$

**Output dimensions** (same formula as convolution with $p=0$):
$$H_{out} = \left\lfloor \frac{H_{in} - k}{s} \right\rfloor + 1$$

**Typical usage**: $k=2$, $s=2$ → halves spatial dimensions

**Benefits of pooling**:
1. **Dimensionality reduction**: Reduces computation and parameters in subsequent layers
2. **Translation invariance**: Small shifts in input don't change pooled output
3. **Feature selection**: Max pooling selects strongest activations

### 5.7 Receptive Field

The **receptive field** is the region of the input that affects a particular output unit.

For a single convolutional layer with kernel size $k$: receptive field = $k \times k$

For L stacked conv layers with kernel size k each:
$$\text{Receptive field} = 1 + L \times (k - 1)$$

With pooling layers (stride 2), receptive field grows exponentially.

### 5.8 Feature Hierarchy in CNNs

CNNs learn hierarchical features:
- **Early layers**: Detect low-level features (edges, textures, colors)
- **Middle layers**: Detect mid-level features (shapes, parts)
- **Deep layers**: Detect high-level features (objects, semantic concepts)

### 5.9 Complete CNN Architecture

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → (N, 32, 64, 64)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),                  # → (N, 32, 32, 32)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (N, 64, 32, 32)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),                  # → (N, 64, 16, 16)
            
            nn.Flatten(),                                 # → (N, 64×16×16)
        )
        
        # Classifier
        # Input size: 64 channels × 16 height × 16 width = 16,384
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
```

### 5.10 Backpropagation Through Convolution

For convolution $Y = X * W$, the gradients are:

**Gradient w.r.t. weights**:
$$\frac{\partial \mathcal{L}}{\partial W} = X * \frac{\partial \mathcal{L}}{\partial Y}$$

(This is a convolution of input with upstream gradient)

**Gradient w.r.t. input**:
$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} * \text{rot}_{180}(W)$$

(This is a full convolution with the rotated kernel)

---

## 6. Training and Evaluating Image Classifiers

### 6.1 Softmax and Cross-Entropy

For multi-class classification, the network outputs logits $z \in \mathbb{R}^K$, converted to probabilities via **softmax**:

$$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Properties:
- $\hat{y}_k \in (0, 1)$ for all k
- $\sum_{k=1}^{K} \hat{y}_k = 1$

**Cross-entropy loss** with one-hot encoded target $y$:

$$\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k) = -\log(\hat{y}_c)$$

where c is the true class.

**PyTorch's `nn.CrossEntropyLoss`** combines LogSoftmax and NLLLoss:
$$\mathcal{L} = -z_c + \log\left(\sum_{j=1}^{K} e^{z_j}\right)$$

This is numerically more stable than computing softmax then log.

### 6.2 Precision, Recall, and F1 Score

For binary classification:

**Confusion Matrix**:
|  | Predicted + | Predicted - |
|--|-------------|-------------|
| **Actual +** | TP | FN |
| **Actual -** | FP | TN |

**Precision** (positive predictive value):
$$\text{Precision} = \frac{TP}{TP + FP}$$

"Of all positive predictions, what fraction are correct?"

**Recall** (sensitivity, true positive rate):
$$\text{Recall} = \frac{TP}{TP + FN}$$

"Of all actual positives, what fraction did we identify?"

**F1 Score** (harmonic mean of precision and recall):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### 6.3 Multi-Class Averaging

For K classes, we have K precision/recall values. Aggregation methods:

**Micro-averaging** (global calculation):
$$\text{Precision}_{micro} = \frac{\sum_{k=1}^{K} TP_k}{\sum_{k=1}^{K} (TP_k + FP_k)}$$

**Macro-averaging** (average of per-class metrics):
$$\text{Precision}_{macro} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k$$

**Weighted-averaging**:
$$\text{Precision}_{weighted} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \text{Precision}_k$$

where $n_k$ is the number of samples in class k.

```python
from torchmetrics import Precision, Recall

# Per-class metrics
precision = Precision(task="multiclass", num_classes=7, average=None)

# Aggregated metrics
precision_macro = Precision(task="multiclass", num_classes=7, average='macro')
precision_micro = Precision(task="multiclass", num_classes=7, average='micro')
```

---

## 7. Handling Sequences with PyTorch

### 7.1 Sequential Data Definition

**Sequential data** is data where the ordering of elements carries information. Mathematically, a sequence is an ordered collection:

$$X = (x_1, x_2, ..., x_T)$$

where the index t often represents time.

**Key property**: $x_t$ may depend on previous elements $x_1, ..., x_{t-1}$.

### 7.2 Time Series Forecasting Setup

**Objective**: Given past observations $(x_1, ..., x_T)$, predict future value $x_{T+1}$.

**Windowed approach**: Use a sliding window of length L:
- Input: $(x_t, x_{t+1}, ..., x_{t+L-1})$
- Target: $x_{t+L}$

This creates N - L training examples from N data points.

```python
def create_sequences(df, seq_length):
    """
    Creates input-target pairs using sliding window.
    
    For sequence length L and data of length N:
    - Number of examples: N - L
    - Input shape: (N-L, L)
    - Target shape: (N-L,)
    """
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i:(i + seq_length), 1].values  # L consecutive values
        y = df.iloc[i + seq_length, 1]              # Next value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Example: 24 hours of 15-min data = 96 time steps
X_train, y_train = create_sequences(train_data, seq_length=96)
```

### 7.3 Temporal Train-Test Split

**Critical**: Never shuffle sequential data randomly for train-test split.

**Look-ahead bias**: If future data appears in training, the model learns spurious patterns that won't exist at inference time.

**Correct approach**: Split by time
$$\text{Train}: t \in [1, T_{split}], \quad \text{Test}: t \in (T_{split}, T]$$

---

## 8. Recurrent Neural Networks (Deep Dive)

### 8.1 Motivation for Recurrent Architectures

**Problem with feedforward networks for sequences**:
1. Fixed input size — can't handle variable-length sequences
2. No parameter sharing across time — each position learned independently
3. No memory of past inputs

**RNN solution**: Maintain a hidden state that carries information across time steps.

### 8.2 The Simple RNN (Elman Network)

At each time step t, the RNN updates its hidden state and produces an output:

**Hidden state update**:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

**Output computation**:
$$y_t = W_{hy} h_t + b_y$$

**Parameters**:
- $W_{xh} \in \mathbb{R}^{d_h \times d_x}$: Input-to-hidden weights
- $W_{hh} \in \mathbb{R}^{d_h \times d_h}$: Hidden-to-hidden weights
- $W_{hy} \in \mathbb{R}^{d_y \times d_h}$: Hidden-to-output weights
- $b_h \in \mathbb{R}^{d_h}$, $b_y \in \mathbb{R}^{d_y}$: Biases

**Initial hidden state**: $h_0 = \mathbf{0}$ (typically)

**Total parameters**: $d_h(d_x + d_h + d_y) + d_h + d_y$ (independent of sequence length!)

### 8.3 Unrolling Through Time

Unrolling visualizes the RNN as a feedforward network across T time steps:

```
x_1 ──→ [RNN] ──→ y_1
           ↓ h_1
x_2 ──→ [RNN] ──→ y_2
           ↓ h_2
x_3 ──→ [RNN] ──→ y_3
           ↓ h_3
          ...
```

Each "RNN" block shares the same parameters $(W_{xh}, W_{hh}, W_{hy})$.

### 8.4 RNN Architectures

**Sequence-to-Sequence**: Output at every time step
$$\{y_1, y_2, ..., y_T\} = \text{RNN}(x_1, x_2, ..., x_T)$$
Use case: Real-time speech recognition, POS tagging

**Sequence-to-Vector**: Only final output matters
$$y = h_T \text{ or } y = f(h_T)$$
Use case: Sentiment analysis, time series forecasting

**Vector-to-Sequence**: Single input, multiple outputs
$$\{y_1, ..., y_T\} = \text{RNN}(x, \mathbf{0}, ..., \mathbf{0})$$
Use case: Text generation from topic vector

**Encoder-Decoder**: Process all input, then generate all output
$$c = \text{Encoder}(x_1, ..., x_T)$$
$$\{y_1, ..., y_{T'}\} = \text{Decoder}(c)$$
Use case: Machine translation

### 8.5 Backpropagation Through Time (BPTT)

For a loss summed over time steps $\mathcal{L} = \sum_{t=1}^{T} \ell_t$:

**Gradient of loss at time t w.r.t. hidden state at time k** (for $k \leq t$):

$$\frac{\partial \ell_t}{\partial h_k} = \frac{\partial \ell_t}{\partial h_t} \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

**The Jacobian**:
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(W_{hh}h_{t-1} + W_{xh}x_t + b_h)) \cdot W_{hh}$$

**Gradient w.r.t. weights** (summed over all time contributions):
$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \ell_t}{\partial h_t} \left(\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}\right) \frac{\partial h_k}{\partial W_{hh}}$$

### 8.6 The Vanishing/Exploding Gradient Problem in RNNs

The product $\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$ involves repeated multiplication of the Jacobian.

**Analysis**: Let $\lambda_{max}$ be the largest eigenvalue of $W_{hh}$.

If $|\lambda_{max}| < 1$: Product shrinks exponentially → **vanishing gradients**
If $|\lambda_{max}| > 1$: Product grows exponentially → **exploding gradients**

**Consequence**: Simple RNNs struggle to learn long-range dependencies (typically fail beyond ~10-20 time steps).

**Solutions**:
1. Gradient clipping (for exploding): $g \leftarrow \min(1, \frac{\theta}{\|g\|}) \cdot g$
2. Better architectures: LSTM, GRU (designed to preserve gradients)

### 8.7 PyTorch RNN Implementation

```python
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        # Input shape: (batch, seq_len, input_size)
        # Output shape: (batch, seq_len, hidden_size)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'  # Default activation
        )
        
        # Output layer: maps final hidden state to prediction
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state: shape (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward through RNN
        # out: (batch, seq_len, hidden_size) - all hidden states
        # hn: (num_layers, batch, hidden_size) - final hidden state
        out, hn = self.rnn(x, h0)
        
        # Sequence-to-vector: use only last time step
        # out[:, -1, :] has shape (batch, hidden_size)
        prediction = self.fc(out[:, -1, :])
        
        return prediction
```

---

## 9. LSTM Networks (Deep Dive)

### 9.1 Motivation for LSTM

**Problem**: Simple RNNs cannot maintain information over long sequences due to vanishing gradients.

**Key insight**: We need a mechanism to:
1. Remember important information for long periods
2. Forget irrelevant information
3. Control what information flows in and out

**Solution**: Long Short-Term Memory (LSTM) with gated memory cells.

### 9.2 LSTM Architecture

LSTM maintains TWO state vectors:
- **Cell state** $c_t \in \mathbb{R}^{d_h}$: Long-term memory (information highway)
- **Hidden state** $h_t \in \mathbb{R}^{d_h}$: Short-term memory / output

THREE gates control information flow:
- **Forget gate** $f_t$: What to remove from cell state
- **Input gate** $i_t$: What new information to add
- **Output gate** $o_t$: What to output from cell state

### 9.3 LSTM Equations

At each time step t, given input $x_t$, previous hidden state $h_{t-1}$, and previous cell state $c_{t-1}$:

**Forget Gate**: Decides what information to discard from cell state
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

where $\sigma$ is the sigmoid function, so $f_t \in (0, 1)^{d_h}$.

**Input Gate**: Decides what new information to store
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell State**: New information to potentially add
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Cell State Update**: Combine forgetting and adding
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Here $\odot$ is element-wise multiplication:
- $f_t \odot c_{t-1}$: Retain portion of old memory
- $i_t \odot \tilde{c}_t$: Add portion of new candidate

**Output Gate**: Decides what to output
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State Update**:
$$h_t = o_t \odot \tanh(c_t)$$

### 9.4 Why LSTM Solves Vanishing Gradients

**Key insight**: The cell state update is *additive*, not *multiplicative*:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Gradient flow through cell state**:
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Over many time steps:
$$\frac{\partial c_T}{\partial c_k} = \prod_{i=k+1}^{T} f_i$$

If the network learns $f_t \approx 1$ for important information, gradients flow without vanishing!

**The cell state acts as an "information highway"** — gradients can flow unchanged for arbitrarily long distances.

### 9.5 LSTM Parameter Count

For input dimension $d_x$ and hidden dimension $d_h$:

Each gate has parameters for input and hidden state:
- $W_f, W_i, W_c, W_o \in \mathbb{R}^{d_h \times (d_h + d_x)}$
- $b_f, b_i, b_c, b_o \in \mathbb{R}^{d_h}$

**Total parameters per LSTM layer**:
$$4 \times [d_h \times (d_h + d_x) + d_h] = 4d_h(d_h + d_x + 1)$$

This is 4× the parameters of a simple RNN (one set for each of the 4 weight matrices).

### 9.6 LSTM Variants

**Peephole Connections**: Gates also look at cell state
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t, c_{t-1}] + b_f)$$

**Coupled Forget and Input Gates**: $i_t = 1 - f_t$

**GRU**: Simplified version (see next section)

### 9.7 PyTorch LSTM Implementation

```python
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize BOTH hidden states
        # h0: short-term memory, c0: long-term memory
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM returns output and tuple of (h_n, c_n)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last time step for sequence-to-vector
        prediction = self.fc(out[:, -1, :])
        return prediction
```

---

## 10. GRU Networks (Deep Dive)

### 10.1 GRU Motivation

**Observation**: LSTM's three gates and two state vectors may be more complex than necessary.

**GRU (Gated Recurrent Unit)**: Simplified gating mechanism with:
- ONE hidden state (combines cell and hidden state of LSTM)
- TWO gates (reset and update)

### 10.2 GRU Equations

At each time step t:

**Update Gate**: Controls how much of past information to keep
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

**Reset Gate**: Controls how much of past information to forget when computing candidate
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

**Candidate Hidden State**: New hidden state proposal
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

Note: The reset gate $r_t$ determines how much of $h_{t-1}$ to use.

**Hidden State Update**: Interpolate between old and candidate
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 10.3 Understanding GRU Gates

**Update gate $z_t$** acts like a combination of LSTM's forget and input gates:
- $z_t \approx 0$: Keep old hidden state, ignore new candidate → preserve memory
- $z_t \approx 1$: Replace with new candidate → update memory

**Reset gate $r_t$** controls dependency on previous hidden state:
- $r_t \approx 0$: Ignore previous hidden state → fresh start
- $r_t \approx 1$: Fully use previous hidden state → preserve context

### 10.4 GRU vs LSTM Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Hidden states | 2 (h, c) | 1 (h) |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | 4d_h(d_h + d_x + 1) | 3d_h(d_h + d_x + 1) |
| Memory mechanism | Separate cell state | Combined in hidden state |
| Computation | Slower | ~25% faster |
| Long sequences | Generally better | Comparable |

**GRU parameters**: 75% of LSTM parameters for same hidden size.

### 10.5 When to Use Each

**Use LSTM when**:
- Sequences are very long (>100 time steps)
- You need fine-grained control over memory
- Model capacity is not a concern

**Use GRU when**:
- Computational efficiency matters
- Sequences are moderate length
- You want fewer hyperparameters to tune
- As a first try before LSTM

**Empirical finding**: Performance difference is often task-dependent; both should be tried.

### 10.6 Mathematical Analysis of GRU Gradient Flow

**Hidden state update**: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

**Gradient of h_t w.r.t. h_{t-1}**:
$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + \frac{\partial z_t}{\partial h_{t-1}} \odot (h_{t-1} - \tilde{h}_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

**Key term**: $\text{diag}(1 - z_t)$

When $z_t \approx 0$ (preserve memory), gradient flows through directly with factor $(1 - z_t) \approx 1$, preventing vanishing.

### 10.7 PyTorch GRU Implementation

```python
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # GRU has single hidden state (like simple RNN)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # GRU returns output and final hidden state
        out, hn = self.gru(x, h0)
        
        prediction = self.fc(out[:, -1, :])
        return prediction
```

---

## 11. Training and Evaluating RNNs

### 11.1 Loss Functions for Sequence Tasks

**Regression (forecasting)**:

Mean Squared Error:
$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$

Mean Absolute Error:
$$\mathcal{L}_{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$$

**Sequence classification**: Cross-entropy on final output

**Sequence-to-sequence**: Sum of losses over all time steps
$$\mathcal{L} = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t)$$

### 11.2 Tensor Shape Management

**RNN input shape**: $(N, T, D)$ — batch size, sequence length, features

**Common issue**: DataLoader may return $(N, T)$ for single-feature sequences.

**Solution**: Expand dimensions
```python
# From (32, 96) to (32, 96, 1)
x = x.view(batch_size, seq_length, 1)
# or equivalently
x = x.unsqueeze(-1)
```

**For output comparison**:
```python
# Model output: (32, 1), Labels: (32,)
# Squeeze to match: (32, 1) → (32,)
output = model(x).squeeze()
```

### 11.3 Training Loop

```python
model = LSTMNet(input_size=1, hidden_size=32, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for sequences, targets in dataloader_train:
        # Reshape: (batch, seq_len) → (batch, seq_len, 1)
        sequences = sequences.view(-1, seq_length, 1)
        
        optimizer.zero_grad()
        predictions = model(sequences).squeeze()
        loss = criterion(predictions, targets)
        loss.backward()
        
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
```

### 11.4 Evaluation Metrics for Regression

**Mean Squared Error (MSE)**:
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Root Mean Squared Error (RMSE)**:
$$RMSE = \sqrt{MSE}$$

Same units as target variable; more interpretable.

**Mean Absolute Error (MAE)**:
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

Less sensitive to outliers than MSE.

**Mean Absolute Percentage Error (MAPE)**:
$$MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

Scale-independent; problematic when $y_i \approx 0$.

**R² (Coefficient of Determination)**:
$$R^2 = 1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

Fraction of variance explained; $R^2 = 1$ is perfect.

```python
import torchmetrics

mse_metric = torchmetrics.MeanSquaredError()
mae_metric = torchmetrics.MeanAbsoluteError()

model.eval()
with torch.no_grad():
    for sequences, targets in dataloader_test:
        sequences = sequences.view(-1, seq_length, 1)
        predictions = model(sequences).squeeze()
        
        mse_metric(predictions, targets)
        mae_metric(predictions, targets)

print(f"Test MSE: {mse_metric.compute():.4f}")
print(f"Test MAE: {mae_metric.compute():.4f}")
```

---

## 12. Multi-Input and Multi-Output Models

### 12.1 Multi-Input Architectures

**Motivation**: Combine multiple data sources for richer predictions.

**Architecture pattern**:
1. Process each input through dedicated sub-network
2. Concatenate learned representations
3. Pass combined representation to classifier/regressor

**Tensor concatenation**: For tensors $A \in \mathbb{R}^{n \times d_A}$ and $B \in \mathbb{R}^{n \times d_B}$:
$$[A; B] \in \mathbb{R}^{n \times (d_A + d_B)}$$

```python
class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Image processing branch: ℝ^(1×64×64) → ℝ^128
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128)
        )
        
        # Auxiliary input branch: ℝ^30 → ℝ^8
        self.aux_branch = nn.Sequential(
            nn.Linear(30, 8),
            nn.ELU()
        )
        
        # Combined classifier: ℝ^(128+8) → ℝ^K
        self.classifier = nn.Linear(128 + 8, num_classes)
    
    def forward(self, x_image, x_aux):
        # Process each input
        z_image = self.image_branch(x_image)   # (N, 128)
        z_aux = self.aux_branch(x_aux)          # (N, 8)
        
        # Concatenate along feature dimension
        z_combined = torch.cat([z_image, z_aux], dim=1)  # (N, 136)
        
        return self.classifier(z_combined)
```

### 12.2 Multi-Output Architectures

**Use cases**:
- Multi-task learning: Predict multiple related targets
- Auxiliary losses: Intermediate supervision for better gradients
- Multi-label classification: Multiple labels per sample

**Architecture pattern**:
1. Shared feature extractor
2. Multiple task-specific heads

```python
class MultiOutputNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor: ℝ^(1×64×64) → ℝ^128
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128)
        )
        
        # Task-specific heads
        self.head_alphabet = nn.Linear(128, 30)   # 30 alphabets
        self.head_character = nn.Linear(128, 964) # 964 characters
    
    def forward(self, x):
        features = self.backbone(x)
        
        out_alphabet = self.head_alphabet(features)
        out_character = self.head_character(features)
        
        return out_alphabet, out_character
```

### 12.3 Multi-Task Loss Functions

For multiple outputs with losses $\{\mathcal{L}_1, ..., \mathcal{L}_K\}$:

**Simple sum** (equal importance):
$$\mathcal{L}_{total} = \sum_{k=1}^{K} \mathcal{L}_k$$

**Weighted sum** (task importance):
$$\mathcal{L}_{total} = \sum_{k=1}^{K} \lambda_k \mathcal{L}_k$$

where $\lambda_k$ reflects task importance.

**Uncertainty weighting** (Kendall et al., 2018):
$$\mathcal{L}_{total} = \sum_{k=1}^{K} \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k$$

where $\sigma_k$ are learned parameters representing task uncertainty.

### 12.4 Loss Scale Normalization

**Problem**: Losses on different scales cause optimization imbalance.

Example: MSE for house prices (~10,000) vs. CE for ratings (~2)

**Solution**: Normalize losses before combining:
$$\tilde{\mathcal{L}}_k = \frac{\mathcal{L}_k}{\max(\mathcal{L}_k^{batch})}$$

Or use running statistics:
$$\tilde{\mathcal{L}}_k = \frac{\mathcal{L}_k}{EMA(\mathcal{L}_k)}$$

```python
# Training loop with weighted multi-task loss
criterion = nn.CrossEntropyLoss()

for images, labels_alpha, labels_char in dataloader:
    optimizer.zero_grad()
    
    out_alpha, out_char = model(images)
    
    loss_alpha = criterion(out_alpha, labels_alpha)
    loss_char = criterion(out_char, labels_char)
    
    # Weighted combination (character 2× more important)
    total_loss = 0.33 * loss_alpha + 0.67 * loss_char
    
    total_loss.backward()
    optimizer.step()
```

### 12.5 Multi-Output Evaluation

```python
from torchmetrics import Accuracy

def evaluate_multi_output(model, dataloader):
    acc_alpha = Accuracy(task="multiclass", num_classes=30)
    acc_char = Accuracy(task="multiclass", num_classes=964)
    
    model.eval()
    with torch.no_grad():
        for images, labels_alpha, labels_char in dataloader:
            out_alpha, out_char = model(images)
            
            pred_alpha = out_alpha.argmax(dim=1)
            pred_char = out_char.argmax(dim=1)
            
            acc_alpha.update(pred_alpha, labels_alpha)
            acc_char.update(pred_char, labels_char)
    
    return {
        'alphabet_accuracy': acc_alpha.compute().item(),
        'character_accuracy': acc_char.compute().item()
    }
```

---

## Quick Reference: Mathematical Formulas

### Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | $\mathbb{1}_{z>0}$ |
| ELU | $z$ if $z>0$, else $\alpha(e^z-1)$ | $1$ if $z>0$, else $\alpha e^z$ |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $s_i(\delta_{ij} - s_j)$ |

### Loss Functions

| Task | Loss | Formula |
|------|------|---------|
| Binary Classification | BCE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Multi-class Classification | CE | $-\log(\hat{y}_c)$ |
| Regression | MSE | $(y - \hat{y})^2$ |
| Regression | MAE | $|y - \hat{y}|$ |

### Optimizer Updates

| Optimizer | Update Rule |
|-----------|-------------|
| SGD | $\theta \leftarrow \theta - \eta g$ |
| Momentum | $v \leftarrow \gamma v + g$; $\theta \leftarrow \theta - \eta v$ |
| Adam | $m \leftarrow \beta_1 m + (1-\beta_1)g$; $v \leftarrow \beta_2 v + (1-\beta_2)g^2$; $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}}+\epsilon}\hat{m}$ |

### Convolution Output Size

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

### Recurrent Cell Equations

**Simple RNN**:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**LSTM**:
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$

**GRU**:
$$z_t = \sigma(W_z[h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r[h_{t-1}, x_t] + b_r)$$
$$\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
