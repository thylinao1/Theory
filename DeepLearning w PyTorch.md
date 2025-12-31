# Intermediate Deep Learning with PyTorch - Complete Notes

## Table of Contents
1. [PyTorch and Object-Oriented Programming](#1-pytorch-and-object-oriented-programming)
2. [Optimizers, Training, and Evaluation](#2-optimizers-training-and-evaluation)
3. [Vanishing and Exploding Gradients](#3-vanishing-and-exploding-gradients)
4. [Handling Images with PyTorch](#4-handling-images-with-pytorch)
5. [Convolutional Neural Networks](#5-convolutional-neural-networks)
6. [Training and Evaluating Image Classifiers](#6-training-and-evaluating-image-classifiers)
7. [Handling Sequences with PyTorch](#7-handling-sequences-with-pytorch)
8. [Recurrent Neural Networks](#8-recurrent-neural-networks)
9. [LSTM and GRU Cells](#9-lstm-and-gru-cells)
10. [Training and Evaluating RNNs](#10-training-and-evaluating-rnns)
11. [Multi-Input Models](#11-multi-input-models)
12. [Multi-Output Models](#12-multi-output-models)

---

## 1. PyTorch and Object-Oriented Programming

### 1.1 Key Definitions

**Object-Oriented Programming (OOP)**: A programming paradigm where we create "objects" (virtual entities), each with abilities called **methods** and data called **attributes**. OOP provides flexibility in defining PyTorch Datasets and Models.

**Class**: A blueprint for creating objects that defines what attributes and methods those objects will have.

**`__init__` method**: A special method (constructor) called automatically when an object is created. It initializes the object's attributes. Written with double underscores on either side.

**`self`**: A reference to the current instance of the class. Used to access attributes and methods belonging to that specific object.

**Attributes**: Data stored within an object (e.g., `self.balance` stores the balance value).

**Methods**: Functions defined inside a class that perform operations on the object's data.

### 1.2 OOP Example: BankAccount Class

```python
class BankAccount:
    def __init__(self, balance):
        # The __init__ method is called when object is created
        # self.balance becomes an attribute of the object
        self.balance = balance
    
    def deposit(self, amount):
        # Methods can modify the object's attributes
        self.balance += amount

# Creating an instance (object) of the class
account = BankAccount(100)
print(account.balance)  # Accessing attribute: 100
account.deposit(50)     # Calling method
print(account.balance)  # Now: 150
```

### 1.3 PyTorch Dataset

**PyTorch Dataset**: A class that inherits from `torch.utils.data.Dataset` and must implement three methods:
- `__init__()`: Loads and stores the data
- `__len__()`: Returns the total number of samples
- `__getitem__(idx)`: Returns a single sample (features and label) at the given index

**`super().__init__()`**: Ensures the custom Dataset class inherits all behaviors from its parent class (`torch.utils.data.Dataset`).

```python
import pandas as pd
from torch.utils.data import Dataset

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()  # Initialize parent class
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()  # Store as NumPy array
    
    def __len__(self):
        # Returns number of samples (rows in the data)
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Returns features (all columns except last) and label (last column)
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label
```

### 1.4 PyTorch DataLoader

**DataLoader**: A utility that wraps a Dataset and provides batching, shuffling, and parallel data loading. It yields batches of data ready for training.

**Batch size**: The number of samples processed together before the model's parameters are updated.

**Shuffling**: Randomizing the order of training samples each epoch to prevent the model from learning patterns based on data order.

```python
from torch.utils.data import DataLoader

# Create Dataset instance
dataset_train = WaterDataset('water_train.csv')

# Create DataLoader with batch_size=2 and shuffling enabled
dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
)

# Get one batch using next(iter(...))
features, labels = next(iter(dataloader_train))
# With batch_size=2: features shape is (2, 9), labels shape is (2,)
```

### 1.5 PyTorch Model as a Class

**nn.Module**: PyTorch's base class for all neural network modules. Custom models must inherit from this.

**Sequential Model**: A simple way to define models as a sequence of layers, but less flexible for complex architectures.

**Class-based Model**: More flexible approach that defines layers in `__init__()` and data flow in `forward()`.

**`forward()` method**: Defines how input data passes through the network layers. Called automatically when you pass data to the model.

```python
import torch.nn as nn
import torch.nn.functional as F

# Sequential approach (simpler but less flexible)
net_sequential = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

# Class-based approach (more flexible)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers in __init__
        self.fc1 = nn.Linear(9, 16)   # 9 inputs → 16 outputs
        self.fc2 = nn.Linear(16, 8)   # 16 inputs → 8 outputs
        self.fc3 = nn.Linear(8, 1)    # 8 inputs → 1 output
    
    def forward(self, x):
        # Define how data flows through layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
```

---

## 2. Optimizers, Training, and Evaluation

### 2.1 Key Definitions

**Training Loop**: The iterative process of: forward pass → loss computation → backward pass → parameter update, repeated over multiple epochs.

**Loss Function (Criterion)**: A function that measures how far the model's predictions are from the true labels. The goal of training is to minimize this.

**Optimizer**: An algorithm that updates the model's parameters based on computed gradients to minimize the loss.

**Learning Rate**: A hyperparameter that controls how much to adjust parameters in each update step. Too high causes divergence; too low causes slow training.

**Epoch**: One complete pass through the entire training dataset.

**Gradient**: The derivative of the loss with respect to a parameter, indicating the direction and magnitude of change needed to reduce the loss.

**Backpropagation**: The algorithm that computes gradients by propagating errors backward through the network.

### 2.2 Training Loop Structure

```python
import torch.nn as nn
import torch.optim as optim

net = Net()

# Define loss function (Binary Cross-Entropy for binary classification)
criterion = nn.BCELoss()

# Define optimizer (SGD with learning rate 0.01)
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for features, labels in dataloader_train:
        # 1. Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # 2. Forward pass: get model predictions
        outputs = net(features)
        
        # 3. Compute loss (reshape labels to match output shape)
        loss = criterion(outputs, labels.view(-1, 1))
        
        # 4. Backward pass: compute gradients
        loss.backward()
        
        # 5. Update parameters based on gradients
        optimizer.step()
```

### 2.3 How Optimizers Work

The optimizer receives parameter values and their gradients, then computes updates for each parameter.

**Direction of Update**: Determined by the gradient's sign.
- Positive gradient → decrease parameter (negative update)
- Negative gradient → increase parameter (positive update)

**Size of Update**: Depends on the optimizer algorithm.

### 2.4 Types of Optimizers

**Stochastic Gradient Descent (SGD)**
- Simplest optimizer
- Update size depends only on the learning rate (fixed for all parameters)
- Computationally efficient but rarely used in practice due to simplicity
- Formula: `param = param - lr * gradient`

```python
optimizer = optim.SGD(net.parameters(), lr=0.001)
```

**Adaptive Gradient (Adagrad)**
- Adapts learning rate per parameter
- Decreases learning rate for frequently updated parameters
- Well-suited for sparse data
- Drawback: Can decrease learning rate too aggressively

```python
optimizer = optim.Adagrad(net.parameters(), lr=0.001)
```

**Root Mean Square Propagation (RMSprop)**
- Addresses Adagrad's aggressive learning rate decay
- Adapts learning rate based on recent gradient magnitudes
- Uses exponential moving average of squared gradients

```python
optimizer = optim.RMSprop(net.parameters(), lr=0.001)
```

**Adaptive Moment Estimation (Adam)**
- Most versatile and widely used optimizer
- Combines RMSprop with momentum
- **Momentum**: Weighted average of past gradients (recent gradients have more weight)
- Accelerates training by considering both gradient size and momentum
- Recommended default choice for most tasks

```python
optimizer = optim.Adam(net.parameters(), lr=0.001)
```

### 2.5 Model Evaluation

**Evaluation Mode (`net.eval()`)**: Switches the model to evaluation mode, disabling dropout and using running statistics for batch normalization.

**`torch.no_grad()`**: Context manager that disables gradient computation, reducing memory usage and speeding up inference.

**Binary Accuracy**: Fraction of correct predictions for binary classification.

```python
import torch
from torchmetrics import Accuracy

# Set up accuracy metric for binary classification
acc = Accuracy(task='binary')

net.eval()  # Switch to evaluation mode

with torch.no_grad():  # Disable gradient computation
    for features, labels in dataloader_test:
        outputs = net(features)
        # Convert probabilities to binary predictions (threshold = 0.5)
        preds = (outputs >= 0.5).float()
        # Update accuracy metric
        acc(preds, labels.view(-1, 1))

# Compute final accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")
```

---

## 3. Vanishing and Exploding Gradients

### 3.1 Key Definitions

**Gradient Instability**: A common problem in deep neural networks where gradients become either too small (vanishing) or too large (exploding) during backpropagation.

**Vanishing Gradients**: Gradients become progressively smaller during backward pass. Earlier layers receive negligible updates, so the model fails to learn. Common with sigmoid/tanh activations in deep networks.

**Exploding Gradients**: Gradients become increasingly large during backward pass. Leads to huge parameter updates, causing training to diverge (loss becomes NaN or infinity).

### 3.2 Three-Step Solution

1. **Proper Weight Initialization**
2. **Good Activation Functions**
3. **Batch Normalization**

### 3.3 Weight Initialization

**Why it matters**: Improper initialization can cause activations to explode or vanish even before training begins.

**Goal of proper initialization**:
- Variance of layer inputs ≈ variance of layer outputs
- Variance of gradients same before and after passing through layer

**He (Kaiming) Initialization**: Designed for ReLU and similar activations. Scales initial weights based on the number of input units to maintain variance.

```python
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
        # Apply He/Kaiming initialization to each layer
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        # For sigmoid activation, specify nonlinearity
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')
    
    def forward(self, x):
        x = nn.functional.elu(self.fc1(x))
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
```

### 3.4 Activation Functions

**ReLU (Rectified Linear Unit)**
- Most commonly used activation
- Formula: `f(x) = max(0, x)`
- Advantages: Simple, computationally efficient, helps with vanishing gradients
- Drawback: **Dying Neuron Problem** - neurons outputting zero become "dead" since gradient is zero for negative inputs

```python
x = nn.functional.relu(x)
```

**ELU (Exponential Linear Unit)**
- Designed to improve upon ReLU
- Formula: `f(x) = x if x > 0 else α(e^x - 1)`
- Advantages:
  - Non-zero gradients for negative values (no dying neurons)
  - Average output near zero (reduces vanishing gradients)

```python
x = nn.functional.elu(x)
```

### 3.5 Batch Normalization

**Batch Normalization**: A technique applied after a layer that normalizes the outputs to have roughly normal distribution, then learns optimal scaling and shifting.

**How it works**:
1. **Normalize**: Subtract mean and divide by standard deviation (from current batch)
2. **Scale and Shift**: Apply learnable parameters γ (scale) and β (shift)

**Benefits**:
- Stabilizes training throughout (not just at initialization)
- Accelerates convergence (faster loss decrease)
- Provides regularization effect
- Allows the model to learn optimal input distribution for each layer

**`nn.BatchNorm1d(num_features)`**: Takes the number of features (must match preceding layer's output size).

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)  # 16 = output of fc1
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)   # 8 = output of fc2
        self.fc3 = nn.Linear(8, 1)
        
        # Apply He initialization
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(self.fc3.weight, nonlinearity="sigmoid")
    
    def forward(self, x):
        # Order: Linear → BatchNorm → Activation
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        
        x = nn.functional.sigmoid(self.fc3(x))
        return x
```

---

## 4. Handling Images with PyTorch

### 4.1 Key Definitions

**Pixel (Picture Element)**: The smallest unit of a digital image—a tiny square representing a single point with numerical color information.

**Grayscale Image**: Image where each pixel is represented by a single integer (0-255), representing shades from black (0) to white (255).

**Color Image (RGB)**: Image where each pixel has three integers representing Red, Green, and Blue channel intensities (0-255 each).

**Image Shape in PyTorch**: `(batch_size, channels, height, width)`
- Grayscale: 1 channel
- Color (RGB): 3 channels

### 4.2 Loading Images with ImageFolder

**ImageFolder**: A PyTorch Dataset class that automatically loads images from a directory structure where each subfolder represents a class.

**Required Directory Structure**:
```
clouds_train/
├── cirriform clouds/
│   ├── image1.jpg
│   └── ...
├── clear sky/
├── cumulonimbus clouds/
├── cumulus clouds/
├── high cumuliform clouds/
├── stratiform clouds/
└── stratocumulus clouds/
```

**Transforms**: Operations applied to images as they are loaded (e.g., converting to tensor, resizing).

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformations
train_transforms = transforms.Compose([
    transforms.ToTensor(),        # Convert PIL Image to Tensor
    transforms.Resize((128, 128)), # Resize to uniform dimensions
])

# Create Dataset (automatically assigns labels based on folder names)
dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)

# Create DataLoader
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=1)
```

### 4.3 Displaying Images

Images loaded by PyTorch have shape `(batch, channels, height, width)`. For matplotlib display, we need `(height, width, channels)`.

```python
import matplotlib.pyplot as plt

image, label = next(iter(dataloader_train))
# image shape: (1, 3, 128, 128)

# Reshape for display:
# 1. squeeze() removes batch dimension (1, 3, 128, 128) → (3, 128, 128)
# 2. permute(1, 2, 0) reorders (3, 128, 128) → (128, 128, 3)
image = image.squeeze().permute(1, 2, 0)

plt.imshow(image)
plt.show()
```

### 4.4 Data Augmentation

**Data Augmentation**: Technique of applying random transformations to training images to artificially increase dataset size and diversity.

**Benefits**:
- Generates more training data
- Makes model robust to variations in real-world images
- Reduces overfitting (model learns to ignore random transformations)

**Common Augmentations**:
- `RandomHorizontalFlip()`: Randomly flip image horizontally
- `RandomRotation(degrees)`: Randomly rotate by angle up to specified degrees
- `RandomAutocontrast()`: Randomly adjust image contrast

**Important**: Augmentations should NOT change the label. Avoid augmentations that would make the class ambiguous (e.g., color shift on lemons → looks like lime; vertical flip on "W" → looks like "M").

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 50% chance to flip
    transforms.RandomRotation(45),       # Rotate 0-45 degrees randomly
    transforms.RandomAutocontrast(),     # Adjust contrast randomly
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)
```

---

## 5. Convolutional Neural Networks

### 5.1 Key Definitions

**Convolutional Neural Network (CNN)**: A type of neural network designed for processing grid-like data (especially images) using convolutional layers that detect spatial patterns.

**Why not Linear Layers for Images?**
- A 256×256 color image has 256×256×3 = 196,608 inputs
- A single layer with 1,000 neurons → ~200 million parameters
- Problems: Slow training, high overfitting risk
- Linear layers don't recognize spatial patterns (same object in different locations isn't recognized)

**Filter (Kernel)**: A small grid of learnable parameters that slides over the input to detect features. Typical sizes: 3×3, 5×5.

**Feature Map**: The output of applying a filter to an input—it preserves spatial information about detected features.

**Convolution Operation**: Element-wise multiplication between input patch and filter, then summing all values to produce a single output value.

### 5.2 Convolutional Layer

**`nn.Conv2d(in_channels, out_channels, kernel_size, padding)`**:
- `in_channels`: Number of input feature maps (3 for RGB images)
- `out_channels`: Number of filters (each produces one feature map)
- `kernel_size`: Size of each filter (e.g., 3 for 3×3)
- `padding`: Zero-padding around input edges

**Zero-Padding**: Adding zeros around the input image before convolution.
- Maintains spatial dimensions (input size ≈ output size)
- Ensures border pixels are processed equally (without padding, they're covered fewer times)

```python
# Creates 32 filters of size 3×3 applied to 3-channel input
conv_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```

### 5.3 Convolution Operation Explained

For a 3×3 filter on a 3×3 input patch:
1. Multiply corresponding elements: `input[i,j] * filter[i,j]`
2. Sum all 9 products
3. Result is one value in the output feature map

Example:
```
Input patch:     Filter:          Element-wise:
[1, 2, 3]       [2, 0, 1]        [2, 0, 3]
[4, 5, 6]   ×   [1, 1, 0]    =   [4, 5, 0]
[7, 8, 9]       [0, 2, 1]        [0, 16, 9]

Sum = 2+0+3+4+5+0+0+16+9 = 39
```

### 5.4 Max Pooling

**Max Pooling**: Downsampling operation that slides a non-overlapping window over feature maps and selects the maximum value in each window.

**Purpose**:
- Reduces spatial dimensions (reduces computation and parameters)
- Provides translation invariance
- Retains most important features (maximum activations)

**`nn.MaxPool2d(kernel_size)`**: With `kernel_size=2`, halves height and width.

```python
# 2×2 window reduces 64×64 → 32×32
pool_layer = nn.MaxPool2d(kernel_size=2)
```

### 5.5 CNN Architecture

A typical CNN has two parts:
1. **Feature Extractor**: Convolutional layers + activations + pooling (learns feature representations)
2. **Classifier**: Linear layers that predict classes from learned features

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Feature extractor: Conv → Activation → Pool (repeated)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3, 64, 64) → (32, 64, 64)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),                  # (32, 64, 64) → (32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (32, 32, 32) → (64, 32, 32)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),                  # (64, 32, 32) → (64, 16, 16)
            nn.Flatten(),                                 # (64, 16, 16) → (64*16*16,)
        )
        
        # Classifier: Linear layer(s)
        self.classifier = nn.Linear(64*16*16, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
```

### 5.6 Calculating Feature Extractor Output Size

Starting with input (3, 64, 64):
1. Conv2d(3→32, padding=1): **(32, 64, 64)** — padding preserves H×W
2. MaxPool2d(2): **(32, 32, 32)** — halves H and W
3. Conv2d(32→64, padding=1): **(64, 32, 32)** — padding preserves H×W
4. MaxPool2d(2): **(64, 16, 16)** — halves H and W
5. Flatten: **(64×16×16 = 16,384)**

---

## 6. Training and Evaluating Image Classifiers

### 6.1 Choosing Appropriate Augmentations

**Task-Aware Augmentation**: Choose augmentations that don't change the semantic meaning of the image for your specific task.

| Augmentation | Good For | Avoid When |
|--------------|----------|------------|
| Horizontal Flip | Most natural images | Text, directional symbols |
| Rotation | Clouds, cells | Characters that look different rotated |
| Color Shift | Non-color-dependent classes | Fruit classification, traffic lights |
| Vertical Flip | Aerial views | Characters, scenes with gravity |

**Appropriate for Cloud Classification**:
- `RandomRotation`: Different angles of cloud formations
- `RandomHorizontalFlip`: Different viewpoints of sky
- `RandomAutocontrast`: Different lighting conditions

```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64, 64)),  # Smaller images for faster training
])
```

### 6.2 Loss Functions for Classification

**Binary Cross-Entropy (BCELoss)**: For binary classification (2 classes).

**Cross-Entropy Loss (CrossEntropyLoss)**: For multi-class classification (>2 classes). Combines LogSoftmax and NLLLoss.

```python
# Binary classification (water potability: 0 or 1)
criterion = nn.BCELoss()

# Multi-class classification (7 cloud types)
criterion = nn.CrossEntropyLoss()
```

### 6.3 Image Classifier Training Loop

```python
# Define model for 7 classes
net = Net(num_classes=7)

# Multi-class classification loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(3):
    running_loss = 0.0
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
```

### 6.4 Test Data: No Augmentation!

**Critical**: Test data should NOT have augmentation transforms. We want to evaluate on original images, not random transformations.

```python
# Test transforms: only convert and resize
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

dataset_test = ImageFolder("clouds_test", transform=test_transforms)
dataloader_test = DataLoader(dataset_test, batch_size=16)
```

### 6.5 Precision and Recall

**Precision**: Of all positive predictions, what fraction were correct?
- Formula: `True Positives / (True Positives + False Positives)`
- High precision = few false alarms

**Recall**: Of all actual positives, what fraction were identified?
- Formula: `True Positives / (True Positives + False Negatives)`
- High recall = few missed positives

### 6.6 Multi-Class Averaging Methods

For multi-class problems, we get one precision/recall per class. We can aggregate them:

**Micro Average**: Calculate globally by counting total TP, FP, FN across all classes.
- Good for imbalanced datasets
- Dominated by majority classes

**Macro Average**: Calculate metric per class, then average.
- Treats all classes equally regardless of size
- Good when minority class performance matters

**Weighted Average**: Calculate per class, then weighted average by class size.
- Accounts for imbalance
- Larger classes have more impact

```python
from torchmetrics import Precision, Recall

# Macro-averaged metrics
metric_precision = Precision(task="multiclass", num_classes=7, average='macro')
metric_recall = Recall(task="multiclass", num_classes=7, average='macro')

# Options: average='micro', 'macro', 'weighted', or None (per-class)
```

### 6.7 Multi-Class Evaluation Loop

```python
from torchmetrics import Precision, Recall

metric_precision = Precision(task="multiclass", num_classes=7, average='macro')
metric_recall = Recall(task="multiclass", num_classes=7, average='macro')

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        # Get predicted class (highest score)
        _, preds = torch.max(outputs, 1)
        
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

### 6.8 Per-Class Analysis

```python
# Get scores per class (average=None)
metric_precision = Precision(task='multiclass', num_classes=7, average=None)

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)

precision = metric_precision.compute()

# Map class names to their precision scores
precision_per_class = {
    k: precision[v].item()
    for k, v in dataset_test.class_to_idx.items()
}
print(precision_per_class)
# Example output: {'clear sky': 1.0, 'cumulus clouds': 0.85, ...}
```

---

## 7. Handling Sequences with PyTorch

### 7.1 Key Definitions

**Sequential Data**: Data where order matters and points have temporal or spatial dependencies.
- Time series (stock prices, weather, electricity usage)
- Text (word order determines meaning)
- Audio (sample order determines sound)

**Sequence Length**: Number of consecutive data points used as input for one prediction.

**Look-Ahead Bias**: Contamination of training data with future information that wouldn't be available at prediction time.

### 7.2 Time Series Train-Test Split

**Critical**: Don't split randomly! Use temporal split to avoid look-ahead bias.

```
|-------- Training Data --------|---- Test Data ----|
  Year 1      Year 2     Year 3        Year 4
```

### 7.3 Creating Sequences from Time Series

For electricity consumption prediction with 15-minute intervals:
- Predict next value based on previous 24 hours
- 24 hours × 4 readings/hour = 96 data points per sequence

```python
import numpy as np

def create_sequences(df, seq_length):
    """
    Create input-target pairs from sequential data.
    
    Args:
        df: DataFrame with timestamp and value columns
        seq_length: Number of time steps in each input sequence
    
    Returns:
        xs: Array of input sequences
        ys: Array of target values
    """
    xs, ys = [], []
    
    # Loop ensures seq_length points + 1 target available
    for i in range(len(df) - seq_length):
        # Input: seq_length consecutive values
        x = df.iloc[i:(i+seq_length), 1]  # Column 1 = consumption
        # Target: next value after the sequence
        y = df.iloc[i+seq_length, 1]
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

# Create sequences (96 steps = 24 hours of 15-min intervals)
X_train, y_train = create_sequences(train_data, 24*4)
print(X_train.shape)  # (num_sequences, 96)
print(y_train.shape)  # (num_sequences,)
```

### 7.4 TensorDataset for Sequences

**TensorDataset**: Wraps tensors into a Dataset where each sample is a tuple of corresponding tensor elements.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create sequences
X_train, y_train = create_sequences(train_data, 24*4)

# Create TensorDataset from NumPy arrays
dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(),  # Convert to float tensor
    torch.from_numpy(y_train).float(),
)

print(len(dataset_train))  # Number of sequences

# Create DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
```

---

## 8. Recurrent Neural Networks

### 8.1 Key Definitions

**Feed-Forward Neural Network**: Network where data flows in one direction (input → output) with no loops.

**Recurrent Neural Network (RNN)**: Network with connections that loop back, allowing information to persist across time steps. Also called "memory cells."

**Hidden State (h)**: Internal representation maintained by an RNN cell that captures information from previous time steps.

**Unrolling**: Visualizing an RNN by showing one copy of the cell for each time step, making the temporal flow explicit.

### 8.2 RNN Cell Operation

At each time step t, the RNN cell:
1. Receives input x_t and previous hidden state h_{t-1}
2. Computes weighted combination and applies activation
3. Produces output y_t and new hidden state h_t

The first hidden state h_0 is typically initialized to zeros.

### 8.3 Deep RNNs

Multiple RNN layers stacked vertically. Each layer's output becomes the next layer's input.

### 8.4 RNN Architectures

**Sequence-to-Sequence**: Input sequence → Output at every time step
- Use case: Real-time speech recognition

**Sequence-to-Vector**: Input sequence → Single output at end
- Use case: Text classification, time series forecasting
- Let model process entire sequence before predicting

**Vector-to-Sequence**: Single input → Output sequence
- Use case: Text/music generation from prompt

**Encoder-Decoder**: Input sequence (encoder) → Output sequence (decoder)
- Different from seq-to-seq: all input processed before any output
- Use case: Machine translation

### 8.5 Building an RNN in PyTorch

**`nn.RNN(input_size, hidden_size, num_layers, batch_first)`**:
- `input_size`: Number of features per time step (1 for univariate time series)
- `hidden_size`: Dimension of hidden state
- `num_layers`: Number of stacked RNN layers
- `batch_first`: If True, input shape is (batch, seq_len, features)

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,      # 1 feature (electricity consumption)
            hidden_size=32,    # 32-dimensional hidden state
            num_layers=2,      # 2 stacked RNN layers
            batch_first=True,  # Input: (batch, seq_len, features)
        )
        # Map final hidden state to prediction
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(2, x.size(0), 32)
        
        # RNN forward pass
        # out: all outputs (batch, seq_len, hidden_size)
        # _: final hidden state (not needed here)
        out, _ = self.rnn(x, h0)
        
        # Take only last time step's output for seq-to-vector
        # out[:, -1, :] has shape (batch, hidden_size)
        out = self.fc(out[:, -1, :])
        return out
```

---

## 9. LSTM and GRU Cells

### 9.1 The Short-Term Memory Problem

**Problem with Plain RNNs**: Hidden state loses information about distant past. By the time a long sequence is processed, early information is largely forgotten.

**Solution**: More sophisticated cells (LSTM, GRU) with gating mechanisms.

### 9.2 Long Short-Term Memory (LSTM)

**LSTM Cell**: Has TWO hidden states:
- **h (short-term memory)**: Similar to basic RNN hidden state
- **c (long-term memory)**: Cell state that can preserve information over many time steps

**Three Gates** (learned mechanisms controlling information flow):
1. **Forget Gate**: Decides what to remove from long-term memory
2. **Input Gate**: Decides what new information to store in long-term memory
3. **Output Gate**: Decides what to output based on current state

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        # Initialize BOTH hidden states
        h0 = torch.zeros(2, x.size(0), 32)  # Short-term
        c0 = torch.zeros(2, x.size(0), 32)  # Long-term
        
        # Pass both states as tuple
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
```

### 9.3 Gated Recurrent Unit (GRU)

**GRU Cell**: Simplified LSTM with ONE hidden state.
- Merges long-term and short-term memory
- No separate output gate
- Fewer parameters → less computation
- Often performs comparably to LSTM

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        # GRU has single hidden state (like basic RNN)
        h0 = torch.zeros(2, x.size(0), 32)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### 9.4 RNN vs LSTM vs GRU Comparison

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Hidden States | 1 | 2 (h, c) | 1 |
| Gates | 0 | 3 | 2 |
| Parameters | Fewest | Most | Medium |
| Long-term Memory | Poor | Best | Good |
| Computation | Fastest | Slowest | Medium |
| Common Use | Rarely used | Default choice | Alternative to LSTM |

**Recommendation**:
- Plain RNN: Educational purposes only
- LSTM: Default choice, especially for longer sequences
- GRU: Try when LSTM is slow; often similar performance with less computation

---

## 10. Training and Evaluating RNNs

### 10.1 Mean Squared Error Loss

**MSE Loss**: For regression tasks (predicting continuous values).

Formula: `MSE = (1/n) × Σ(predicted - actual)²`

**Why squaring?**
1. Positive and negative errors don't cancel out
2. Large errors are penalized more heavily

```python
criterion = nn.MSELoss()
```

### 10.2 Tensor Shape Operations

**Problem**: RNN layers expect input shape `(batch, seq_len, features)`, but DataLoader may drop the features dimension if there's only one feature.

**Expanding Tensors (`view`)**: Add dimensions to match expected shape.

```python
# DataLoader gives shape: (batch_size, seq_length) = (32, 96)
# RNN expects: (batch_size, seq_length, num_features) = (32, 96, 1)

seqs = seqs.view(32, 96, 1)  # Add feature dimension
```

**Squeezing Tensors (`squeeze`)**: Remove dimensions of size 1.

```python
# Model output: (batch_size, 1) = (32, 1)
# Labels: (batch_size,) = (32,)
# For loss computation, shapes must match

outputs = net(seqs).squeeze()  # (32, 1) → (32,)
```

### 10.3 RNN Training Loop

```python
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(3):
    for seqs, labels in dataloader_train:
        # Expand input to add feature dimension
        seqs = seqs.view(32, 96, 1)
        
        outputs = net(seqs)
        loss = criterion(outputs.squeeze(), labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

### 10.4 RNN Evaluation Loop

```python
import torchmetrics

mse = torchmetrics.MeanSquaredError()

net.eval()
with torch.no_grad():
    for seqs, labels in dataloader_test:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs).squeeze()  # Match label shape
        mse(outputs, labels)

test_mse = mse.compute()
print(f"Test MSE: {test_mse}")
```

---

## 11. Multi-Input Models

### 11.1 Key Definitions

**Multi-Input Model**: Model that accepts multiple data sources/modalities as input.

**Use Cases**:
- **Multiple information sources**: Two images of car → predict model
- **Multi-modal learning**: Image + text → answer question about image
- **Metric learning**: Compare two inputs (passport photo vs. live photo)
- **Self-supervised learning**: Two augmented versions → learn they're the same

### 11.2 Multi-Input Dataset

The Dataset's `__getitem__` must return all inputs plus the label.

```python
from torch.utils.data import Dataset
from PIL import Image

class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
        """
        Args:
            transform: Image transformations
            samples: List of (image_path, alphabet_vector, label) tuples
        """
        self.transform = transform
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, alphabet, label = self.samples[idx]
        
        # Load and transform image
        img = Image.open(img_path).convert('L')  # 'L' = grayscale
        img_transformed = self.transform(img)
        
        # Return all inputs and label
        return img_transformed, alphabet, label
```

### 11.3 Tensor Concatenation

**`torch.cat(tensors, dim)`**: Concatenates tensors along specified dimension.

```python
import torch

# 2D tensors
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along dim=0 (vertically/rows)
torch.cat((a, b), dim=0)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# Concatenate along dim=1 (horizontally/columns)
torch.cat((a, b), dim=1)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
```

### 11.4 Multi-Input Model Architecture

**Strategy**: Process each input through dedicated layers, concatenate outputs, pass to classifier.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Sub-network for image input
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)  # Output: 128-dim vector
        )
        
        # Sub-network for alphabet input (one-hot vector of 30 alphabets)
        self.alphabet_layer = nn.Sequential(
            nn.Linear(30, 8),  # Output: 8-dim vector
            nn.ELU(),
        )
        
        # Classifier takes concatenated outputs
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 964),  # 964 character classes
        )
    
    def forward(self, x_image, x_alphabet):
        # Process each input separately
        x_image = self.image_layer(x_image)
        x_alphabet = self.alphabet_layer(x_alphabet)
        
        # Concatenate along feature dimension
        x = torch.cat((x_image, x_alphabet), dim=1)
        
        return self.classifier(x)
```

### 11.5 Multi-Input Training Loop

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, alphabets, labels in dataloader_train:
        optimizer.zero_grad()
        
        # Pass both inputs to model
        outputs = net(images, alphabets)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## 12. Multi-Output Models

### 12.1 Key Definitions

**Multi-Output Model**: Model that produces multiple predictions from one input.

**Use Cases**:
- **Multi-task learning**: Predict car make AND model from image
- **Multi-label classification**: Image can have multiple labels (beach AND people)
- **Auxiliary outputs**: Extra predictions from intermediate layers for regularization

### 12.2 Multi-Output Dataset

For two outputs (alphabet and character), the Dataset returns: `(image, label_alphabet, label_character)`.

Both labels are integers (class indices), not one-hot vectors.

```python
# samples structure: (image_path, alphabet_idx, character_idx)
dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=32)
```

### 12.3 Multi-Output Model Architecture

**Strategy**: Shared feature extractor → multiple classifier heads.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        
        # Separate classifiers for each output
        self.classifier_alpha = nn.Linear(128, 30)   # 30 alphabets
        self.classifier_char = nn.Linear(128, 964)   # 964 characters
    
    def forward(self, x):
        # Shared processing
        x_features = self.image_layer(x)
        
        # Separate predictions
        output_alpha = self.classifier_alpha(x_features)
        output_char = self.classifier_char(x_features)
        
        return output_alpha, output_char
```

### 12.4 Multi-Output Training Loop

**Key**: Calculate separate losses, combine them into total loss for optimization.

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)

for epoch in range(num_epochs):
    for images, labels_alpha, labels_char in dataloader_train:
        optimizer.zero_grad()
        
        # Get both outputs
        outputs_alpha, outputs_char = net(images)
        
        # Compute loss for each output
        loss_alpha = criterion(outputs_alpha, labels_alpha)
        loss_char = criterion(outputs_char, labels_char)
        
        # Combine losses (equal weighting)
        loss = loss_alpha + loss_char
        
        loss.backward()
        optimizer.step()
```

### 12.5 Loss Weighting

**Default (equal importance)**: `loss = loss_1 + loss_2`

**Weighted by importance**:
```python
# Character classification twice as important
loss = loss_alpha + 2 * loss_char

# Or use normalized weights summing to 1
loss = 0.33 * loss_alpha + 0.67 * loss_char
```

### 12.6 Warning: Losses on Different Scales

**Problem**: If losses have very different magnitudes, smaller loss is ignored.

Example:
- MSE for house price: can be 10,000+
- Cross-entropy for quality rating: typically 0-5

**Solution**: Scale losses before weighting.

```python
# Scale each loss by its batch maximum
loss_scaled = loss / loss.max()

# Then apply weights
total_loss = w1 * loss1_scaled + w2 * loss2_scaled
```

### 12.7 Multi-Output Evaluation

```python
import torch
from torchmetrics import Accuracy

def evaluate_model(model):
    # Separate metric for each output
    acc_alpha = Accuracy(task="multiclass", num_classes=30)
    acc_char = Accuracy(task="multiclass", num_classes=964)
    
    model.eval()
    with torch.no_grad():
        for images, labels_alpha, labels_char in dataloader_test:
            outputs_alpha, outputs_char = model(images)
            
            # Get predictions
            _, pred_alpha = torch.max(outputs_alpha, 1)
            _, pred_char = torch.max(outputs_char, 1)
            
            # Update metrics
            acc_alpha(pred_alpha, labels_alpha)
            acc_char(pred_char, labels_char)
    
    print(f"Alphabet Accuracy: {acc_alpha.compute()}")
    print(f"Character Accuracy: {acc_char.compute()}")
```

---

## Quick Reference: Common PyTorch Patterns

### Import Statements
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics import Accuracy, Precision, Recall, MeanSquaredError
```

### Training Loop Template
```python
net = Net()
criterion = nn.CrossEntropyLoss()  # or BCELoss, MSELoss
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Evaluation Loop Template
```python
metric = Accuracy(task='multiclass', num_classes=N)

net.eval()
with torch.no_grad():
    for inputs, labels in dataloader_test:
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        metric(preds, labels)

print(metric.compute())
```

### Layer Dimension Cheat Sheet
| Layer | Input → Output |
|-------|----------------|
| `nn.Linear(in, out)` | `(batch, in)` → `(batch, out)` |
| `nn.Conv2d(in_ch, out_ch, k)` | `(batch, in_ch, H, W)` → `(batch, out_ch, H', W')` |
| `nn.MaxPool2d(2)` | `(batch, ch, H, W)` → `(batch, ch, H/2, W/2)` |
| `nn.RNN/LSTM/GRU` | `(batch, seq, features)` → `(batch, seq, hidden)` |
| `nn.Flatten()` | `(batch, ch, H, W)` → `(batch, ch*H*W)` |
