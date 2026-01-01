# Deep Reinforcement Learning: Comprehensive Study Notes

## Table of Contents

1. [Introduction to Deep Reinforcement Learning](#chapter-1-introduction-to-deep-reinforcement-learning)
2. [Deep Q-Learning Fundamentals](#chapter-2-deep-q-learning-fundamentals)
3. [DQN Improvements and Extensions](#chapter-3-dqn-improvements-and-extensions)
4. [Policy Gradient Methods](#chapter-4-policy-gradient-methods)
5. [Actor-Critic Methods](#chapter-5-actor-critic-methods)
6. [Proximal Policy Optimization (PPO)](#chapter-6-proximal-policy-optimization-ppo)
7. [Advanced Training Techniques](#chapter-7-advanced-training-techniques)
8. [Hyperparameter Optimization](#chapter-8-hyperparameter-optimization)
9. [Advanced DRL Methods for Finance](#chapter-9-advanced-drl-methods-for-finance)

---

## Chapter 1: Introduction to Deep Reinforcement Learning

### 1.1 Why Deep Reinforcement Learning?

Traditional Reinforcement Learning excels in **low-dimensional tasks** like the Frozen Lake environment, where the state and action spaces are small enough to be represented in tables. However, real-world applications such as video games, robotics, and financial trading require **high-dimensional state and action spaces** where traditional RL struggles due to the curse of dimensionality.

**Deep Reinforcement Learning (DRL)** addresses this limitation by combining two powerful ingredients:

1. **Reinforcement Learning concepts** — the framework for sequential decision-making
2. **Deep Learning** — neural networks that can approximate complex functions

```
┌─────────────────────────────────────────────────────────────┐
│                    DRL = RL + Deep Learning                 │
├─────────────────────────────────────────────────────────────┤
│  Traditional RL:  Q-tables, small state spaces              │
│  Deep Learning:   Function approximation via neural nets    │
│  DRL:             Neural networks learn value/policy funcs  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Reinforcement Learning Framework

In reinforcement learning, an **agent** interacts with an **environment** over discrete time steps within an **episode**:

```
          ┌──────────────────────────────────────┐
          │            ENVIRONMENT               │
          │                                      │
          │   ┌─────────┐      ┌─────────────┐   │
          │   │ State   │      │   Reward    │   │
          │   │  s_t    │      │   r_{t+1}   │   │
          │   └────┬────┘      └──────┬──────┘   │
          │        │                  │          │
          └────────┼──────────────────┼──────────┘
                   │                  │
                   ▼                  ▼
          ┌────────────────────────────────────┐
          │              AGENT                  │
          │                                     │
          │   Observes s_t → Selects a_t       │
          │                                     │
          └─────────────┬──────────────────────┘
                        │
                        │ Action a_t
                        ▼
          ┌──────────────────────────────────────┐
          │            ENVIRONMENT               │
          │                                      │
          │   Responds with r_{t+1} and s_{t+1}  │
          └──────────────────────────────────────┘
```

**The RL Loop at each step t:**
1. Agent observes state $s_t$ from the environment
2. Agent takes action $a_t$ based on its policy
3. Environment responds with reward $r_{t+1}$ and new state $s_{t+1}$
4. Process repeats until episode termination

### 1.3 Key Definitions

**Policy $\pi(s_t)$**: The mapping from a given state to the action the agent will select. Policies can be:
- **Deterministic**: Always selects the same action for a given state
- **Stochastic**: Outputs a probability distribution over possible actions

**Trajectory $\tau$**: The sequence of all states and actions in an episode:
$$\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$$

**Episode Return $R_\tau$**: The discounted sum of rewards accumulated along trajectory $\tau$:
$$R_\tau = \sum_{t=0}^{T} \gamma^t r_{t+1}$$

where $\gamma \in [0,1]$ is the **discount factor** that determines how much future rewards are valued compared to immediate rewards.

### 1.4 Basic DRL Training Loop

The fundamental structure underlying most DRL algorithms:

```python
# Initialize environment and neural network
env = gym.make('LunarLander-v2')
network = QNetwork(state_size=8, action_size=4)
optimizer = optim.Adam(network.parameters(), lr=0.0001)

# Training loop
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # Agent selects action based on state and network
        action = select_action(network, state)
        
        # Environment responds
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Calculate loss and update network
        loss = calculate_loss(network, state, action, next_state, reward, done)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
```

**Key insight**: Unlike supervised learning where labels are provided, in DRL there are no labels. The agent creates its own training data by experiencing the environment. The loss function is a tool constructed to obtain gradients that guide the agent toward better policies.

---

## Chapter 2: Deep Q-Learning Fundamentals

### 2.1 From Q-Learning to Deep Q-Learning

**Q-Learning** is a value-based method that learns the **Action-Value Function Q**, associating a value to any combination of state $s$ and action $a$.

**Definition — Action-Value Function**:
$$Q_\pi(s, a) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s, a_t = a, \pi\right]$$

This represents the expected cumulative reward if action $a$ is taken in state $s$, then following policy $\pi$ thereafter.

**The Optimal Policy Principle**: If an agent had perfect knowledge of the Q function, it could always select the action with the highest value:
$$a^* = \arg\max_a Q(s, a)$$

### 2.2 The Bellman Equation

Q-values satisfy a recursive relationship called the **Bellman Equation**:

$$Q(s_t, a_t) = r_{t+1} + \gamma \max_{a_{t+1}} Q(s_{t+1}, a_{t+1})$$

```
┌──────────────────────────────────────────────────────────────────┐
│                      BELLMAN EQUATION                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Q(s_t, a_t) = r_{t+1} + γ · max Q(s_{t+1}, a_{t+1})           │
│       ↑            ↑            a_{t+1}                          │
│       │            │              ↑                              │
│   Current      Immediate      Best Q-value                       │
│   Q-value       reward        in next state                      │
│                                                                  │
│   "The value of being in state s and taking action a equals     │
│    the immediate reward plus the discounted value of the         │
│    best action in the next state"                                │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 The Q-Network Architecture

**Challenge with Traditional Q-Learning**: Uses a Q-table to store values for every state-action pair. This does not scale — as state space grows, the table size explodes exponentially.

**Solution — The Q-Network**: Replace the Q-table with a neural network that approximates the Q function:

```
            Input Layer          Hidden Layers         Output Layer
         ┌─────────────┐      ┌───────────────┐     ┌─────────────┐
         │   State     │      │               │     │  Q(s, a_0)  │
         │   s_t       │──────│    64 nodes   │─────│  Q(s, a_1)  │
         │             │      │    ReLU       │     │  Q(s, a_2)  │
         │  (dim = 8)  │──────│    64 nodes   │─────│  Q(s, a_3)  │
         │             │      │    ReLU       │     │             │
         └─────────────┘      └───────────────┘     └─────────────┘
                                                         ↓
                                                   Action = argmax
```

**Advantages of Neural Network Approximation**:
- Handles high-dimensional states (e.g., raw pixels from video games)
- Generalizes across similar states
- Scales to large or continuous state spaces

### 2.4 Q-Network Implementation

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Two hidden layers with 64 nodes each
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        # Output layer: one Q-value per action
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        # ReLU activations for hidden layers
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        # Linear output (no activation) for Q-values
        return self.fc3(x)
```

**Architecture Constraints**:
- Input size = dimension of state space
- Output size = number of possible actions
- Hidden layers are flexible design choices

### 2.5 The Barebone DQN Algorithm

The simplest form of Deep Q-Learning, combining:
1. The generic DRL training loop
2. A Q-network for value approximation
3. Core principles of Q-learning

**Action Selection**:
```python
def select_action(q_network, state):
    # Forward pass to get Q-values for all actions
    q_values = q_network(state)
    # Select action with highest Q-value
    action = torch.argmax(q_values).item()
    return action
```

**Loss Function — Squared Bellman Error**:

$$\mathcal{L}(\theta) = \left(r_{t+1} + \gamma \max_{a_{t+1}} \hat{Q}_\theta(s_{t+1}, a_{t+1}) - \hat{Q}_\theta(s_t, a_t)\right)^2$$

```
┌────────────────────────────────────────────────────────────────┐
│                    DQN LOSS FUNCTION                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│              ┌──────────────────────────────────┐ 2            │
│   L(θ) =     │  TD Target  -  Current Q Value  │              │
│              └──────────────────────────────────┘              │
│                    ↓               ↓                           │
│              r + γ·max Q(s',a')   Q(s,a)                       │
│                                                                │
│   This is also called the "Temporal Difference (TD) Error"     │
└────────────────────────────────────────────────────────────────┘
```

```python
def calculate_loss(q_network, state, action, next_state, reward, done):
    gamma = 0.99  # Discount factor
    
    # Get Q-values for current state
    q_values = q_network(state)
    current_state_q_value = q_values[action]
    
    # Get max Q-value for next state (TD target component)
    next_state_q_value = q_network(next_state).max()
    
    # TD target: reward + discounted next state value
    # Multiply by (1-done) to zero out next state value if episode ends
    target_q_value = reward + gamma * next_state_q_value * (1 - done)
    
    # Squared Bellman (TD) error
    loss = nn.MSELoss()(current_state_q_value, target_q_value)
    return loss
```

**Limitations of Barebone DQN**:
- Uses only the latest experience at each update
- Consecutive experiences are highly correlated → poor learning
- No exploration mechanism → may miss optimal actions
- Unstable updates → target shifts with each network update

---

## Chapter 3: DQN Improvements and Extensions

### 3.1 Experience Replay

**Problem with Barebone DQN**: Learning from consecutive, highly correlated experiences is inefficient. The agent may also "forget" important early experiences as it focuses on recent ones.

**Solution — Experience Replay**: Store experiences in a **Replay Memory Buffer** and learn from random batches of past experiences at each step.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIENCE REPLAY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: Store experience (s, a, r, s', done) in buffer        │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Experience Buffer (Replay Memory)                       │   │
│   │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐    │   │
│   │  │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │ e8 │ e9 │ eN │    │   │
│   │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘    │   │
│   │  Oldest                                      Newest      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Step 2: Sample random batch for training                       │
│                                                                  │
│         ┌────┐   ┌────┐   ┌────┐   ┌────┐                       │
│         │ e3 │   │ e7 │   │ e1 │   │ e5 │  → Train network      │
│         └────┘   └────┘   └────┘   └────┘                       │
│                                                                  │
│   Benefits:                                                      │
│   • Breaks correlation between consecutive samples               │
│   • Each experience can be used multiple times                   │
│   • More stable and efficient learning                           │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation — Double-Ended Queue (Deque)**:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        # Deque automatically drops oldest items when at capacity
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a new experience in the buffer"""
        experience_tuple = (state, action, reward, next_state, done)
        self.memory.append(experience_tuple)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        batch = random.sample(self.memory, batch_size)
        
        # Unpack into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors with appropriate shapes
        states_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
    
    def __len__(self):
        return len(self.memory)
```

**Updated Loss Calculation for Batches**:

```python
# Sample batch from replay buffer
states, actions, rewards, next_states, dones = replay_buffer.sample(64)

# Get Q-values for selected actions
q_values = q_network(states).gather(1, actions).squeeze(1)

# Get max Q-values for next states
next_state_q_values = q_network(next_states).amax(1)

# Calculate TD targets
target_q_values = rewards + gamma * next_state_q_values * (1 - dones)

# Mean Squared Bellman Error over batch
loss = nn.MSELoss()(q_values, target_q_values)
```

### 3.2 Epsilon-Greedy Exploration

**Problem**: Without exploration, the agent may get stuck in suboptimal policies by always exploiting current knowledge.

**Solution — Decayed Epsilon-Greedy**: With probability $\epsilon$, select a random action; otherwise, select the best action according to Q-values. Decay $\epsilon$ over time to shift from exploration to exploitation.

**Epsilon Decay Schedule**:
$$\epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \cdot e^{-\frac{step}{decay}}$$

```
        ε
    1.0 ┤╲
        │ ╲
    0.9 ┤  ╲
        │   ╲
    0.5 ┤    ╲__
        │       ╲___
    0.2 ┤           ╲_____
        │                 ╲__________
    0.05┤                            ═══════════════
        └───────────────────────────────────────────→ steps
              Early training        Later training
              (more exploration)   (more exploitation)
```

```python
import math
import random

def select_action(q_values, step, start=0.9, end=0.05, decay=1000):
    """Epsilon-greedy action selection with decay"""
    # Calculate current epsilon
    epsilon = end + (start - end) * math.exp(-step / decay)
    
    # With probability epsilon, choose random action
    if random.random() < epsilon:
        return random.choice(range(len(q_values)))
    
    # Otherwise, choose best action
    return torch.argmax(q_values).item()
```

### 3.3 Fixed Q-Targets (Target Network)

**Problem**: The TD target in the loss function involves the same network being updated. This causes the target to shift with each update, destabilizing training.

```
┌────────────────────────────────────────────────────────────────┐
│                   THE MOVING TARGET PROBLEM                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Loss = (Target - Prediction)²                                │
│              ↓           ↓                                     │
│          Uses Q-net   Uses Q-net                               │
│                                                                │
│   When we update Q-net to minimize loss,                       │
│   BOTH the target AND prediction change!                       │
│                                                                │
│   → Target keeps moving → Unstable training                    │
└────────────────────────────────────────────────────────────────┘
```

**Solution — Fixed Q-Targets**: Introduce a second network (target network) that is updated slowly:

```
┌──────────────────────────────────────────────────────────────────┐
│                     TWO-NETWORK ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐              ┌─────────────────┐          │
│   │  ONLINE NETWORK │              │  TARGET NETWORK │          │
│   │    (θ)          │   slowly     │     (θ⁻)        │          │
│   │                 │───updates────│                 │          │
│   │  Updated every  │      →       │  More stable    │          │
│   │  step via       │              │  parameters     │          │
│   │  gradient       │              │                 │          │
│   │  descent        │              │                 │          │
│   └─────────────────┘              └─────────────────┘          │
│          ↓                                ↓                      │
│   Current Q-value                   TD Target                    │
│   Q(s,a)                           r + γ·max Q(s',a')            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Soft Update Rule**:
$$\theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^-$$

where $\tau$ is typically small (e.g., 0.001), making the target network update slowly.

```python
def update_target_network(target_network, online_network, tau=0.001):
    """Soft update of target network parameters"""
    target_net_state_dict = target_network.state_dict()
    online_net_state_dict = online_network.state_dict()
    
    for key in online_net_state_dict:
        # Weighted average: mostly keep target, slightly move toward online
        target_net_state_dict[key] = (
            tau * online_net_state_dict[key] + 
            (1 - tau) * target_net_state_dict[key]
        )
    
    target_network.load_state_dict(target_net_state_dict)
```

**Updated Loss Calculation**:

```python
# Use online network for current Q-values
q_values = online_network(states).gather(1, actions).squeeze(1)

# Use target network for TD targets (no gradient tracking needed)
with torch.no_grad():
    next_q_values = target_network(next_states).amax(1)
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

loss = nn.MSELoss()(q_values, target_q_values)
```

### 3.4 The Complete DQN Algorithm

The full DQN algorithm (DeepMind, 2015) combines all three improvements:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE DQN ALGORITHM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Improvements over Barebone DQN:                                │
│  ✓ Experience Replay      → Breaks correlation, reuses data    │
│  ✓ Epsilon-Greedy        → Balances exploration/exploitation    │
│  ✓ Fixed Q-Targets       → Stabilizes training                  │
│                                                                 │
│  Algorithm:                                                     │
│  1. Initialize online network θ and target network θ⁻ = θ      │
│  2. Initialize replay buffer                                    │
│  3. For each episode:                                           │
│     a. Reset environment                                        │
│     b. For each step:                                           │
│        i.   Select action via ε-greedy from online network     │
│        ii.  Execute action, observe reward and next state       │
│        iii. Store (s, a, r, s', done) in replay buffer         │
│        iv.  Sample random batch from buffer                     │
│        v.   Compute TD target using TARGET network              │
│        vi.  Update ONLINE network via gradient descent          │
│        vii. Soft update target network                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Double DQN (DDQN)

**Problem — Maximization Bias**: Standard DQN uses the same network to both select and evaluate the best action in the TD target. When taking the maximum of noisy estimates, we tend to be overoptimistic.

$$\text{TD Target} = r + \gamma \max_a Q_{\theta^-}(s', a)$$

The $\max$ operation selects the action that happens to have the highest estimated value, which may be inflated due to noise.

**Solution — Double DQN**: Decouple action selection from value estimation:
- Use the **online network** to select the best action
- Use the **target network** to evaluate that action's value

```
┌────────────────────────────────────────────────────────────────┐
│                    DQN vs DOUBLE DQN                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Standard DQN:                                                 │
│  ─────────────                                                 │
│  TD Target = r + γ · max Q_target(s', a')                     │
│                     └────────────────────┘                     │
│                     Both selection AND evaluation              │
│                     use target network                         │
│                                                                │
│  Double DQN:                                                   │
│  ───────────                                                   │
│  a* = argmax Q_online(s', a')    ← Selection: online network  │
│                                                                │
│  TD Target = r + γ · Q_target(s', a*)  ← Evaluation: target   │
│                                                                │
│  By decoupling, we reduce overestimation bias                  │
└────────────────────────────────────────────────────────────────┘
```

**Implementation Change**:

```python
# Standard DQN
next_q_values = target_network(next_states).amax(1)

# Double DQN
with torch.no_grad():
    # Online network SELECTS the best action
    next_actions = online_network(next_states).argmax(1).unsqueeze(1)
    # Target network EVALUATES that action
    next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
```

**Key Insight**: DDQN requires only a minimal code change but significantly reduces Q-value overestimation, leading to more stable learning and often better performance.

### 3.6 Prioritized Experience Replay (PER)

**Problem**: Uniform random sampling from the replay buffer treats all experiences equally, but not all experiences are equally valuable for learning.

**Solution — Prioritized Experience Replay**: Assign priorities to experiences based on their "surprise" level (TD error). Experiences with high TD errors indicate more learning potential.

```
┌────────────────────────────────────────────────────────────────┐
│               PRIORITIZED EXPERIENCE REPLAY                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Core Idea: Sample experiences proportional to TD error       │
│                                                                │
│   Priority p_i = |δ_i| + ε    where δ_i = TD error            │
│                                                                │
│   Sampling probability:                                        │
│                         α                                      │
│              p_i                                                │
│   P(i) = ──────────                                            │
│              α                                                 │
│           Σ p_j                                                 │
│            j                                                   │
│                                                                │
│   α controls prioritization strength:                          │
│   • α = 0 → Uniform sampling (standard replay)                │
│   • α = 1 → Full prioritization                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Importance Sampling Weights**: High-priority samples are seen more often, which introduces bias. We correct this with importance sampling weights:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$$

where $\beta$ starts small and increases toward 1 over training.

**Five Key Modifications**:

1. **Priority initialization**: New transitions get highest priority
2. **Probability-based sampling**: Sample based on priority distribution
3. **Priority update**: Set sampled priorities to their TD error
4. **Importance weighting**: Weight losses by importance sampling
5. **Beta annealing**: Increase $\beta$ toward 1 over time

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha          # Priority exponent
        self.beta = beta            # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon      # Small constant to avoid zero priority
    
    def push(self, state, action, reward, next_state, done):
        """New experiences get maximum priority"""
        max_priority = max(self.priorities) if self.memory else 1.0
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        """Sample based on priorities"""
        priorities = np.array(self.priorities)
        
        # Calculate sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (1 / (len(self.memory) * probabilities)) ** self.beta
        weights = weights / np.max(weights)  # Normalize
        
        # Extract experiences
        states, actions, rewards, next_states, dones = zip(
            *[self.memory[idx] for idx in indices]
        )
        weights = [weights[idx] for idx in indices]
        
        # Convert to tensors...
        return (states_tensor, actions_tensor, rewards_tensor, 
                next_states_tensor, dones_tensor, indices, weights_tensor)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities to absolute TD error + epsilon"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + self.epsilon
    
    def increase_beta(self):
        """Anneal beta toward 1"""
        self.beta = min(1.0, self.beta + self.beta_increment)
```

**Modified Loss with Importance Weights**:

```python
# Include importance weights in loss calculation
loss = torch.sum(weights * (q_values - target_q_values) ** 2)
```

### 3.7 Summary: Evolution of DQN

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DQN EVOLUTION SUMMARY                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Barebone DQN                                                          │
│       │                                                                │
│       │ Problem: Correlated samples, forgetfulness                     │
│       ▼                                                                │
│  + Experience Replay ──────────────────────────────────────────────────│
│       │                                                                │
│       │ Problem: No exploration                                        │
│       ▼                                                                │
│  + Epsilon-Greedy ─────────────────────────────────────────────────────│
│       │                                                                │
│       │ Problem: Unstable targets                                      │
│       ▼                                                                │
│  + Fixed Q-Targets (Target Network) = Complete DQN                     │
│       │                                                                │
│       │ Problem: Q-value overestimation                                │
│       ▼                                                                │
│  + Action/Value Decoupling = Double DQN                                │
│       │                                                                │
│       │ Problem: Uniform sampling inefficient                          │
│       ▼                                                                │
│  + Priority-based Sampling = DQN with Prioritized Experience Replay    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Chapter 4: Policy Gradient Methods

### 4.1 Value-Based vs Policy-Based Methods

In **value-based methods** (DQN family), we:
1. Learn the action-value function Q
2. Derive policy by selecting action with highest Q-value
3. Policy is deterministic (or ε-greedy)

In **policy-based methods**, we:
1. Learn the policy directly
2. Policy can be stochastic (probability distribution over actions)
3. No need to estimate value functions (though we may use them)

```
┌────────────────────────────────────────────────────────────────┐
│              VALUE-BASED vs POLICY-BASED                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  VALUE-BASED (DQN):                                            │
│  ─────────────────                                             │
│  State → Q-Network → [Q(s,a₀), Q(s,a₁), Q(s,a₂), Q(s,a₃)]     │
│                              ↓                                 │
│                         argmax → Action                        │
│                                                                │
│  POLICY-BASED (Policy Gradient):                               │
│  ───────────────────────────────                               │
│  State → Policy Network → [P(a₀|s), P(a₁|s), P(a₂|s), P(a₃|s)]│
│                              ↓                                 │
│                         Sample → Action                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Advantages of Policy Methods**:
- Policy can be stochastic (useful when optimal behavior is random)
- Can handle continuous action spaces naturally
- Directly optimizes what we care about (expected return)
- Better convergence properties in some cases

**Disadvantages**:
- High variance in gradient estimates
- Less sample efficient
- Can converge to local optima

### 4.2 The Policy Network

For discrete actions, the policy network uses a **softmax output layer** to produce action probabilities:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(torch.tensor(state)))
        x = torch.relu(self.fc2(x))
        # Softmax ensures outputs sum to 1 (valid probability distribution)
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs
```

**Sampling Actions**:

```python
from torch.distributions import Categorical

def select_action(policy_network, state):
    action_probs = policy_network(state)
    
    # Create categorical distribution from probabilities
    action_dist = Categorical(action_probs)
    
    # Sample action from distribution
    action = action_dist.sample()
    
    # Get log probability (needed for loss calculation)
    log_prob = action_dist.log_prob(action)
    
    return action.item(), log_prob.reshape(1)
```

### 4.3 The Policy Gradient Theorem

**Objective**: Maximize expected return under policy $\pi_\theta$:
$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R_\tau]$$

**The Challenge**: How do we compute the gradient $\nabla_\theta J$ when the expectation depends on the policy?

**Policy Gradient Theorem** provides a tractable expression:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R_\tau \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

```
┌────────────────────────────────────────────────────────────────┐
│               POLICY GRADIENT THEOREM INTUITION                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ∇θ J = E_τ [ R_τ · Σ ∇θ log π(a_t|s_t) ]                    │
│               ↑        ↑                                       │
│           Episode   Sum of gradients of log                    │
│           return    probabilities of actions                   │
│                                                                │
│   Intuition:                                                   │
│   • If episode had HIGH return → increase probability of       │
│     the actions that were taken                                │
│   • If episode had LOW return → decrease probability of        │
│     the actions that were taken                                │
│                                                                │
│   We're "reinforcing" actions in successful episodes           │
└────────────────────────────────────────────────────────────────┘
```

### 4.4 REINFORCE Algorithm

**REINFORCE** is the simplest policy gradient algorithm. It's a Monte Carlo method — updates happen at the end of each episode using the complete trajectory.

**Key Differences from DQN**:
- Monte Carlo (end of episode) vs Temporal Difference (every step)
- No value function, target network, experience replay, or ε-greedy
- Updates based on actual episode returns, not bootstrap estimates

**REINFORCE Loss Function**:
$$\mathcal{L}(\theta) = -R_\tau \sum_{t=0}^{T} \log \pi_\theta(a_t | s_t)$$

The negative sign converts maximization to minimization (for gradient descent).

```
┌────────────────────────────────────────────────────────────────┐
│                    REINFORCE LOSS FUNCTION                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                              T                                 │
│   L(θ) = -R_τ · Σ log π_θ(a_t|s_t)                            │
│                 t=0                                            │
│           ↑              ↑                                     │
│       Episode       Sum of action                              │
│       return        log probabilities                          │
│                                                                │
│   • Negative because we minimize loss but maximize return      │
│   • Episode return weights how much we adjust probabilities    │
│   • Sum of log probs = log(product of probs) = log P(τ|θ)     │
└────────────────────────────────────────────────────────────────┘
```

**REINFORCE Training Loop**:

```python
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_log_probs = torch.tensor([])
    R = 0  # Episode return
    step = 0
    
    while not done:
        step += 1
        
        # Select action and get log probability
        action, log_prob = select_action(policy_network, state)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Accumulate log probabilities
        episode_log_probs = torch.cat((episode_log_probs, log_prob))
        
        # Accumulate discounted return
        R += (gamma ** step) * reward
        
        state = next_state
    
    # Calculate loss AFTER episode ends (Monte Carlo)
    loss = -R * episode_log_probs.sum()
    
    # Update policy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Limitations of REINFORCE**:
- **High variance**: Using single trajectories to estimate gradients
- **Sample inefficient**: Each experience used only once
- **Only learns at episode end**: Cannot update during long episodes

---

## Chapter 5: Actor-Critic Methods

### 5.1 The Actor-Critic Paradigm

**Motivation**: REINFORCE has high variance and only updates at episode end. Can we get the benefits of policy gradients while using value function estimates for more stable, frequent updates?

**Actor-Critic Solution**: Use two networks:
1. **Actor**: The policy network that decides actions
2. **Critic**: A value network that evaluates how good those actions were

```
┌────────────────────────────────────────────────────────────────┐
│                    ACTOR-CRITIC ARCHITECTURE                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│                      ┌─────────────────┐                       │
│                      │   ENVIRONMENT   │                       │
│                      └───────┬─────────┘                       │
│                              │ state, reward                   │
│                              ▼                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │                                                          │ │
│   │  ┌──────────────┐              ┌──────────────┐         │ │
│   │  │    ACTOR     │              │    CRITIC    │         │ │
│   │  │   (Policy)   │              │   (Value)    │         │ │
│   │  │              │   TD Error   │              │         │ │
│   │  │  π(a|s)      │◄─────────────│    V(s)      │         │ │
│   │  │              │   feedback   │              │         │ │
│   │  └──────┬───────┘              └──────────────┘         │ │
│   │         │                                                │ │
│   │         │ action                                         │ │
│   │         ▼                                                │ │
│   │  ┌─────────────────┐                                     │ │
│   │  │   ENVIRONMENT   │                                     │ │
│   │  └─────────────────┘                                     │ │
│   │                                                          │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                                │
│   Analogy: Student (Actor) and Study Group (Critic)           │
│   • Actor decides what to study and answers questions          │
│   • Critic provides feedback on how well the Actor is doing    │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 The Critic Network

Unlike Q-networks that estimate Q(s,a), the Critic estimates the **state-value function** V(s):

$$V_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]$$

**Key Difference from Q-Network**:
- Q-Network: Input = state, Output = Q-value for each action (4 outputs for 4 actions)
- Critic: Input = state, Output = single scalar value V(s)

```python
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output for state value
    
    def forward(self, state):
        x = torch.relu(self.fc1(torch.tensor(state)))
        value = self.fc2(x)
        return value
```

### 5.3 Advantage Actor-Critic (A2C)

**TD Error as Advantage**: The TD error tells us how much better (or worse) the action was compared to expectation:

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

- $\delta_t > 0$: Action was better than expected → increase its probability
- $\delta_t < 0$: Action was worse than expected → decrease its probability

**A2C Loss Functions**:

**Actor Loss** (Policy):
$$L_{actor}(\theta) = -\log \pi_\theta(a_t|s_t) \cdot \delta_t$$

**Critic Loss** (Value):
$$L_{critic}(\theta_c) = \delta_t^2$$

```
┌────────────────────────────────────────────────────────────────┐
│                        A2C LOSS FUNCTIONS                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Actor Loss:                                                  │
│   ───────────                                                  │
│   L_actor = -log π(a|s) · (r + γV(s') - V(s))                 │
│                              └───────────────┘                 │
│                                  TD Error                      │
│                                 (Advantage)                    │
│                                                                │
│   Intuition:                                                   │
│   • Positive TD error → action was good → increase π(a|s)     │
│   • Negative TD error → action was bad → decrease π(a|s)      │
│                                                                │
│   Critic Loss:                                                 │
│   ────────────                                                 │
│   L_critic = (r + γV(s') - V(s))²                             │
│                                                                │
│   Same as TD error squared — train critic to predict values    │
│   accurately                                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
def calculate_losses(critic_network, action_log_prob, reward, state, next_state, done):
    # Get value estimates
    value = critic_network(state)
    next_value = critic_network(next_state)
    
    # Calculate TD target and TD error
    td_target = reward + gamma * next_value * (1 - done)
    td_error = td_target - value
    
    # Actor loss: negative log prob times advantage
    # detach() prevents gradient from flowing to critic
    actor_loss = -action_log_prob * td_error.detach()
    
    # Critic loss: squared TD error
    critic_loss = td_error ** 2
    
    return actor_loss, critic_loss
```

**A2C Training Loop**:

```python
for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        # Actor selects action
        action, action_log_prob = select_action(actor, state)
        
        # Environment responds
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Calculate losses
        actor_loss, critic_loss = calculate_losses(
            critic, action_log_prob, reward, state, next_state, done
        )
        
        # Update both networks
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        state = next_state
```

**A2C vs REINFORCE**:

| Aspect | REINFORCE | A2C |
|--------|-----------|-----|
| Update timing | End of episode | Every step |
| Value function | None | Critic network |
| Variance | High | Lower (via baseline) |
| Learning speed | Slower | Faster |
| Sample efficiency | Lower | Higher |

---

## Chapter 6: Proximal Policy Optimization (PPO)

### 6.1 The Challenge with A2C

**Problem**: A2C can make large, unstable policy updates because the loss depends on:
- Action probabilities (which change during training)
- Advantage estimates (which are learned and can be noisy)

Large policy changes can harm performance — imagine a Mars rover making sudden directional changes on rough terrain.

**Solution — PPO**: Constrain how much the policy can change in each update.

### 6.2 The Probability Ratio

**Core Innovation**: Track the ratio between the new policy and the old policy:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

- $r_t = 1$: New policy same as old
- $r_t > 1$: New policy more likely to take this action
- $r_t < 1$: New policy less likely to take this action

**Implementation with Log Probabilities**:

```python
def calculate_ratios(action_log_prob, action_log_prob_old, epsilon):
    # Convert to probabilities
    prob = action_log_prob.exp()
    prob_old = action_log_prob_old.exp()
    
    # Detach old probability (treat as constant)
    prob_old_detached = prob_old.detach()
    
    # Calculate ratio
    ratio = prob / prob_old_detached
    
    return ratio
```

### 6.3 Clipping the Probability Ratio

**Key Insight**: Clip the ratio to prevent too-large policy changes:

$$r_t^{clip} = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$$

```
┌────────────────────────────────────────────────────────────────┐
│                    CLIPPING VISUALIZATION                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   r(θ)                                                         │
│    │                                                           │
│ 1.2┤                    ═══════════                            │
│    │                   ╱           (clipped)                   │
│ 1.0┤                  ╱                                        │
│    │                 ╱                                         │
│ 0.8┤═══════════════╱                                           │
│    │     (clipped)                                             │
│    └────────────────────────────────────────────→ actual ratio │
│           0.5   0.8   1.0   1.2   1.5                         │
│                                                                │
│   With ε = 0.2:                                                │
│   • Ratio clipped to range [0.8, 1.2]                         │
│   • Prevents policy from changing more than ±20%               │
└────────────────────────────────────────────────────────────────┘
```

```python
# Clip ratio to [1-epsilon, 1+epsilon]
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
```

### 6.4 The Clipped Surrogate Objective

**PPO Objective Function**:

$$J^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $\hat{A}_t$ is the advantage estimate (TD error).

```
┌────────────────────────────────────────────────────────────────┐
│                  PPO CLIPPED SURROGATE OBJECTIVE               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   J^CLIP = E_t [ min( r(θ)·A , clip(r(θ), 1-ε, 1+ε)·A ) ]     │
│                       ↑              ↑                         │
│                   unclipped       clipped                      │
│                   objective       objective                    │
│                                                                │
│   The minimum ensures:                                         │
│   • When A > 0 (good action): ratio capped at 1+ε             │
│     (don't over-increase probability)                          │
│   • When A < 0 (bad action): ratio floored at 1-ε             │
│     (don't over-decrease probability)                          │
│                                                                │
│   This creates a "trust region" — policy can only change       │
│   within bounds, ensuring stable updates                       │
└────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
def calculate_losses(critic_network, action_log_prob, action_log_prob_old,
                    reward, state, next_state, done, epsilon=0.2):
    # Calculate TD error (advantage)
    value = critic_network(state)
    next_value = critic_network(next_state)
    td_target = reward + gamma * next_value * (1 - done)
    td_error = td_target - value
    
    # Calculate probability ratios
    ratio, clipped_ratio = calculate_ratios(action_log_prob, action_log_prob_old, epsilon)
    
    # Surrogate objectives
    surr1 = ratio * td_error.detach()
    surr2 = clipped_ratio * td_error.detach()
    
    # Clipped surrogate objective (take minimum)
    objective = torch.min(surr1, surr2)
    
    # Negate for loss (we minimize loss, want to maximize objective)
    actor_loss = -objective
    
    # Critic loss unchanged
    critic_loss = td_error ** 2
    
    return actor_loss, critic_loss
```

### 6.5 Entropy Bonus

**Problem**: Policy gradient methods may collapse into deterministic behavior too early, assigning zero probability to some actions prematurely.

**Solution — Entropy Bonus**: Add the policy entropy to the objective to encourage exploration:

**Entropy Definition** (for discrete distributions):
$$H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$$

- High entropy = probability spread across actions (more exploration)
- Low entropy = concentrated on few actions (more exploitation)

```
┌────────────────────────────────────────────────────────────────┐
│                       ENTROPY EXAMPLES                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Distribution: [0.25, 0.25, 0.25, 0.25]                       │
│   Entropy: 2 bits (maximum uncertainty, uniform)               │
│                                                                │
│   Distribution: [0.5, 0.5, 0.0, 0.0]                           │
│   Entropy: 1 bit (like a coin flip)                            │
│                                                                │
│   Distribution: [1.0, 0.0, 0.0, 0.0]                           │
│   Entropy: 0 bits (completely deterministic)                   │
│                                                                │
│   Higher entropy = more exploration                            │
└────────────────────────────────────────────────────────────────┘
```

**Modified Actor Loss with Entropy**:

```python
# Get action distribution and entropy
action_dist = Categorical(action_probs)
entropy = action_dist.entropy()

# Subtract entropy bonus from loss (equivalent to adding to objective)
actor_loss = actor_loss - c_entropy * entropy

# c_entropy typically around 0.01
```

### 6.6 PPO vs A2C Summary

```
┌────────────────────────────────────────────────────────────────┐
│                        PPO vs A2C                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   A2C:                                                         │
│   ────                                                         │
│   • Actor loss = -log π(a|s) · TD_error                       │
│   • Can make arbitrarily large policy changes                  │
│   • More prone to instability                                  │
│                                                                │
│   PPO:                                                         │
│   ────                                                         │
│   • Actor loss = -min(r·A, clip(r)·A)                         │
│   • Policy changes bounded by ε (typically 0.2)               │
│   • More stable training                                       │
│   • Optional entropy bonus for exploration                     │
│                                                                │
│   Key Innovation: Clipping creates a "trust region"            │
│   preventing the new policy from straying too far from         │
│   the old policy in any single update                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Chapter 7: Advanced Training Techniques

### 7.1 Batch Updates

**Problem with Step-by-Step Updates**:
- Single-sample gradients have high variance
- Doesn't leverage parallel processing capabilities
- Can be inefficient

**Solution — Batch Updates**: Accumulate experiences over a "rollout" before updating:

```
┌────────────────────────────────────────────────────────────────┐
│                STEPWISE vs BATCH UPDATES                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   STEPWISE:                                                    │
│   Step 1 → Update → Step 2 → Update → Step 3 → Update → ...   │
│                                                                │
│   BATCH (Rollout length = 3):                                  │
│   Step 1 → Step 2 → Step 3 → Update → Step 4 → Step 5 → ...   │
│          ╲________╱                   ╲________╱               │
│           rollout 1                    rollout 2               │
│                                                                │
│   Benefits:                                                    │
│   • Lower variance (average over multiple samples)             │
│   • Better hardware utilization (batch operations)             │
│   • More stable learning                                       │
└────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```python
actor_losses = torch.tensor([])
critic_losses = torch.tensor([])
batch_size = 10

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    
    while not done:
        action, action_log_prob = select_action(actor, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Calculate losses and accumulate
        actor_loss, critic_loss = calculate_losses(...)
        actor_losses = torch.cat((actor_losses, actor_loss))
        critic_losses = torch.cat((critic_losses, critic_loss))
        
        # Update when batch is full
        if len(actor_losses) >= batch_size:
            # Average loss over batch
            actor_loss_batch = actor_losses.mean()
            critic_loss_batch = critic_losses.mean()
            
            # Gradient descent
            actor_optimizer.zero_grad()
            actor_loss_batch.backward()
            actor_optimizer.step()
            
            critic_optimizer.zero_grad()
            critic_loss_batch.backward()
            critic_optimizer.step()
            
            # Reset accumulators
            actor_losses = torch.tensor([])
            critic_losses = torch.tensor([])
        
        state = next_state
```

### 7.2 Multiple Agents

**Concept**: Run multiple agents with the same policy in parallel to collect diverse experiences:

```
┌────────────────────────────────────────────────────────────────┐
│                    PARALLEL AGENTS                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Environment 1 → Agent 1 → Experiences ╲                      │
│   Environment 2 → Agent 2 → Experiences  ╲                     │
│   Environment 3 → Agent 3 → Experiences   → Aggregate → Update │
│   Environment 4 → Agent 4 → Experiences  ╱                     │
│   Environment 5 → Agent 5 → Experiences ╱                      │
│                                                                │
│   Benefits:                                                    │
│   • More diverse experiences (reduces correlation)             │
│   • Faster data collection                                     │
│   • Better utilization of multi-core CPUs                      │
└────────────────────────────────────────────────────────────────┘
```

### 7.3 Minibatches and Multiple Epochs

**Minibatches**: For large rollouts, divide into smaller minibatches:

```
┌────────────────────────────────────────────────────────────────┐
│                    MINIBATCH PROCESSING                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Rollout (1000 experiences)                                   │
│   │                                                            │
│   │ Shuffle                                                    │
│   ▼                                                            │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐         │
│   │ M1 │ M2 │ M3 │ M4 │ M5 │ M6 │ M7 │ M8 │ M9 │M10 │         │
│   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘         │
│    100   100  100  100  100  100  100  100  100  100           │
│     │     │    │    │    │    │    │    │    │    │            │
│     ▼     ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼            │
│   Update Update ... (10 updates per rollout)                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Multiple Epochs**: PPO can reuse the same rollout data multiple times:

```
┌────────────────────────────────────────────────────────────────┐
│                    MULTIPLE EPOCHS (PPO)                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Rollout → Epoch 1 → Epoch 2 → Epoch 3 → ... → New Rollout   │
│             (shuffle)  (shuffle)  (shuffle)                    │
│                                                                │
│   PPO can do this because:                                     │
│   • Clipped objective prevents policy from diverging too far   │
│   • Data remains "on-policy enough" for a few epochs          │
│                                                                │
│   A2C cannot do this effectively because:                      │
│   • No clipping → updated policy diverges from data source    │
│   • Becomes "off-policy" quickly                              │
│                                                                │
│   This makes PPO more SAMPLE EFFICIENT                         │
└────────────────────────────────────────────────────────────────┘
```

---

## Chapter 8: Hyperparameter Optimization

### 8.1 Key Hyperparameters in DRL

| Category | Hyperparameter | Typical Range | Description |
|----------|----------------|---------------|-------------|
| **General** | γ (discount) | 0.95 - 0.99 | How much to value future rewards |
| | Learning rate | 1e-5 - 1e-3 | Step size for gradient descent |
| **DQN** | Buffer size | 10K - 1M | Experience replay capacity |
| | Batch size | 32 - 256 | Samples per update |
| | ε decay | 500 - 10000 | Exploration schedule |
| | τ (soft update) | 0.001 - 0.01 | Target network update rate |
| **PPO** | ε (clip) | 0.1 - 0.3 | Policy change bound |
| | c_entropy | 0.001 - 0.1 | Entropy bonus weight |
| **Architecture** | Hidden layers | 1 - 3 | Network depth |
| | Hidden nodes | 32 - 512 | Network width |

### 8.2 Optuna for Hyperparameter Search

**Optuna** is a powerful hyperparameter optimization framework:

```python
import optuna

def objective(trial: optuna.Trial):
    # Define hyperparameter search space
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # Train agent with these hyperparameters
    agent = train_agent(gamma=gamma, lr=lr, batch_size=batch_size)
    
    # Return metric to optimize (e.g., average reward)
    return evaluate_agent(agent)

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Best hyperparameters
print(study.best_params)
```

**Optuna Features**:
- **Intelligent sampling**: Uses TPE (Tree-structured Parzen Estimator) by default
- **Pruning**: Early stopping of unpromising trials
- **Visualization**: Built-in plots for analysis
- **Persistence**: Save/load studies for continuation

---

## Chapter 9: Advanced DRL Methods for Finance

This chapter provides a brief overview of more sophisticated DRL methods that are particularly relevant for quantitative finance applications.

### 9.1 Deep Deterministic Policy Gradient (DDPG)

**Use Case**: Portfolio optimization, continuous trading strategies

**Key Innovation**: Extends DQN to **continuous action spaces** (e.g., exact position sizes, not just buy/sell/hold)

**Architecture**:
- **Actor network**: Outputs continuous action directly (e.g., portfolio weights)
- **Critic network**: Estimates Q(s, a) for continuous actions
- Uses **target networks** for both actor and critic
- Employs **Ornstein-Uhlenbeck noise** for exploration

**Why It Matters for Finance**:
- Can output exact portfolio allocations (e.g., 23.5% in asset A)
- Natural for market making with continuous bid-ask spreads
- Better for hedging with precise position sizing

```
┌────────────────────────────────────────────────────────────────┐
│                         DDPG OVERVIEW                          │
├────────────────────────────────────────────────────────────────┤
│   DQN:  State → Q-values for discrete actions → argmax        │
│   DDPG: State → Actor → Continuous action                      │
│               → Critic(state, action) → Q-value                │
│                                                                │
│   Finance Application: Portfolio weights as continuous actions │
└────────────────────────────────────────────────────────────────┘
```

### 9.2 Soft Actor-Critic (SAC)

**Use Case**: Risk-adjusted trading, robust strategies

**Key Innovation**: Maximizes **entropy-regularized return**:
$$J(\pi) = \sum_t \mathbb{E}\left[r_t + \alpha H(\pi(\cdot|s_t))\right]$$

**Main Differences from DDPG**:
- **Stochastic policy** (vs deterministic in DDPG)
- **Automatic entropy tuning** (learns optimal exploration level)
- **Twin critics** (two Q-networks to reduce overestimation)
- **Better sample efficiency** and more robust

**Why It Matters for Finance**:
- Natural handling of uncertainty in markets
- Built-in exploration prevents overconfident strategies
- More robust to changing market regimes
- Better risk-adjusted performance in practice

### 9.3 Multi-Agent Reinforcement Learning (MARL)

**Use Case**: Market simulation, game-theoretic trading, competing strategies

**Key Concepts**:
- Multiple agents interact in shared environment
- Each agent's optimal policy depends on others' policies
- Can model **competitive** (zero-sum) or **cooperative** settings

**Applications in Finance**:
- **Market microstructure simulation**: Model interactions between traders
- **Adversarial testing**: Train against adaptive adversaries
- **Order flow modeling**: Multiple agents placing orders
- **Regulatory impact analysis**: How do rule changes affect all participants?

```
┌────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT IN FINANCE                      │
├────────────────────────────────────────────────────────────────┤
│   Single Agent: Learn optimal policy assuming static market    │
│   Multi-Agent:  Learn while other agents also learn/adapt      │
│                                                                │
│   Example: Training a market-making algorithm that must        │
│   compete against other market makers (also learning)          │
└────────────────────────────────────────────────────────────────┘
```

### 9.4 Model-Based Reinforcement Learning

**Use Case**: Sample-efficient trading, when real market interaction is costly

**Key Innovation**: Learn a **model of the environment** (market dynamics), then plan using that model

**Approaches**:
- **Dyna-style**: Interleave real experience with simulated experience from learned model
- **Model Predictive Control**: Plan actions by simulating future trajectories
- **World Models**: Learn compressed representations of environment dynamics

**Why It Matters for Finance**:
- **Sample efficiency**: Markets are expensive to interact with (transaction costs, slippage)
- **Risk management**: Test strategies in simulated model before real trading
- **Scenario analysis**: Generate "what-if" scenarios for stress testing

**Trade-off**: Model errors can compound; less robust than model-free methods

### 9.5 Reinforcement Learning from Human Feedback (RLHF)

**Use Case**: Aligning trading strategies with human preferences, interpretable AI

**Key Innovation**: Train a **reward model** from human feedback, then optimize policy using that reward

**Process**:
1. Generate candidate strategies/decisions
2. Humans rank or rate them
3. Train reward model to predict human preferences
4. Use reward model to train policy via RL

**Applications in Finance**:
- **Strategy alignment**: Ensure trading behavior matches firm's risk appetite
- **Compliance**: Train agents to follow regulatory preferences
- **Client-specific optimization**: Learn individual investor preferences
- **Explainability**: Reward model can explain why certain actions are preferred

```
┌────────────────────────────────────────────────────────────────┐
│                      RLHF IN FINANCE                           │
├────────────────────────────────────────────────────────────────┤
│   Traditional RL: Maximize returns                             │
│   RLHF: Maximize alignment with human-specified preferences    │
│                                                                │
│   Example: A portfolio manager prefers smooth equity curves    │
│   over higher returns with large drawdowns. RLHF can learn     │
│   this preference from their feedback on historical trades.    │
└────────────────────────────────────────────────────────────────┘
```

### 9.6 Summary: Choosing the Right Method

| Method | Best For | Key Advantage | Limitation |
|--------|----------|---------------|------------|
| **DQN/DDQN** | Discrete decisions | Simple, well-understood | Can't handle continuous actions |
| **PPO** | General purpose | Stable, reliable | May be less sample-efficient |
| **DDPG** | Continuous control | Exact position sizing | Sensitive to hyperparameters |
| **SAC** | Robust strategies | Better exploration, risk-aware | More complex |
| **MARL** | Competitive settings | Models market dynamics | Computationally expensive |
| **Model-Based** | Limited data | Sample efficient | Model errors accumulate |
| **RLHF** | Preference alignment | Incorporates human judgment | Requires human feedback |

---

## Key Takeaways

1. **DRL combines RL with neural networks** to handle high-dimensional problems that traditional RL cannot solve.

2. **Value-based methods** (DQN family) learn Q-values and derive policies; **policy-based methods** learn policies directly.

3. **Experience replay, target networks, and double Q-learning** address key instabilities in DQN.

4. **Policy gradient methods** (REINFORCE, A2C) directly optimize expected return and naturally handle stochastic policies.

5. **PPO's clipped objective** provides stable policy updates, making it a reliable default choice.

6. **Batch updates and parallel agents** improve sample efficiency and training stability.

7. **For finance applications**, consider continuous action methods (DDPG, SAC) for position sizing, multi-agent methods for market simulation, and model-based methods for sample efficiency.

---

*Notes compiled from Deep Reinforcement Learning course materials*
