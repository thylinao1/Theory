# Reinforcement Learning with Gymnasium in Python

## Table of Contents

1. [Fundamentals of Reinforcement Learning](#1-fundamentals-of-reinforcement-learning)
2. [The RL Framework](#2-the-rl-framework)
3. [Gymnasium Environments](#3-gymnasium-environments)
4. [Markov Decision Processes (MDPs)](#4-markov-decision-processes-mdps)
5. [Policies and Value Functions](#5-policies-and-value-functions)
6. [Model-Based Methods](#6-model-based-methods)
7. [Model-Free Methods](#7-model-free-methods)
8. [Exploration vs Exploitation](#8-exploration-vs-exploitation)
9. [Multi-Armed Bandits](#9-multi-armed-bandits)
10. [Comparison of Learning Approaches](#10-comparison-of-learning-approaches)

---

## 1. Fundamentals of Reinforcement Learning

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a unique branch of machine learning where an **agent** learns to make decisions through **trial and error**. Unlike supervised or unsupervised learning, RL involves an agent that observes and acts within an environment, receiving rewards for good decisions and penalties for bad ones. The agent's goal is to devise a strategy that maximizes positive feedback over time.

**Analogy**: Think of RL as training a pet. Just as you would reward your pet for following a command correctly, in RL, an agent receives rewards for making correct decisions. The process is iterative and based on trial and error, much like how a pet learns from repeated training sessions.

### RL vs Other Machine Learning Types

| Aspect | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|--------|--------------------|-----------------------|------------------------|
| **Training Data** | Labeled data | Unlabeled data | No training data |
| **Learning Method** | Predict outcomes from examples | Identify patterns/structures | Trial and error |
| **Suitable For** | Classification, Regression | Clustering, Association | Decision-making tasks |
| **Feedback** | Direct labels | None | Rewards/penalties |

### When to Use RL

RL is well-suited for scenarios requiring:
- **Sequential decision-making** where each decision influences future observations
- **Learning through rewards and penalties** without direct supervision
- **Strategy development** for complex, dynamic environments

**Appropriate Example**: Playing video games where the player makes sequential decisions (jumping, avoiding enemies) and learns through points (rewards) and losing lives (penalties).

**Inappropriate Example**: In-game object recognition, which doesn't involve sequential decision-making—supervised learning with labeled data works better here.

### RL Applications

RL has broad applications across various sectors:
- **Robotics**: Teaching robots tasks through trial and error (walking, object manipulation)
- **Finance**: Optimizing trading and investment strategies to maximize profit
- **Autonomous Vehicles**: Enhancing safety and efficiency of self-driving cars
- **Chatbots**: Improving conversational skills and response accuracy over time

---

## 2. The RL Framework

### Core Components

The RL framework consists of five key components:

1. **Agent**: The learner or decision-maker (like a player in a game)
2. **Environment**: The world the agent interacts with, presenting challenges to solve
3. **State**: A specific moment in time capturing the current situation (like a video game frame)
4. **Action**: The agent's response to a state
5. **Reward**: Feedback from the environment—positive to encourage, negative to discourage behaviors

### The Interaction Loop

```python
# Generic RL interaction loop
env = create_environment()
state = env.reset()

while not done:
    action = agent.select_action(state)  # Agent chooses action based on state
    next_state, reward, done = env.step(action)  # Environment provides feedback
    agent.update(state, action, reward)  # Agent learns from experience
    state = next_state
```

### Episodic vs Continuous Tasks

**Episodic Tasks**: Divided into distinct episodes with defined beginnings and ends. Example: Each chess game is an episode—once concluded, the environment resets for the next game.

**Continuous Tasks**: Ongoing interaction without distinct episodes. Example: An agent continuously adjusting traffic lights to optimize flow—the task never truly "ends."

### Return and Discounted Return

**Return**: The sum of all rewards the agent expects to accumulate throughout its journey. The agent learns to anticipate action sequences yielding the highest possible return.

**Discounted Return**: Prioritizes recent rewards over future ones by multiplying each reward by a discount factor (γ) raised to its time step:

$$G_t = r_1 + \gamma r_2 + \gamma^2 r_3 + ... + \gamma^{n-1} r_n$$

**Discount Factor (γ)**: Ranges between 0 and 1:
- **γ close to 0**: Agent prioritizes immediate gains
- **γ close to 1**: Agent values long-term benefits equally
- **γ = 0**: Focus solely on immediate rewards
- **γ = 1**: No discount applied (all rewards equally valued)

```python
import numpy as np

expected_rewards = np.array([3, 2, -1, 5])
discount_factor = 0.9

# Compute discounts for each time step
discounts = np.array([discount_factor**i for i in range(len(expected_rewards))])

# Calculate discounted return
discounted_return = np.sum(expected_rewards * discounts)
print(f"Discounted return: {discounted_return}")  # Output: 8.83
```

---

## 3. Gymnasium Environments

### Overview

Gymnasium provides standardized environments for RL tasks, abstracting the complexity of defining RL problems and enabling focus on algorithm development. It offers a plethora of environments from classic control tasks to complex Atari games.

### Key Environments

| Environment | Description |
|-------------|-------------|
| **CartPole** | Keep a pole balanced on a moving cart |
| **MountainCar** | Drive up a steep hill by building momentum |
| **FrozenLake** | Find a safe path across a hole-filled grid |
| **Taxi** | Pick up and drop off passengers efficiently |
| **CliffWalking** | Navigate from start to goal avoiding cliffs |

### Creating and Using Environments

```python
import gymnasium as gym

# Create environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Initialize and get initial state
initial_state, info = env.reset(seed=42)

# Visualize the environment
def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()
```

### Performing Actions

```python
# Actions in CartPole: 0 = left, 1 = right
action = 1  # Move right

# Execute action and receive feedback
next_state, reward, terminated, truncated, info = env.step(action)
```

The `env.step()` method returns five values:
- **next_state**: The new state after the action
- **reward**: Immediate reward received
- **terminated**: Whether the agent reached a terminal state (goal or failure)
- **truncated**: Whether a condition like time limit was met
- **info**: Auxiliary diagnostic information

### State and Action Spaces

```python
# Get number of possible actions
num_actions = env.action_space.n  # e.g., 4 for FrozenLake

# Get number of possible states
num_states = env.observation_space.n  # e.g., 16 for 4x4 FrozenLake
```

---

## 4. Markov Decision Processes (MDPs)

### Definition

An MDP provides a mathematical framework for modeling RL environments. It defines four key components:

1. **States (S)**: All possible situations the agent can be in
2. **Actions (A)**: All possible moves the agent can make
3. **Rewards (R)**: Immediate feedback for state-action pairs
4. **Transition Probabilities (P)**: Likelihood of moving from one state to another given an action

### The Markov Property

The future state depends **only on the current state and action**, not on previous events. Like chess: the next position is determined by the current arrangement and move made, not move history.

### Frozen Lake as MDP Example

**States**: Grid positions (16 states for a 4×4 grid)  
**Terminal States**: Goal (reward) or holes (failure)  
**Actions**: Up (3), Down (1), Left (0), Right (2)  
**Transitions**: With `is_slippery=True`, actions may result in unintended movements

### Accessing MDP Components in Gymnasium

```python
env = gym.make('FrozenLake-v1', is_slippery=True)

# Access transition probabilities and rewards
# env.unwrapped.P[state][action] returns list of (probability, next_state, reward, done)
transitions = env.unwrapped.P[6][0]  # From state 6, taking action 0 (left)

for prob, next_state, reward, done in transitions:
    print(f"P={prob:.2f}: next_state={next_state}, reward={reward}, terminal={done}")
```

---

## 5. Policies and Value Functions

### Policies

A **policy** is a roadmap guiding the agent by specifying optimal actions in each state to maximize return. It maps states to actions.

```python
# Define a deterministic policy
policy = {
    0: 2,  # In state 0, go right
    1: 2,  # In state 1, go right
    2: 1,  # In state 2, go down
    # ... etc.
}
```

### State-Value Functions V(s)

The **state-value function** V(s) calculates the expected discounted return starting from state s and following a policy:

$$V^\pi(s) = E[G_t | S_t = s]$$

```python
def compute_state_value(state, policy, terminal_state, gamma=1.0):
    if state == terminal_state:
        return 0
    action = policy[state]
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state, policy, terminal_state, gamma)

# Compute all state values
state_values = {s: compute_state_value(s, policy, terminal_state) for s in range(num_states)}
```

### Action-Value Functions Q(s, a)

The **action-value function** (Q-value) estimates the expected return of starting in a state, taking a certain action, then following a policy:

$$Q^\pi(s, a) = r(s, a) + \gamma V^\pi(s')$$

```python
def compute_q_value(state, action, policy, terminal_state, gamma=1.0):
    if state == terminal_state:
        return None
    _, next_state, reward, _ = env.unwrapped.P[state][action][0]
    if next_state == terminal_state:
        return reward
    return reward + gamma * compute_state_value(next_state, policy, terminal_state, gamma)
```

### The Bellman Equation

A recursive formula connecting each state's value to its successors:

$$V(s) = r + \gamma V(s')$$

For deterministic environments, this standard formula suffices. Non-deterministic environments require modifications to incorporate transition probabilities.

### Policy Improvement

Using Q-values, we can improve a policy by selecting the action with the highest Q-value for each state:

```python
improved_policy = {}
for state in range(num_states - 1):
    max_action = max(range(num_actions), key=lambda a: Q[(state, a)])
    improved_policy[state] = max_action
```

---

## 6. Model-Based Methods

Model-based methods optimize policies or value functions using knowledge of the environment's dynamics (transition probabilities and rewards) **without requiring direct environment interaction**.

### Policy Iteration

An iterative process alternating between policy evaluation and policy improvement until convergence:

1. **Start** with a random policy
2. **Evaluate**: Compute state-value function for current policy
3. **Improve**: Choose actions maximizing state-value at each state
4. **Repeat** until policy stabilizes (optimal policy found)

```python
def policy_iteration():
    policy = {0: 2, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0, 6: 2, 7: 2}  # Initial random policy
    
    while True:
        # Policy Evaluation
        V = {state: compute_state_value(state, policy) for state in range(num_states)}
        
        # Policy Improvement
        Q = {(s, a): compute_q_value(s, a, policy) 
             for s in range(num_states) for a in range(num_actions)}
        
        improved_policy = {}
        for state in range(num_states - 1):
            max_action = max(range(num_actions), key=lambda a: Q[(state, a)])
            improved_policy[state] = max_action
        
        # Check convergence
        if improved_policy == policy:
            break
        policy = improved_policy
    
    return policy, V
```

### Value Iteration

A more efficient algorithm combining policy evaluation and improvement into a single step:

1. **Initialize** value function V (typically zeros)
2. **Update** each state's value by considering maximum expected return from any action
3. **Repeat** until changes are below a threshold
4. **Derive policy** from final value function

```python
def value_iteration(threshold=0.001):
    V = {s: 0 for s in range(num_states)}
    policy = {s: 0 for s in range(num_states - 1)}
    
    while True:
        new_V = {}
        for state in range(num_states - 1):
            max_action, max_q_value = get_max_action_and_value(state, V)
            new_V[state] = max_q_value
            policy[state] = max_action
        
        # Check convergence
        if all(abs(new_V[s] - V[s]) < threshold for s in range(num_states - 1)):
            break
        V = new_V
    
    return policy, V
```

---

## 7. Model-Free Methods

Model-free methods learn optimal policies through **direct environment interaction** without requiring knowledge of transition probabilities or rewards beforehand.

### Monte Carlo Methods

Estimate Q-values and derive policies from **complete episodes** of experience:

1. **Collect** several episodes with random actions
2. **Estimate** Q-values from episode returns
3. **Derive** optimal policy by choosing highest Q-value actions

#### First-Visit Monte Carlo

Averages returns only from the **first occurrence** of each state-action pair in an episode:

```python
def first_visit_mc(num_episodes):
    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))
    
    for _ in range(num_episodes):
        episode = generate_episode()
        visited = set()
        
        for j, (state, action, reward) in enumerate(episode):
            if (state, action) not in visited:
                # Calculate return from this point
                G = sum([x[2] for x in episode[j:]])
                returns_sum[state, action] += G
                returns_count[state, action] += 1
                visited.add((state, action))
        
        # Update Q-values
        nonzero = returns_count != 0
        Q[nonzero] = returns_sum[nonzero] / returns_count[nonzero]
    
    return Q
```

#### Every-Visit Monte Carlo

Averages returns from **every occurrence** of each state-action pair:

```python
def every_visit_mc(num_episodes):
    Q = np.zeros((num_states, num_actions))
    returns_sum = np.zeros((num_states, num_actions))
    returns_count = np.zeros((num_states, num_actions))
    
    for _ in range(num_episodes):
        episode = generate_episode()
        
        for j, (state, action, reward) in enumerate(episode):
            G = sum([x[2] for x in episode[j:]])
            returns_sum[state, action] += G
            returns_count[state, action] += 1
        
        nonzero = returns_count != 0
        Q[nonzero] = returns_sum[nonzero] / returns_count[nonzero]
    
    return Q
```

### Temporal Difference (TD) Learning

Updates value estimates at **each step** within an episode based on the most recent experience, making it more flexible than Monte Carlo methods.

**Analogy**: Think of TD learning as weather forecasting, where predictions are constantly updated as new data comes in, rather than waiting for the whole day's outcome.

#### SARSA (State-Action-Reward-State-Action)

An **on-policy** TD method that learns the value of the policy currently being followed:

$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma Q(s', a')]$$

```python
def sarsa_update(state, action, reward, next_state, next_action, alpha=0.1, gamma=0.99):
    old_value = Q[state, action]
    next_value = Q[next_state, next_action]
    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    action = env.action_space.sample()
    terminated = False
    
    while not terminated:
        next_state, reward, terminated, _, _ = env.step(action)
        next_action = env.action_space.sample()  # On-policy: use same strategy
        sarsa_update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
```

#### Q-Learning

An **off-policy** TD method that learns the optimal Q-table by considering the maximum possible Q-value from the next state, regardless of the action actually taken:

$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a')]$$

```python
def q_learning_update(state, action, reward, next_state, alpha=0.1, gamma=0.99):
    old_value = Q[state, action]
    next_max = np.max(Q[next_state])  # Key difference: use max Q-value
    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        action = env.action_space.sample()
        next_state, reward, terminated, _, _ = env.step(action)
        q_learning_update(state, action, reward, next_state)
        state = next_state
```

#### Expected SARSA

A TD method that uses the **expected value** of the next state based on all possible actions:

$$Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma E[Q(s', A)]]$$

```python
def expected_sarsa_update(state, action, next_state, reward, alpha=0.1, gamma=0.99):
    expected_q = np.mean(Q[next_state])  # Average over all actions
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * expected_q)
```

This approach is more robust to changes and uncertainties as it considers the average outcome of all possible next actions.

#### Double Q-Learning

Addresses Q-learning's tendency to **overestimate Q-values** by maintaining two separate Q-tables that update each other:

```python
Q = [np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))]

def double_q_update(state, action, reward, next_state, alpha=0.1, gamma=0.99):
    i = np.random.randint(2)  # Randomly choose which table to update
    best_next_action = np.argmax(Q[i][next_state])
    # Update Q[i] using Q[1-i]'s estimate
    Q[i][state, action] = (1 - alpha) * Q[i][state, action] + \
                          alpha * (reward + gamma * Q[1-i][next_state, best_next_action])

# After training, combine tables
final_Q = Q[0] + Q[1]
```

---

## 8. Exploration vs Exploitation

### The Trade-off

**Exploration**: Trying new actions to discover potentially better rewards  
**Exploitation**: Using current knowledge to maximize immediate rewards

Continuous exploration prevents strategy refinement, while exclusive exploitation risks missing undiscovered opportunities.

**Analogy**: Like choosing where to eat—exploration is trying a new restaurant, exploitation is going to your favorite. Both are important, but balance is key.

### Epsilon-Greedy Strategy

Choose a random action with probability ε, otherwise choose the best-known action:

```python
def epsilon_greedy(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])  # Exploit
```

### Decayed Epsilon-Greedy Strategy

Gradually reduce ε over time—more exploration early, more exploitation later:

```python
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

for episode in range(num_episodes):
    # ... training loop using epsilon_greedy ...
    
    # Decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
```

---

## 9. Multi-Armed Bandits

### The Problem

A classic RL problem where an agent faces multiple "slot machines" (arms), each with unknown probability of winning. The goal: maximize winnings by deciding which arm to play and when to switch.

This perfectly encapsulates the exploration-exploitation trade-off without the complexity of state transitions.

### Implementation

```python
def create_multi_armed_bandit(n_bandits, n_iterations):
    true_probs = np.random.rand(n_bandits)  # Unknown to agent
    counts = np.zeros(n_bandits)  # Times each arm was pulled
    values = np.zeros(n_bandits)  # Estimated winning probability
    rewards = np.zeros(n_iterations)
    selected_arms = np.zeros(n_iterations, dtype=int)
    return true_probs, counts, values, rewards, selected_arms

# Solving with decayed epsilon-greedy
for i in range(n_iterations):
    arm = epsilon_greedy_bandit(values, epsilon)
    reward = np.random.rand() < true_probs[arm]  # Binary reward
    
    # Update estimates incrementally
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
```

### Analysis

Track selection percentages over iterations to observe how the agent learns to favor higher-probability arms:

```python
selections = np.zeros((n_iterations, n_bandits))
for i in range(n_iterations):
    selections[i, selected_arms[i]] = 1

# Cumulative selection percentages
cumulative_selections = np.cumsum(selections, axis=0) / np.arange(1, n_iterations+1).reshape(-1, 1)
```

---

## 10. Comparison of Learning Approaches

### Model-Based vs Model-Free

| Aspect | Model-Based | Model-Free |
|--------|-------------|------------|
| **Environment Knowledge** | Requires transition probabilities and rewards | Learns from direct interaction |
| **Data Efficiency** | More efficient (no sampling needed) | Less efficient (needs many samples) |
| **Computational Cost** | Can be high for complex models | Varies by algorithm |
| **Adaptability** | Limited to known dynamics | Highly adaptable to unknown environments |
| **Examples** | Policy Iteration, Value Iteration | Monte Carlo, TD Learning |

### TD Learning Algorithms Comparison

| Algorithm | Type | Update Target | Key Characteristic |
|-----------|------|---------------|-------------------|
| **SARSA** | On-policy | Q(s', a') where a' is the actual next action | Learns value of current policy; more conservative |
| **Q-Learning** | Off-policy | max Q(s', a') for all a' | Learns optimal policy directly; can overestimate |
| **Expected SARSA** | Off-policy | E[Q(s', A)] average over all actions | More stable; robust to uncertainties |
| **Double Q-Learning** | Off-policy | Uses two Q-tables to reduce overestimation | Most stable; less bias |

### Monte Carlo vs Temporal Difference

| Aspect | Monte Carlo | Temporal Difference |
|--------|-------------|---------------------|
| **Update Timing** | After episode completion | After each step |
| **Episode Requirement** | Needs complete episodes | Works with incomplete episodes |
| **Variance** | Higher (uses actual returns) | Lower (uses bootstrapping) |
| **Bias** | Unbiased | Can have bias |
| **Best For** | Short, well-defined episodes | Long or continuous tasks |
| **Learning Speed** | Slower (waits for episode end) | Faster (immediate updates) |

### Comprehensive Algorithm Comparison

| Algorithm | Model Requirement | Update Frequency | Exploration Strategy | Best Use Case |
|-----------|------------------|------------------|---------------------|---------------|
| **Policy Iteration** | Model-based | Per iteration | N/A (uses complete model) | Known, small MDPs |
| **Value Iteration** | Model-based | Per iteration | N/A (uses complete model) | Known, small MDPs |
| **First-Visit MC** | Model-free | Per episode | Random or ε-greedy | Episodic tasks with clear termination |
| **Every-Visit MC** | Model-free | Per episode | Random or ε-greedy | Episodic tasks needing all samples |
| **SARSA** | Model-free | Per step | ε-greedy (follows policy) | Safe exploration needed |
| **Q-Learning** | Model-free | Per step | ε-greedy (optimal) | Maximum reward desired |
| **Expected SARSA** | Model-free | Per step | ε-greedy | Stable learning in uncertain environments |
| **Double Q-Learning** | Model-free | Per step | ε-greedy | Preventing overestimation bias |

### Key Formulas Summary

| Method | Update Rule |
|--------|-------------|
| **Bellman (State-Value)** | V(s) = r + γV(s') |
| **Bellman (Action-Value)** | Q(s,a) = r + γQ(s',a') |
| **SARSA** | Q(s,a) ← (1-α)Q(s,a) + α[r + γQ(s',a')] |
| **Q-Learning** | Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',·)] |
| **Expected SARSA** | Q(s,a) ← (1-α)Q(s,a) + α[r + γ E{Q(s',·)}] |
| **Double Q-Learning** | Q_i(s,a) ← (1-α)Q_i(s,a) + α[r + γ Q_{1-i}(s', argmax Q_i(s',·))] |

---

## Quick Reference: When to Use Each Approach

**Use Model-Based Methods (Policy/Value Iteration) when:**
- You have complete knowledge of environment dynamics
- The state space is small enough to enumerate
- Computational resources for planning are available

**Use Monte Carlo Methods when:**
- Episodes are well-defined and relatively short
- You need unbiased value estimates
- The environment is episodic

**Use SARSA when:**
- You need safe, conservative learning
- The agent must learn from the policy it's following
- Risk-sensitive applications

**Use Q-Learning when:**
- You want to learn the optimal policy directly
- Sample efficiency is important
- You can tolerate some overestimation

**Use Expected SARSA when:**
- You need stability in learning
- The environment has high variance
- You want benefits of both SARSA and Q-learning

**Use Double Q-Learning when:**
- Overestimation bias is a concern
- You need the most stable learning
- Complex environments with many similar-valued actions

---

## Hyperparameters Reference

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| **Learning Rate** | α | 0.01 - 0.5 | Controls speed of Q-value updates |
| **Discount Factor** | γ | 0.9 - 0.99 | Balances immediate vs future rewards |
| **Exploration Rate** | ε | 0.1 - 1.0 | Controls exploration vs exploitation |
| **Epsilon Decay** | - | 0.99 - 0.999 | Rate at which exploration decreases |
| **Min Epsilon** | - | 0.01 - 0.1 | Minimum exploration maintained |
| **Episodes** | - | 100 - 10000+ | Training iterations |

---

*Notes compiled from DataCamp's "Reinforcement Learning with Gymnasium in Python" course*
