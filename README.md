# Multi-Armed Bandit Problem: Analyzing Action Selection Strategies

This repository contains the implementation of various action-selection methods to solve the **Multi-Armed Bandit Problem**, a fundamental challenge in reinforcement learning. The code simulates multiple strategies, including Greedy, Epsilon-Greedy, Upper-Confidence-Bound (UCB), Optimistic Initial Values, and Gradient Bandit Algorithms. Performance comparisons are drawn to demonstrate the strengths and weaknesses of each method in terms of average rewards and optimal action selection.

For a detailed explanation of these strategies and the theoretical background, refer to the [accompanying Medium article](https://medium.com/@david-de-villiers/mastering-the-multi-armed-bandit-problem-in-reinforcement-learning-b4206700b779) or the [YouTube Video](https://www.youtube.com/watch?v=bnyA97_J9H0).

## Overview

The **Multi-Armed Bandit Problem** is a simplified reinforcement learning task that requires balancing **exploration** (trying new actions to gather information) and **exploitation** (choosing actions known to yield high rewards). Each action has an unknown reward distribution, and the goal is to maximize cumulative reward over a sequence of action choices.

### Implemented Agents:
1. **Greedy Agent** - Always selects the action with the highest estimated value.
2. **Epsilon-Greedy Agent** - Selects a random action with probability $\epsilon$ and exploits the best-known action otherwise.
3. **UCB (Upper Confidence Bound) Agent** - Selects actions based on their estimated value and an upper-confidence bound to encourage exploration.
4. **Optimistic Initial Values Agent** - Starts with inflated action-value estimates to encourage early exploration.
5. **Gradient Bandit Agent** - Learns preferences over actions using a softmax probability distribution.

## Project Structure
```
├── bandit.py                 # Bandit class defining the multi-armed bandit environment.
├── agents/                   # Directory containing implementations of different bandit agents.
│   ├── greedy_agent.py       # Greedy agent.
│   ├── epsilon_greedy_agent.py  # Epsilon-Greedy agent.
│   ├── ucb_agent.py          # UCB agent.
│   ├── optimistic_agent.py   # Optimistic Initial Values agent.
│   └── gradient_bandit_agent.py  # Gradient Bandit agent.
├── main.py                   # Main script to run simulations and visualize results.
├── performance_comparison.py # Compares performance of different methods over different parameter values
└── README.md                 # Project documentation.
```

## Getting Started

### Prerequisites
The following packages are required to run the project:

- **Python 3.6+**
- **NumPy**
- **Matplotlib**
- **tqdm**

You can install the necessary packages using:

```bash
pip install numpy matplotlib tqdm
```

### Running the Code
The main entry point is `main.py`. To run the simulations and visualize the results:
```bash
python main.py
```

The script runs simulations for 1000 steps over 1000 independent runs and outputs two primary plots:
- **Average Reward vs. Steps**: Shows the average reward for each method over time.
- **Percentage of Optimal Action vs. Steps**: Illustrates the percentage of time the optimal action was selected for each method.

Be sure to comment out any agents you do not want to run!

### Simulation Parameters
You can adjust the simulation parameters (e.g., number of steps, runs, agent parameters) in the main.py file:
```
# Simulation parameters
n_steps = 1000
n_runs = 1000

# Agent-specific parameters
epsilons = [0.1]  # Epsilon values for the Epsilon-Greedy agent
c = 2             # UCB parameter
initial_value = 5 # Optimistic initial values
alpha = 0.1       # Gradient Bandit step size
```

### Results and Analysis
The performance of each agent is evaluated using the following metrics:
- **Average Reward**: How well each agent maximizes rewards over time.
- **Optimal Action Selection Percentage**: How frequently the agent selects the optimal action.

For detailed interpretations of the results, refer to the [Medium Article](https://medium.com/@david-de-villiers/mastering-the-multi-armed-bandit-problem-in-reinforcement-learning-b4206700b779).

## License
This project is licensed under the MIT License. Feel free to modify and use it for educational purposes.
