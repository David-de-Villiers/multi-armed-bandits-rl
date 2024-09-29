import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from tqdm import tqdm

# Bandit agents
from agents.greedy_agent import GreedyAgent
from agents.epsilon_greedy_agent import EpsilonGreedyAgent
from agents.ucb_agent import UCBAgent
from agents.optimistic_agent import OptimisticAgent
from agents.gradient_bandit_agent import GradientBanditAgent


# LaTeX-style rendering for plots
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for text rendering
    "font.family": "serif",  # Use serif fonts by default
    "font.serif": ["Computer Modern"],  # Match LaTeX's default font
    "axes.labelsize": 14,  # Set axis label font size
    "font.size": 14,  # Set general font size
    "legend.fontsize": 12,  # Set legend font size
    "xtick.labelsize": 12,  # Set x-axis tick label font size
    "ytick.labelsize": 12,  # Set y-axis tick label font size
})

def run_bandit(current_agent, steps=1000, runs=1000):
    """
    Simulate agent interaction with bandit.

    :param current_agent: Function that returns an agent instance when passed a bandit.
    :param steps: Number of steps to take per run.
    :param runs: Number of runs to average over.

    :return: average_rewards, average rewards over time.
    :return: optimal_action_counts, percentage of optimal action selections.
    """
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    # Run simulations
    for run in tqdm(range(runs), desc="Running simulations", unit="run"):
        bandit = Bandit(k=10)
        agent = current_agent(bandit)

        for step in range(steps):
            action = agent.select_action()
            reward = bandit.pull_lever(action)
            agent.update_estimates(action, reward)

            # Record rewards and optimal actions for visualization
            rewards[run, step] = reward
            if action == bandit.optimal_action:
                optimal_actions[run, step] = 1

    average_rewards = rewards.mean(axis=0)
    optimal_action_counts = optimal_actions.mean(axis=0) * 100
    reward_confidence = (1.96 * rewards.std(axis=0)) / np.sqrt(runs)
    optimal_confidence = (1.96 * optimal_actions.std(axis=0)) / np.sqrt(runs) * 100

    return average_rewards, optimal_action_counts, reward_confidence, optimal_confidence

# Define agent-generating functions for different strategies
def greedy_agent(bandit):
    return GreedyAgent(bandit)

def eps_greedy_agent(epsilon, alpha=None):
    def agent(bandit):
        return EpsilonGreedyAgent(bandit, epsilon, alpha)
    return agent

def ucb_agent(c):
    def agent(bandit):
        return UCBAgent(bandit, c)
    return agent

def optimistic_agent(initial_value, alpha=None):
    def agent(bandit):
        return OptimisticAgent(bandit, initial_value, alpha)
    return agent

def grad_bandit_agent(alpha, baseline=True):
    def agent(bandit):
        return GradientBanditAgent(bandit, alpha, baseline)
    return agent

if __name__ == "__main__":
    np.random.seed(0)

    # Simulation parameters
    n_steps = 1000
    n_runs = 1000

    # Dictionary to store results for each agent
    agent_results = {}

    # Agent-specific parameters
    epsilons = [0.1]  # Epsilon-greedy vals
    c = 2                   # UCB parameter
    initial_value = 5       # Optimistic initial values
    alpha = 0.1             # Gradient bandit step size

    # ======
    # AGENTS
    # ======

    # Greedy agent
    greedy_results = run_bandit(greedy_agent, n_steps, n_runs)
    agent_results['Greedy'] = greedy_results

    # Epsilon-Greedy agents
    for epsilon in epsilons:
        label = f'Epsilon-Greedy ($\epsilon$={epsilon}, $\\alpha$=0.1)'
        results = run_bandit(eps_greedy_agent(epsilon, 0.1), n_steps, n_runs)
        agent_results[label] = results

    # UCB agent
    ucb_results = run_bandit(ucb_agent(c), n_steps, n_runs)
    agent_results[f'UCB (c={c})'] = ucb_results

    # Optimistic Initial Values agent
    optimistic_results = run_bandit(optimistic_agent(initial_value, 0.1), n_steps, n_runs)
    agent_results[f'Optimistic Init ($Q_1$={initial_value}, $\\alpha$=0.1)'] = optimistic_results

    # Gradient Bandit agent
    gradient_results = run_bandit(grad_bandit_agent(alpha), n_steps, n_runs)
    agent_results[f'Gradient Bandit ($\\alpha$={alpha})'] = gradient_results

    gradient_results_no_baseline = run_bandit(grad_bandit_agent(alpha, baseline=False), n_steps, n_runs)
    agent_results[f'Gradient Bandit ($\\alpha$={alpha}, baseline removed)'] = gradient_results_no_baseline

    # ==========================================================================

    # Plot Average Rewards
    plt.figure(figsize=(8, 4))
    i = 0
    colors = ["tab:blue", "tab:green", "tab:green", "tab:red"]
    for label, (avg_rewards, _, _, _) in agent_results.items():
        plt.plot(avg_rewards, label=label, color=colors[i], alpha=0.8)
        i += 1
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Steps')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

    # Plot % Optimal Action
    plt.figure(figsize=(8, 4))
    i = 0
    colors = ["tab:blue", "tab:green", "tab:green", "tab:red"]
    for label, (_, opt_actions, _, _) in agent_results.items():
        plt.plot(opt_actions, label=label, color=colors[i], alpha=0.8)
        i += 1
    plt.xlabel('Steps')
    plt.ylabel('\% Optimal Action')
    plt.title('\% Optimal Action vs. Steps')
    plt.legend()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
