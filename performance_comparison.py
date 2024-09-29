import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from tqdm import tqdm
from scipy.ndimage.filters import uniform_filter1d  # For smoothing

# Bandit agents
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

def run_bandit(current_agent, steps=1000, runs=500):
    """
    Simulate agent interaction with bandit.

    :param current_agent: Function that returns an agent instance when passed a bandit.
    :param steps: Number of steps to take per run.
    :param runs: Number of runs to average over.

    :return: average_rewards, average rewards over runs.
    """
    rewards = np.zeros(runs)

    # Run simulations
    for run in range(runs):
        bandit = Bandit(k=10)
        agent = current_agent(bandit)

        total_reward = 0
        for _ in range(steps):
            action = agent.select_action()
            reward = bandit.pull_lever(action)
            agent.update_estimates(action, reward)
            total_reward += reward

        # Record average reward for this run
        rewards[run] = total_reward / steps

    average_reward = np.mean(rewards)
    return average_reward

# Define agent-generating functions for different strategies
def eps_greedy_agent(epsilon):
    def agent(bandit):
        return EpsilonGreedyAgent(bandit, epsilon)
    return agent

def ucb_agent(c):
    def agent(bandit):
        return UCBAgent(bandit, c)
    return agent

def optimistic_agent(initial_value, alpha=0.1):
    def agent(bandit):
        return OptimisticAgent(bandit, initial_value, alpha)
    return agent

def grad_bandit_agent(alpha):
    def agent(bandit):
        return GradientBanditAgent(bandit, alpha)
    return agent

if __name__ == "__main__":
    np.random.seed(0)

    # Simulation parameters
    n_steps = 1000
    n_runs = 1000

    # Dictionaries to store results for each agent
    average_rewards_epsilon = []
    average_rewards_alpha = []
    average_rewards_c = []
    average_rewards_q0 = []

    # Parameter values for each method
    epsilon_values = np.arange(0, 1.05, 0.05)
    alpha_values = np.arange(0, 8.2, 0.2)
    c_values = np.arange(0, 8.2, 0.2)
    initial_values = np.arange(0, 8.2, 0.2)

    # ==========================
    # Epsilon-Greedy Agent Sweep
    # ==========================
    print("Starting Epsilon-Greedy parameter sweep...")
    for epsilon in tqdm(epsilon_values, desc='Epsilon-Greedy', unit='param'):
        avg_reward = run_bandit(eps_greedy_agent(epsilon), n_steps, n_runs)
        average_rewards_epsilon.append(avg_reward)

    # ==========================
    # Gradient Bandit Agent Sweep
    # ==========================
    print("Starting Gradient Bandit parameter sweep...")
    for alpha in tqdm(alpha_values, desc='Gradient Bandit', unit='param'):
        avg_reward = run_bandit(grad_bandit_agent(alpha), n_steps, n_runs)
        average_rewards_alpha.append(avg_reward)

    # ==========================
    # UCB Agent Sweep
    # ==========================
    print("Starting UCB Agent parameter sweep...")
    for c in tqdm(c_values, desc='UCB', unit='param'):
        avg_reward = run_bandit(ucb_agent(c), n_steps, n_runs)
        average_rewards_c.append(avg_reward)

    # ====================================
    # Optimistic Initialization Agent Sweep
    # ====================================
    print("Starting Optimistic Initialization parameter sweep...")
    for q0 in tqdm(initial_values, desc='Optimistic Init', unit='param'):
        avg_reward = run_bandit(optimistic_agent(q0, alpha=0.1), n_steps, n_runs)
        average_rewards_q0.append(avg_reward)

    # ==========================
    # Plotting the Results
    # ==========================

    # Smoothing function
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # Set up figure
    plt.figure(figsize=(10, 6))

    # Plot Gradient Bandit results
    alpha_rewards_smooth = smooth(average_rewards_alpha, 3)
    plt.plot(alpha_values + 1e-5, alpha_rewards_smooth, label='Gradient Bandit', color='tab:green')
    plt.fill_between(alpha_values + 1e-5, alpha_rewards_smooth, color='tab:green', alpha=0.2)

    # Plot Optimistic Initialization results
    q0_rewards_smooth = smooth(average_rewards_q0, 3)
    plt.plot(initial_values + 1e-5, q0_rewards_smooth, label='Optimistic Init', color='tab:gray')
    plt.fill_between(initial_values + 1e-5, q0_rewards_smooth, color='tab:gray', alpha=0.2)

    # Plot UCB results
    c_rewards_smooth = smooth(average_rewards_c, 3)
    plt.plot(c_values + 1e-5, c_rewards_smooth, label='UCB', color='tab:blue')
    plt.fill_between(c_values + 1e-5, c_rewards_smooth, color='tab:blue', alpha=0.2)

    # Plot Epsilon-Greedy results
    epsilon_rewards_smooth = smooth(average_rewards_epsilon, 3)
    plt.plot(epsilon_values + 1e-5, epsilon_rewards_smooth, label='Epsilon-Greedy', color='tab:red')
    plt.fill_between(epsilon_values + 1e-5, epsilon_rewards_smooth, color='tab:red', alpha=0.2)

    # Axis labels and title
    plt.xlabel('Parameter Value')
    plt.ylabel('Average Reward after 1000 steps')
    plt.title('Performance Comparison')

    # Set tick label colors
    ax = plt.gca()
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    plt.legend(loc='lower right')

    plt.show()
