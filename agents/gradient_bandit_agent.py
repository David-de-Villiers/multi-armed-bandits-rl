import numpy as np

class GradientBanditAgent:
    """
    Gradient Bandit agent 
    learns action preferences using gradient ascent method.
    """
    def __init__(self, bandit, alpha=0.1, baseline=True):
        """
        Initialize the Gradient Bandit agent.
        Action preferences are initialised to zero

        :param bandit: The bandit environment.
        :param alpha: Step size parameter for updating preferences.
        """
        self.k = bandit.k
        self.alpha = alpha
        self.prefs = np.zeros(self.k)
        self.reward_hist = []
        self.enable_baseline = baseline

    def select_action(self):
        """
        Select an action based on the prefs softmax.

        :return: Selected action index.
        """
        exp_prefs = np.exp(self.prefs)
        self.action_probs = exp_prefs / np.sum(exp_prefs)
        action = np.random.choice(self.k, p=self.action_probs)
        return action

    def update_estimates(self, action, reward):
        """
        Update action prefs based on received reward.

        :param action: Index of the action taken.
        :param reward: Received reward.
        """
        # Update action prefs
        one_hot = np.zeros(self.k)
        one_hot[action] = 1

        if self.enable_baseline:
            # Update average reward and compute baseline
            self.reward_hist.append(reward)
            baseline = np.average(np.array(self.reward_hist))
            self.prefs += self.alpha * (reward - baseline) * (one_hot - self.action_probs)
        else:
            self.prefs += self.alpha * (reward) * (one_hot - self.action_probs)
