import numpy as np

class EpsilonGreedyAgent:
    """
    Epsilon-greedy agent, explores with probability epsilon.
    """
    def __init__(self, bandit, epsilon=0.1, alpha=None):
        self.k = bandit.k
        self.eps = epsilon
        self.q = np.zeros(self.k)
        self.a_count = np.zeros(self.k)
        self.alpha = alpha


    def select_action(self):
        """
        Select action using epsilon-greedy strategy.
        Returns index of the selected action.
        """
        # Exploration - select random action
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.k)
        
        # Exploitation - select best-known action
        else:
            return np.argmax(self.q)


    def update_estimates(self, a, r):
        """
        Update estimated value of selected action.

        :param a: Selected action index.
        :param r: The observed reward.
        """
        self.a_count[a] += 1
        if self.alpha is None:
            self.q[a] += (r - self.q[a]) / self.a_count[a]
        else:
            self.q[a] += self.alpha * (r - self.q[a])