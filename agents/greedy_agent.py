import numpy as np

class GreedyAgent:
    """
    Greedy agent, selects highest estimated value action
    """
    def __init__(self, bandit):
        self.k = bandit.k
        self.q = np.zeros(self.k)
        self.a_count = np.zeros(self.k)

    def select_action(self):
        """
        Return index of action with highest estimated value.
        """
        return np.argmax(self.q)

    def update_estimates(self, a, r):
        """
        Update estimated value of selected action.

        :param a: Index of the action taken.
        :param r: Received reward.
        """
        self.a_count[a] += 1
        self.q[a] += (r - self.q[a]) / self.a_count[a]