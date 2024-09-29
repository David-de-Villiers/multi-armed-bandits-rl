import numpy as np

class OptimisticAgent:
    """
    Agent with action-value estimates initialised to high/optimistic values 
    to encourage exploration.
    """
    def __init__(self, bandit, initial_value=5, alpha=None):
        """
        Initialise agent with optimistic initial values.

        :param bandit: Bandit environment.
        :param initial_value: Initial optimistic value for all action-value ests
        """
        self.k = bandit.k
        self.q = np.ones(self.k) * initial_value
        self.a_count = np.zeros(self.k)
        self.alpha = alpha


    def select_action(self):
        """
        Return action with highest estimated value.
        """
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
