import numpy as np

class UCBAgent:
    """
    Upper Confidence Bound (UCB) agent.
    Selects actions based on upper confidence bounds, 
    balances exploration and exploitation.
    """
    def __init__(self, bandit, c=2):
        """
        Initialise the UCB agent.

        :param bandit: Bandit environment.
        :param c: Confidence level parameter.
        """
        self.k = bandit.k
        self.c = c
        self.q = np.zeros(self.k)
        self.a_count = np.zeros(self.k)
        self.n_steps = 0


    def select_action(self):
        """
        Select action using UCB policy.

        :return: Selected action index.
        """
        self.n_steps += 1

        # Select action that has not been tried yet
        if 0 in self.a_count:
            return np.argmin(self.a_count)
        
        # Compute UCB values
        ucb_values = self.q + self.c * np.sqrt(np.log(self.n_steps) / self.a_count)
        return np.argmax(ucb_values)


    def update_estimates(self, a, r):
        """
        Update estimated value of selected action.

        :param action: The index of the action taken.
        :param reward: The observed reward.
        """
        self.a_count[a] += 1
        self.q[a] += (r - self.q[a]) / self.a_count[a]
