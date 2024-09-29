import numpy as np

class Bandit:
    """Stationary k-armed bandit problem.

    :attribute k: Number of levers/actions to take.
    :attribute q_true: True action values for each arm.
    :attribute optimal_action: The index of the optimal action.
    """

    def __init__(self, k=5, true_mean=0):
        """
        Initialise true rewards from normal distribution with mean 0, variance 1 

        :param k: Number of actions to take, defaults to 5
        """
        self.k = k
        self.q_true = np.random.normal(true_mean, 1, k)
        self.optimal_action = np.argmax(self.q_true)


    def pull_lever(self, action):
        """
        Select action/pull lever

        :param a: The index of the action taken.
        :return: reward, normally distributed around true value, variance 1
        """
        # Rewards are normally distributed around the true value (variance=1)
        return np.random.normal(self.q_true[action], 1)