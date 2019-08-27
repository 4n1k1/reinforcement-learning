import numpy as np


class Policy:
    #num_actions: (int) Number of arms [indexed by 0 ... num_actions-1]
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.total_rewards = np.zeros(num_actions)
        self.total_counts = np.zeros(num_actions)

    def act(self):
        pass

    def feedback(self, action, reward):
        self.total_rewards[action] += reward
        self.total_counts[action] += 1
