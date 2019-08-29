import numpy as np


class Agent(object):
    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, obs):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, actions):
        super(RandomAgent, self).__init__(actions)

    def act(self, obs):
        return np.random.randint(0, self.num_actions)
