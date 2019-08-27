import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import Policy


class RoundRobin(Policy):

    def __init__(self, num_actions):
        Policy.__init__(self, num_actions)
        self.name = "Round Robin"
        self.previous_action = 0 #keep track of previous action
    
    def act(self):
        self.previous_action += 1

        if self.previous_action == self.num_actions:
            self.previous_action = 0

        return self.previous_action


evaluation_seed = 8026
num_actions = 5
trials = 10000
distribution = "bernoulli"

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = RoundRobin(num_actions)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
