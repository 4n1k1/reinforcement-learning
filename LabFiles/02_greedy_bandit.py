import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import Policy


class Greedy(Policy):
    def __init__(self, num_actions):
        Policy.__init__(self, num_actions)

        self.name = "Greedy"

    def act(self):
        current_averages = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)

        #
        # Correctly handles Bernoulli rewards; over-estimates otherwise
        #
        # Give non-zero overages to other actions so that we don't
        # stick to the first non-zero action right away
        #
        current_averages[self.total_counts <= 0] = 0.5
        current_action = np.argmax(current_averages)

        return current_action


evaluation_seed = 8026
num_actions = 5
trials = 100
distribution = "bernoulli"

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = Greedy(num_actions)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
