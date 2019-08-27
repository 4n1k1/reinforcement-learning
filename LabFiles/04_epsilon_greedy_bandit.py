import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import Policy


class EpsilonGreedy(Policy):
    def __init__(self, num_actions, apsilon):
        Policy.__init__(self, num_actions)

        self.epsilon = epsilon

        if self.epsilon is None or self.epsilon < 0 or self.epsilon > 1:
            self.epsilon = 0.01
            
        self.name = "Epsilon Greedy"

    def act(self):
        choice = None

        if self.epsilon == 0:
            choice = 0
        elif self.epsilon == 1:
            choice = 1
        else:
            choice = np.random.binomial(1, self.epsilon)
            
        if choice == 1:
            return np.random.choice(self.num_actions)
        else:
            current_averages = np.divide(self.total_rewards, self.total_counts, where = self.total_counts > 0)
            current_averages[self.total_counts <= 0] = 0.5  #Correctly handles Bernoulli rewards; over-estimates otherwise
            current_action = np.argmax(current_averages)
            return current_action


evaluation_seed = 1239
num_actions = 10
trials = 10000
distribution = "bernoulli"

epsilon = 0.15
env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = EpsilonGreedy(num_actions, epsilon)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
