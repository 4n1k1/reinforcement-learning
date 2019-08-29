import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import Greedy

"""
    This is not exact optimistic greedy which is
    required by the course (on very high values
    it doesn't have random behavior). But in
    general behaves similarly. I wasn't able to
    figure out how to implement exactly required
    policy.
"""

R=10000.0
evaluation_seed = 1239
num_actions = 10
trials = 100
distribution = "bernoulli"

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = Greedy(num_actions, R)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
