import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import GradientOptimism

R=100.0
evaluation_seed = 1239
num_actions = 10
trials = 10000
distribution = "bernoulli"

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = GradientOptimism(num_actions, R)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
