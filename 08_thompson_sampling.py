import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.bandit import BanditEnv
from lib.simulation import Experiment
from shared.policy import ThompsonSampling

evaluation_seed = 1239
num_actions = 10
trials = 100
distribution = "normal"

env = BanditEnv(num_actions, distribution, evaluation_seed)
agent = ThompsonSampling(num_actions)
experiment = Experiment(env, agent)
experiment.run_bandit(trials)
