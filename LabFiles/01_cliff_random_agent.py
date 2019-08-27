import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.cliff_walking import CliffWalkingEnv
from lib.simulation import Experiment
from shared.agent import RandomAgent

interactive = False
max_number_of_episodes = 100
env = CliffWalkingEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)
