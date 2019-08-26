import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.simulation import Experiment


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


interactive = False
max_number_of_episodes = 5
env = SimpleRoomsEnv()
agent = RandomAgent(range(env.action_space.n))
experiment = Experiment(env, agent)
experiment.run_agent(max_number_of_episodes, interactive)
