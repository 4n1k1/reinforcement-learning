import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from lib.envs.simple_rooms import SimpleRoomsEnv
from lib.simulation import Experiment

from shared.agent import Agent


class QLearningAgent(Agent):

    def __init__(self, actions, states, epsilon=0.01, alpha=0.5, gamma=1.0):
        super(QLearningAgent, self).__init__(actions)

        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._Q_table = {}

        for s in states:
            action_values = []

            for action in actions:
                action_value = 1.0  # np.random.random(1)[0]

                action_values.append(action_value)

            self._Q_table[s] = action_values

    def act(self, state):
        state = np.argmax(state)

        choice = np.random.binomial(1, self._epsilon)

        if choice == 1:
            return np.random.choice(self.num_actions)
        else:
            max_value_indices = []
            max_value = max(self._Q_table[state])

            for idx, action_value in enumerate(self._Q_table[state]):
                if action_value == max_value:
                    max_value_indices.append(idx)

            return np.random.choice(max_value_indices)

    def learn(self, state1, action1, reward, state2, done):
        state1 = np.argmax(state1)
        state2 = np.argmax(state2)

        self._Q_table[state1][action1] = self._Q_table[state1][action1] + \
            self._alpha * (
                reward + self._gamma * max(self._Q_table[state2]) - \
                self._Q_table[state1][action1]
            )
        """
          SARSA Update
          Q(s,a) <- Q(s,a) + alpha * (reward + gamma * Q(s',a') - Q(s,a))
          or
          Q(s,a) <- Q(s,a) + alpha * (td_target - Q(s,a))
          or
          Q(s,a) <- Q(s,a) + alpha * td_delta
        """

interactive = False

env = SimpleRoomsEnv()
agent = QLearningAgent(
    range(env.action_space.n),
    env.S,
)
experiment = Experiment(env, agent)
experiment.run_sarsa(50, interactive)
