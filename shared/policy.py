import numpy as np
import random

class Policy:

    def __init__(self, num_actions, r=0.5):
        self.num_actions = num_actions
        self.total_rewards = np.zeros(num_actions)
        self.total_counts = np.zeros(num_actions)
        self.current_averages=np.array([r] * num_actions)
        # print("Current averages")
        # print(self.current_averages)

    def act(self):
        pass

    def feedback(self, action, reward):
        # print("================")
        # print("action: " + str(action))
        # print("Current averages")
        # print(self.current_averages)
        self.total_rewards[action] += reward
        self.total_counts[action] += 1


class RoundRobin(Policy):

    def __init__(self, num_actions):
        Policy.__init__(self, num_actions)
        self.name = "Round Robin"
        self.previous_action = 0
    
    def act(self):
        self.previous_action += 1

        if self.previous_action == self.num_actions:
            self.previous_action = 0

        return self.previous_action


class Greedy(Policy):
    def __init__(self, num_actions, r=0.5):
        Policy.__init__(self, num_actions, r)

        self.name = "Greedy"

    def act(self):
        np.divide(
            self.total_rewards,
            self.total_counts,
            out=self.current_averages,
            where=self.total_counts > 0,
        )
        return np.argmax(self.current_averages)


class EpsilonGreedy(Greedy):
    def __init__(self, num_actions, epsilon):
        super(EpsilonGreedy, self).__init__(num_actions)

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
            return super(EpsilonGreedy, self).act()


class GradientOptimism(Policy):
    """
        This is in my opinion best action picking
        policy. I made it up myself. All rights reserved :)
    """
    def __init__(self, num_actions, r):
        super(GradientOptimism, self).__init__(num_actions, r)
        self.name = "Greedy"

    def act(self):
        np.divide(
            self.total_rewards + self.current_averages,
            self.total_counts,
            out=self.current_averages,
            where=self.total_counts > 0,
        )

        return np.argmax(self.current_averages)


class UCB(Policy):
    def __init__(self, num_actions):
        super(UCB, self).__init__(num_actions)

        self.name = "UCB"
        self.exploration_bonuses = np.array([0] * num_actions)
        self.round = 0
        
    def act(self):
        current_action = None
        self.round += 1

        if self.round <= self.num_actions:
            """The first k rounds, where k is the number of arms/actions, play each arm/action once"""
            current_action = self.round - 1
        else:
            """At round t, play the arms with maximum average and exploration bonus"""
            np.divide(
                self.total_rewards,
                self.total_counts,
                out=self.current_averages,
                where=self.total_counts > 0,
            )

            current_action = np.argmax(
                self.current_averages + self.exploration_bonuses,
            )

        return current_action

    def feedback(self, action, reward):
        super(UCB, self).feedback(action, reward)

        self.exploration_bonuses[action] = np.sqrt(
            2.0 * np.log(self.round) / self.total_counts[action]
        )


class ThompsonSampling(Policy):
    def __init__(self, num_actions):
        Policy.__init__(self, num_actions)

        self.successes = np.zeros(num_actions)
        self.failures = np.zeros(num_actions)

        self.name = "Thompson Beta"

    def act(self):
        max_random = 0
        action_to_pick = 0

        for i in range(self.num_actions):
            random_beta = random.betavariate(self.successes[i] + 1, self.failures[i] + 1)

            if random_beta > max_random:
                max_random = random_beta
                action_to_pick = i

        return action_to_pick

    def feedback(self, action, reward):
        if reward > 0:
            self.successes[action] += 1
        else:
            self.failures[action] += 1

        self.total_counts[action] += 1
