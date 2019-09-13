import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld

state = 2
n_states = 15
actions = gw.get_available_actions(0)


def get_equal_policy(state):
    # build a simple policy where all 4 actions have the same probability, ignoring the specified state
    policy = (("up", .25), ("right", .25), ("down", .25), ("left", .25),)
    return policy


def examine_transitions():
  for action in actions:
      transitions = gw.get_transitions(state=state, action=action)

      # examine each return transition (only 1 per call for this MDP)
      for (trans) in transitions:
          next_state, reward, probability = trans    # unpack tuple

          print(
              "transition(" + str(state) + ", " + action + "):",
              "next_state=", next_state, ", reward=", reward,
              ", probability=", probability,
          )


def policy_eval_in_place(state_count, gamma, theta, get_policy, get_transitions):
    """
      This function uses the two-array approach to evaluate the specified policy for the specified MDP:

      'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
      'gamma' is the MDP discount factor for rewards.
      'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
      'get_policy' is the stochastic policy function - it takes a state parameter and returns list of tuples,
          where each tuple is of the form: (action, probability).  It represents the policy being evaluated.
      'get_transitions' is the state/reward transiton function.  It accepts two parameters, state and action, and returns
          a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).
    """
    V0 = [0.0] * state_count  # array of state values

    while True:
        delta = 0.0

        for idx, value in enumerate(V0):
            new_value = 0.0

            for policy in get_policy(idx):
                next_state_idx, reward, _ = get_transitions(idx, policy[0])[0]

                new_value += policy[1] * (reward + gamma * V0[next_state_idx])

            V0[idx] = new_value

            delta = max([delta, value - V0[idx]])

        if delta < theta:
            break

    return V0


# test our function
values = policy_eval_in_place(
    state_count=n_states,
    gamma=.9,
    theta=.001,
    get_policy=get_equal_policy,
    get_transitions=gw.get_transitions,
)

import numpy as np
a = np.append(values, 0)

print(np.reshape(a, (4,4)))
print("----------------------")

test_dp.policy_eval_in_place_test( policy_eval_in_place )
