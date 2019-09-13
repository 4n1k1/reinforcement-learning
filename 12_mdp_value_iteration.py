import test_dp               # required for testing and grading your code
import gridworld_mdp as gw   # defines the MDP for a 4x4 gridworld


def value_iteration(state_count, gamma, theta, get_available_actions, get_transitions):
    """
    This function computes the optimal value function and policy for the specified MDP, using the Value Iteration algorithm.

    'state_count' is the total number of states in the MDP. States are represented as 0-relative numbers.
    'gamma' is the MDP discount factor for rewards.
    'theta' is the small number threshold to signal convergence of the value function (see Iterative Policy Evaluation algorithm).
    'get_available_actions' returns a list of the MDP available actions for the specified state parameter.
    'get_transitions' is the MDP state / reward transiton function.  It accepts two parameters, state and action, and returns
        a list of tuples, where each tuple is of the form: (next_state, reward, probabiliity).  
    """
    state_values = [0.0] * state_count  # array of state values
    policy = ["up"] * state_count

    while True:
        delta = 0.0

        for idx, value in enumerate(state_values):
            new_value = None

            for action in get_available_actions(idx):
                next_state_idx, reward, probability = get_transitions(idx, action)[0]
                next_value = probability * (reward + gamma * state_values[next_state_idx])

                if new_value is None:
                    new_value = next_value
                else:
                    if new_value < next_value:
                        new_value = next_value

            state_values[idx] = new_value

            delta = max([delta, value - state_values[idx]])

        if delta < theta:
            break

    for idx, value in enumerate(state_values):
        new_value = None

        for action in get_available_actions(idx):
            next_state_idx, reward, probability = get_transitions(idx, action)[0]
            next_value = probability * (reward + gamma * state_values[next_state_idx])

            if new_value is None:
                new_value = next_value
                policy[idx] = action
            else:
                if new_value < next_value:
                    new_value = next_value
                    policy[idx] = action

    return (state_values, policy)

n_states = gw.get_state_count()

# test our function
values, policy = value_iteration(state_count=n_states, gamma=.9, theta=.001, get_available_actions=gw.get_available_actions, \
    get_transitions=gw.get_transitions)

print("Values=", values)
print("Policy=", policy)

# test our function using the test_db helper
test_dp.value_iteration_test( value_iteration ) 
