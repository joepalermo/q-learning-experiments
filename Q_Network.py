import network2
import numpy as np
import time

from Environment import Environment


class Q_Network:

    def __init__(self, env):
        # extract relevant environment data
        self.x_limit = env.x_limit
        self.y_limit = env.y_limit
        self.goal_states = env.goal_states
        self.action_space = env.action_space
        # the input size determines the network architecture
        self.input_size = 2 * self.x_limit * self.y_limit + len(self.action_space)
        # define the network
        self.network = network2.Network([self.input_size, 5, 1], cost=network2.CrossEntropyCost)
        # initialize the network weights
        self.network.default_weight_initializer()

    # get the best action from a given state
    def get_best_action(self, state):
        inpts = [self.construct_input(state, action) for action in self.action_space]
        q_values = [self.network.output(inpt) for inpt in inpts]
        max_q_value = max(q_values)
        best_action_i = q_values.index(max_q_value)
        return self.action_space[best_action_i]

    # get an action from a given state, choosing better actions with higher probability
    def get_best_action_prob(self, state):
        inpts = [self.construct_input(state, action) for action in self.action_space]
        q_values = [self.network.output(inpt) for inpt in inpts]
        softmax_dist = softmax(q_values)
        chosen_action = np.random.choice(self.action_space, 1, p=softmax_dist)[0]
        return chosen_action

    # train the q-network on a single episode of training data
    def episode_train(self, episode_data, gamma, eta):
        # count backwards through each episode
        for i in range(len(episode_data)-1, -1, -1):
            (state, action, reward, next_state) = episode_data[i]
            # construct input
            x = self.construct_input(state, action)
            # construct label
            best_next_action = self.get_best_action(next_state)
            x_prime = self.construct_input(next_state, best_next_action)
            y = reward + gamma * self.network.feedforward(x_prime)
            y = normalize_y(y)
            # contruct the training example
            training_example = (x,y)
            # update the network on the basis of the training example
            self.network.update(training_example, eta)


    # train the q-network on an epoch of training data
    def epoch_train(self, epoch_data, gamma, eta):
        for episode_data in epoch_data:
            self.episode_train(episode_data, gamma, eta)


    # encode the state-action pair as an numpy array
    # goal is also included implicitly as part of the state
    # state-goal-action
    def construct_input(self, state, action):
        # set the one-hot component for the state
        inpt = np.asarray(np.zeros((self.input_size, 1)))
        state_index = state_to_index(state)
        inpt[state_index] = 1
        offset = self.x_limit * self.y_limit
        # assume there is only one goal
        # set the one-hot component for the goals
        goal_indices = [state_to_index(goal_state) for goal_state in self.goal_states]
        for i in goal_indices:
            inpt[offset + i] = 1
        offset += self.x_limit * self.y_limit
        # set the one-hot component for the action
        action_index = self.action_space.index(action)
        inpt[offset + action_index] = 1
        return inpt

    def print_q_function(self):
        for i in range(1, self.x_limit+1):
            for j in range(1, self.y_limit+1):
                for action in self.action_space:
                    state = (i,j)
                    inpt = self.construct_input(state, action)
                    q_value = self.network.output(inpt)
                    print (i, j), action, q_value

# utilities --------------------------------------------------------------------

# compute softmax over a set of scores x, modified by temperature value T
# Note T=1 has no effect, higher values of T result in more randomness
def softmax(x, T=1):
    x = np.array(x) / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# given a state in R ^ 2, find its index counting along successive rows
def state_to_index(state):
    (x, y) = (state[0] - 1, state[1] - 1)
    return x + 3 * y

# ensure that y remains in the range 0 -> 1
def normalize_y(y):
    if y > 1:
        return np.ones((1,1))
    elif y < 0:
        return np.zeros((1,1))
    else:
        return y

def unit_testing():
    env = Environment()
    q_net = Q_Network(env)
    state = (1,1)
    action = "down"
    print q_net.network.construct_input(state, action)
    state = (1,3)
    action = "down"
    print q_net.network.construct_input(state, action)
    state = (3,3)
    action = "down"
    print q_net.network.construct_input(state, action)

#unit_testing()
