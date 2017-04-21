from network3 import Network
from network3 import FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
import network3

import numpy as np
import theano
import theano.tensor as T
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
        self.layers = [
            FullyConnectedLayer(n_in=self.input_size, n_out=30, activation_fn=ReLU),
            FullyConnectedLayer(n_in=30, n_out=1)
        ]
        # initialize the network weights
        self.network = Network(self.layers, env)

    def construct_mb_input_row(self, state, action):
        # set the one-hot component for the state
        inpt = np.asarray(np.zeros(self.input_size), dtype='float32')
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


    def get_best_action(self, state):
        q_values = [self.network.q(state, action) for action in self.action_space]
        max_q_value = max(q_values)
        best_action_i = q_values.index(max_q_value)
        return self.action_space[best_action_i]

    def get_best_action_prob(self, state):
        q_table_values = [self.network.q(state, action) for action in self.action_space]
        q_table_softmax_dist = softmax(q_table_values)
        chosen_action = np.random.choice(self.action_space, 1, p=q_table_softmax_dist)[0]
        return chosen_action

    def print_q_function(self):
        for i in range(1, self.x_limit+1):
            for j in range(1, self.y_limit+1):
                for action in self.action_space:
                    state = (i,j)
                    # inpt = self.network.construct_input(state, action)
                    q_value = self.network.q(state, action)
                    print (i, j), action, q_value


    def train(self, epoch_data, gamma, eta=2):
        training_data = self.construct_training_data(epoch_data, gamma)
        self.network.SGD(training_data, eta)


    # training_data -> (inpt_matrix, label_matrix)
    # inpt_matrix -> np.array of dim (num_states, num_features)
    # label_matrix -> np.array of dim (num_states, 1)
    def construct_training_data(self, epoch_data, gamma):

        def shared(x, y):
            """Place the data into shared variables.  This allows Theano to copy
            the data to the GPU, if one is available.
            """
            shared_x = theano.shared(
                np.asarray(x, dtype=theano.config.floatX), borrow=True)
            shared_y = theano.shared(
                np.asarray(y, dtype=theano.config.floatX), borrow=True)
            return shared_x, T.cast(shared_y, "int32")

        training_x = []
        training_y = []
        for episode_data in epoch_data:
            # count backwards through each episode
            for i in range(len(episode_data)-1, -1, -1):
                (state, action, reward, next_state) = episode_data[i]
                # construct input
                x = self.construct_mb_input_row(state, action)
                training_x.append(x)
                # construct label
                best_next_action = self.get_best_action(next_state)
                y = reward + gamma * self.network.q(next_state, best_next_action)
                training_y.append(y)
        training_x = np.array(training_x)
        training_y = np.array(training_y)
        return shared(training_x, training_y)

# utilities --------------------------------------------------------------------

# compute softmax over a set of scores x, modified by temperature value T
# Note T=1 has no effect, higher values of T result in more randomness
def softmax(x, T=3):
    x = np.array(x) / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def state_to_index(state):
    (x, y) = (state[0] - 1, state[1] - 1)
    return x + 3 * y


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
