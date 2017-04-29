import numpy as np
from random import shuffle

class Q_Table:

    def __init__(self, env):
        self.action_space = env.action_space
        self.q_table = env.initialize_q_table()

    def pprint(self):
        for ((state, action), q_value) in sorted(self.q_table.items()):
            print str((state, action)) + ", " + str(q_value)

    # get the best action from a given state according to current Q-table
    def get_best_action(self, state):
        # get a shuffled version of self.action_space so that actions are
        # considered in random order, preventing an agent from getting stuck on
        # an edge of the environment
        shuffle(self.action_space)
        action = self.action_space[0]
        # best_action -> (current best action, corresponding reward)
        best_action = (action, self.q_table[(state, action)])
        for i in range(1, len(self.action_space)):
            action = self.action_space[i]
            if self.q_table[(state, action)] > best_action[1]:
                best_action = (action, self.q_table[(state, action)])
        return best_action[0]

    def get_best_action_prob(self, state):
        q_table_values = [self.q_table[(state, action)] for action in self.action_space]
        q_table_softmax_dist = softmax(q_table_values)
        chosen_action = np.random.choice(self.action_space, 1, p=q_table_softmax_dist)[0]
        return chosen_action

    def episode_train(self, episode_data, **learning_parameters):
        gamma = learning_parameters['gamma']
        # loop backwards through episode_data, since that is the order in which
        # we want to do updates
        for i in range(len(episode_data)-1, -1, -1):
            step_data = episode_data[i]
            self.update(step_data, gamma)

    # update the Q-table on the basis of step data
    def update(self, step_data, gamma):
        (state, action, reward, next_state) = step_data
        best_next_action = self.get_best_action(next_state)
        self.q_table[(state, action)] = \
        reward + gamma * self.q_table[(next_state, best_next_action)]


# utilities --------------------------------------------------------------------

# compute softmax over a set of scores x, modified by temperature value T
# Note T=1 has no effect, higher values of T result in more randomness
def softmax(x, T=0.1):
    x = np.array(x) / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
