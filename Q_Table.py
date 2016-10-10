from random import shuffle

class Q_Table:

    def __init__(self, env):
        self.action_space = env.action_space
        self.q_table = env.initialize_q_table()

    # get the best action from a given state according to current Q-Table
    def get_best_action(self, state):
        # get a shuffled version of self.action_space so that actions are chosen
        # in random order, preventing an agent from getting stuck on an edge
        action_space = list(self.action_space)
        shuffle(action_space)
        action = action_space[0]
        # best_action -> (current best action, corresponding reward)
        best_action = (action, self.q_table[(state, action)])
        for i in range(1, len(action_space)):
            action = action_space[i]
            if self.q_table[(state, action)] > best_action[1]:
                best_action = (action, self.q_table[(state, action)])
        return best_action[0]

    def update(self, step_data, gamma):
        (state, action, reward, next_state) = step_data
        best_next_action = self.get_best_action(next_state)
        self.q_table[(state, action)] = \
        reward + \
        gamma * self.q_table[(next_state, best_next_action)]
