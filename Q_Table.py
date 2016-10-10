

class Q_Table:

    def __init__(self, env):
        self.q_table = self.initialize_q_table(env)


    def initialize_q_table(self, env):
        self.q_table = {}
        for x in range(1, env.x_limit+1):
            for y in range(1, env.y_limit+1):
                for action in env.action_space:
                    state = (x, y)
                    q_table[(state, action)] = 0
        return q_table

    # return the best action according to current values in Q-Table
    def get_action(self, state, env):
        # best_action -> (current best action, corresponding reward)
        action = env.action_space[0]
        best_action = (action, env.reward(state, action))
        for i_action in range(1, len(env.action_space)):
            action = env.action_space[i]
            if env.reward(state, action) > best_action[1]:
                best_action = (action, env.reward(state, action))
        return best_action[0]
