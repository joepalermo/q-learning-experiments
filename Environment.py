from random import randint

class Environment:

    def __init__(self):
        # define the boundary of the Environment
        self.x_limit = 3
        self.y_limit = 3
        # define the action space
        self.action_space = ["up", "left", "down", "right"]
        # define the goal state
        self.goal_states = [(3,3)]
        # construct a map from states to rewards
        self.state_to_reward = {(3,3): 1}
        # construct a map from state-actions pairs to reward
        self.construct_reward_map()

    # construct a map from state-action pairs to reward
    def construct_reward_map(self):
        self.reward_map = {}
        for state in self.state_to_reward:
            for action in self.action_space:
                (x, y) = state
                if action == "up":
                    state_action_pair = ((x,y+1), action)
                    self.reward_map[state_action_pair] = \
                        self.state_to_reward[state]
                elif action == "left":
                    state_action_pair = ((x+1,y), action)
                    self.reward_map[state_action_pair] = \
                        self.state_to_reward[state]
                elif action == "down":
                    state_action_pair = ((x,y-1), action)
                    self.reward_map[state_action_pair] = \
                        self.state_to_reward[state]
                elif action == "right":
                    state_action_pair = ((x-1,y), action)
                    self.reward_map[state_action_pair] = \
                        self.state_to_reward[state]

    # initialize the Q-table to value 0 for all state-action pairs
    def initialize_q_table(self):
        q_table = {}
        for x in range(1, self.x_limit+1):
            for y in range(1, self.y_limit+1):
                for action in self.action_space:
                    state = (x, y)
                    q_table[(state, action)] = 0
        return q_table

    # reset state in a random non-goal state
    def reset(self):
        while True:
            x = randint(1, self.x_limit)
            y = randint(1, self.y_limit)
            if (x, y) not in self.goal_states:
                break
        return (x, y)

    # get step data resulting from taking a given action in a given state
    def step(self, state, action):
        next_state = self.state_transition(state, action)
        reward = self.reward(state, action)
        done = next_state in self.goal_states
        return (next_state, reward, done)

    # get the succeeding state resulting from taking a given action in a given
    # state
    def state_transition(self, state, action):
        (x, y) = state
        if action == "up":
            next_state = (x, y-1)
        elif action == "down":
            next_state = (x, y+1)
        elif action == "left":
            next_state = (x-1, y)
        elif action == "right":
            next_state = (x+1, y)
        # check and correct for the possibility of state going out of bounds
        if next_state[0] > self.x_limit:
            return (self.x_limit, next_state[1])
        elif next_state[1] > self.y_limit:
            return (next_state[0], self.y_limit)
        elif next_state[0] < 1:
            return (1, next_state[1])
        elif next_state[1] < 1:
            return (next_state[0], 1)
        # else in bounds
        else:
            return next_state

    # get the reward that results from taking a given action in a given state
    def reward(self, state, action):
        if (state, action) in self.reward_map:
            return self.reward_map[(state, action)]
        else:
            return 0
