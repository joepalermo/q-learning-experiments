class Environment:

    def __init__(self):
        # define the boundary of the Environment
        self.x_limit = 10
        self.y_limit = 10
        # define the action space
        self.action_space = ["up", "left", "down", "right"]
        # define a mapping from state to reward
        # use it to generate a mapping from (state, action) to reward
        # in this case we use the convention that
        self.reward_map = \
        Environment.add_actions_to_reward_map({ (10,10): 100,
                                                (5,5): -100}, self.action_space)

    @staticmethod
    def add_actions_to_reward_map(reward_map, action_space):
        reward_map_with_actions = {}
        for state in reward_map:
            for action in action_space:
                (x, y) = state
                if action == "up":
                    state_action_pair = ((x,y-1), action)
                    reward_map_with_actions[state_action_pair] = reward_map[state]
                elif action == "left":
                    state_action_pair = ((x+1,y), action)
                    reward_map_with_actions[state_action_pair] = reward_map[state]
                elif action == "down":
                    state_action_pair = ((x,y+1), action)
                    reward_map_with_actions[state_action_pair] = reward_map[state]
                elif action == "right":
                    state_action_pair = ((x-1,y), action)
                    reward_map_with_actions[state_action_pair] = reward_map[state]
        return reward_map_with_actions

    def initialize_q_table(self):
        q_table = {}
        for x in range(1, self.x_limit+1):
            for y in range(1, self.y_limit+1):
                for action in action_space:
                    state = (x, y)
                    q_table[(state, action)] = 0
        return q_table

    def state_transition(self, state, action):
        (x, y) = state
        if action == "up":
            next_state = (x, y+1)
        elif action == "down":
            next_state = (x, y-1)
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

    def reward(self, state, action):
        if (state, action) in self.reward_map:
            return self.reward_map[(state, action)]
        else:
            return 0
