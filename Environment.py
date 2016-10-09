class Environment:

    def __init__(self):
        # define the boundary of the Environment
        self.x_limit = 10
        self.y_limit = 10
        # define the action space
        self.action_space = ["up", "left", "down", "right"]
        # define a mapping from states to rewards
        self.reward_map = {(10,10): 100, (5,5): -100}

    def initialize_q_table(self):
        q_table = {}
        for x in range(1, self.x_limit+1):
            for y in range(1, self.y_limit+1):
                for action in action_space:
                    state = (x, y)
                    q_table[(state, action)] = 0
        return q_table

    def stateTransition(self, state, action):
        (x, y) = state
        if action == "up":
            nextState = (x, y+1)
        elif action == "down":
            nextState = (x, y-1)
        elif action == "left":
            nextState = (x-1, y)
        elif action == "right":
            nextState = (x+1, y)
        # check and correct for the possibility of state going out of bounds
        if nextState[0] > self.x_limit:
            return (self.x_limit, nextState[1])
        elif nextState[1] > self.y_limit:
            return (nextState[0], self.y_limit)
        elif nextState[0] < 1:
            return (1, nextState[1])
        elif nextState[1] < 1:
            return (nextState[0], 1)
        # else in bounds
        else:
            return nextState

    def reward(self, state, action):
        nextState = self.stateTransition(state, action)
        if nextState in self.reward_map:
            return self.reward_map[nextState]
        else:
            return 0
