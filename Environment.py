class Environment:

    def __init__(self):
        self.size = 10
        self.goals = {(10,10): 100, (5,5): -100}


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
        if nextState[0] > self.size:
            return (self.size, nextState[1])
        elif nextState[1] > self.size:
            return (nextState[0], self.size)
        elif nextState[0] < 1:
            return (1, nextState[1])
        elif nextState[1] < 1:
            return (nextState[0], 1)
        # else in bounds
        else:
            return nextState

    def reward(self, state, action):
        nextState = self.stateTransition(state, action)
        if nextState in self.goals:
            return self.goals[nextState]
        else:
            return 0
