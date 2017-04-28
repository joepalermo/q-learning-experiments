from random import randint

class Moving_Goal_Env:

    def __init__(self):
        # define the boundary of the Environment
        self.x_limit = 5
        self.y_limit = 5
        # define the action space
        self.action_space = ["up", "left", "down", "right"]
        self.reward_map = {'found_goal': 1}

    # reset state in a random non-goal state
    def reset(self):
        while True:
            agent_x = randint(1, self.x_limit)
            agent_y = randint(1, self.y_limit)
            goal_x = randint(1, self.x_limit)
            goal_y = randint(1, self.y_limit)
            if (agent_x, agent_y) != (goal_x, goal_y)
                break
        return {'agent': (agent_x, agent_y), 'goal': (goal_x, goal_y)}

    # get the succeeding state resulting from taking a given action in a given
    # state
    def state_transition(self, state, action):
        # account for the agent's movement
        (agent_x, agent_y) = state['agent']
        if action == "up":
            next_agent_state = (agent_x, agent_y-1)
        elif action == "down":
            next_agent_state = (agent_x, agent_y+1)
        elif action == "left":
            next_agent_state = (agent_x-1, agent_y)
        elif action == "right":
            next_agent_state = (agent_x+1, agent_y)
        # correct for the possibility of agent state going out of bounds
        if next_agent_state[0] > self.x_limit:
            next_agent_state = (self.x_limit, next_agent_state[1])
        elif next_agent_state[1] > self.y_limit:
            next_agent_state = (next_agent_state[0], self.y_limit)
        elif next_agent_state[0] < 1:
            next_agent_state = (1, next_agent_state[1])
        elif next_agent_state[1] < 1:
            next_agent_state = (next_agent_state[0], 1)

        # acount for the goal's movement
        (goal_x, goal_y) = state['goal']
        goal_action = self.action_space[randint(0, len(self.action_space)-1)]
        if goal_action == "up":
            next_goal_state = (goal_x, goal_y-1)
        elif goal_action == "down":
            next_goal_state = (goal_x, goal_y+1)
        elif goal_action == "left":
            next_goal_state = (goal_x-1, goal_y)
        elif goal_action == "right":
            next_goal_state = (goal_x+1, goal_y)
        # correct for the possibility of agent state going out of bounds
        if next_goal_state[0] > self.x_limit:
            next_goal_state = (self.x_limit, next_goal_state[1])
        elif next_goal_state[1] > self.y_limit:
            next_goal_state = (next_goal_state[0], self.y_limit)
        elif next_goal_state[0] < 1:
            next_goal_state = (1, next_goal_state[1])
        elif next_goal_state[1] < 1:
            next_goal_state = (next_goal_state[0], 1)

        return hashabledict({'agent': next_agent_state,
                             'goal': next_goal_state})

    # get step data resulting from taking a given action in a given state
    def step(self, state, action):
        next_state = self.state_transition(state, action)
        reward = self.reward(next_state)
        done = self.done(next_state)
        return (next_state, reward, done)

    # get the reward that results from arriving it to a particular state
    def reward(self, state):
        if state['agent'] == next_state['goal']
            return self.reward_map['found_goal']
        else:
            return 0

    # determine whether the episode is complete
    def done(self, state):
        if state['agent'] == state['goal']:
            return True
        else:
            return False

    # initialize the Q-table to value 0 for all state-action pairs
    def initialize_q_table(self):
        q_table = {}
        for agent_x in range(1, self.x_limit+1):
            for agent_y in range(1, self.y_limit+1):
                for action in self.action_space:
                    for goal_x in range(1, self.x_limit+1):
                        for goal_y in range(1, self.y_limit+1):
                            q_table[
                            (
                                hashabledict(
                                    {
                                        'agent': (agent_x, agent_y),
                                        'goal': (goal_x, goal_y)
                                    }
                                ),
                                action
                            )] = 0
        return q_table
