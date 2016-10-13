from Environment import Environment
from Q_Table import Q_Table
#from Simulation import run_comparative_simulation

# define parameters
gamma = 0.9
num_training_episodes = 10000

def run_episode(state, q_table, env):
    episode_data = []
    done = False
    while not done:
        # get best available action according to q_table
        action = q_table.get_best_action(state)
        # take chosen action
        (next_state, reward, done) = env.step(state, action)
        # record step data
        episode_data.append((state, action, reward, next_state))
        state = next_state
    return episode_data

def train_on_episode(episode_data, q_table):
    for _ in range(len(episode_data)):
        step_data = episode_data.pop()
        q_table.update(step_data, gamma)

# training_log is a list of training episodes
def determine_performance_improvement(training_log):
    interval_1 = training_log[0:10]
    interval_2 = training_log[len(training_log)-10:len(training_log)]
    print sum(interval_1)/float(len(interval_1))
    print sum(interval_2)/float(len(interval_2))


def main():
    # initializations
    training_log = []
    env = Environment()
    q_table = Q_Table(env)

    # training loop
    for _ in range(num_training_episodes):
        # reset the state to a random position in the environment
        initial_state = env.reset()
        # run through an episode to completion, returning the sequence of state,
        # action and reward values
        episode_data = run_episode(initial_state, q_table, env)
        training_log.append(len(episode_data))
        # update the q_table on the data from the most recent episode
        train_on_episode(episode_data, q_table)

    # assess training results
    determine_performance_improvement(training_log)
    # todo: run comparative simulations
    #run_comparative_simulations(training_log[0], training_log[-1])

main()
