from Environment import Environment
from Q_Network import Q_Network
from agent_animation import run_animation

# define parameters
gamma = 0.9
eta = 1
training_epochs = 1
episodes_per_epoch = 1

def main():
    # initializations
    training_log = []
    env = Environment()
    q_network = Q_Network(env)

    # training loop
    for _ in range(training_epochs):
        print "epoch"
        for _ in range(episodes_per_epoch):
            # reset the state to a random position in the environment
            initial_state = env.reset()
            # run through an episode to completion, returning the sequence of state,
            # action and reward values
            episode_data = run_episode(initial_state, q_network, env)
            training_log.append(episode_data)
        # extract the training data for the epoch
        epoch_data = training_log[-episodes_per_epoch:]
        #update the q_network
        q_network.train(epoch_data, gamma, eta)

    #q_network.print_q_function()

    print_training_results_summary(training_log)

    # run comparative animations, before and after training
    run_animation(env.x_limit, env.y_limit, training_log[0])
    run_animation(env.x_limit, env.y_limit, training_log[-1])


def run_episode(state, q_network, env):
    episode_data = []
    done = False
    while not done:
        # get best available action according to q-network
        action = q_network.get_best_action_prob(state)
        # take chosen action
        (next_state, reward, done) = env.step(state, action)
        # record step data
        episode_data.append((state, action, reward, next_state))
        state = next_state
    return episode_data


# take as input a list of training episodes and print a summary of the results
# of training
def print_training_results_summary(training_log):
    num_episodes = len(training_log)
    lengths_of_episodes = [len(episode_data) for episode_data in training_log]
    n = 10
    # compare the average length of the first n training episodes with the
    # average length of the last n training episodes
    first_interval = lengths_of_episodes[0:n]
    early_performance = sum(first_interval)/float(len(first_interval))
    print "The average length of the first " + str(n) + " episodes was " + \
          str(early_performance)
    last_interval = lengths_of_episodes[num_episodes-n:num_episodes]
    late_performance = sum(last_interval)/float(len(last_interval))
    print "The average length of the last " + str(n) + " episodes was " + \
          str(late_performance)


main()
