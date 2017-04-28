from Classic_Env import Classic_Env
from Moving_Goal_Env import Moving_Goal_Env
from Q_Network import Q_Network
from Q_Table import Q_Table
from agent_animation import run_animation



def main():
    # define training parameters
    learning_parameters = {'gamma': 0.5, 'eta': 0.5}
    training_epochs = 1
    episodes_per_epoch = 1
    # initializations
    training_log = []
    env = Moving_Goal_Env()
    #q_function = Q_Network(env)
    q_function = Q_Table(env)

    # visualize pre-training q_function
    #q_function.pprint()

    # training loop
    for i in range(1, training_epochs + 1):
        print "epoch " + str(i)
        for _ in range(episodes_per_epoch):
            # reset the state to a random position in the environment
            initial_state = env.reset()
            # run through an episode to completion, returning the sequence of state,
            # action and reward values
            episode_data = run_episode(initial_state, q_function, env)
            training_log.append(episode_data)
            # train the q_network on the recently completed episode
            q_function.episode_train(episode_data, **learning_parameters)

    # visualize post-training q_function
    #q_function.pprint()

    print_training_results_summary(training_log)

    # run comparative animations, before and after training
    run_animation(env.x_limit, env.y_limit, training_log[0])
    run_animation(env.x_limit, env.y_limit, training_log[-1])


def run_episode(state, q_function, env):
    episode_data = []
    done = False
    while not done:
        # get best available action according to q-network
        action = q_function.get_best_action_prob(state)
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
    n = 50
    # compare the average length of the first n training episodes with the
    # average length of the last n training episodes
    first_interval = lengths_of_episodes[0:n]
    early_performance = sum(first_interval)/float(len(first_interval))
    print "The average length of the first " + str(n) + " episodes was " + \
          str(int(early_performance))
    last_interval = lengths_of_episodes[num_episodes-n:num_episodes]
    late_performance = sum(last_interval)/float(len(last_interval))
    print "The average length of the last " + str(n) + " episodes was " + \
          str(int(late_performance))


main()
