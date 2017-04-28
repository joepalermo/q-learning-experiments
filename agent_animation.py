"""
Pygame program to animate the path taken by a Q-Learning agent at different
points in its learning process.

This code was adapted from here:
http://programarcadegames.com/index.php?chapter=array_backed_grids
"""

import pygame
import time

# set title of screen
TITLE = "A Q-Learning Agent Seeks Home..."

# this sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40

# this sets the margin between each cell
MARGIN = 5

# define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# set the update interval to 200 ms
UPDATE_INTERVAL = 200

# run an animation of an episode
def run_animation(num_cols, num_rows, episode_data):
    # some pygame initializations
    pygame.init()
    pygame.display.set_caption(TITLE)
    window_width = WIDTH * num_rows + MARGIN * (num_rows + 1)
    window_height = HEIGHT * num_cols + MARGIN * (num_cols + 1)
    window_size = [window_width, window_height]
    screen = pygame.display.set_mode(window_size)

    # for keeping track of time, to determine when to update the screen
    clock = pygame.time.Clock()
    time_since_update = 0

    # get a list of states in the episode
    episode_states = get_episode_states(episode_data)
    episode_length = len(episode_states)
    state_index = 0

    # animation loop
    done = False
    while not done:

        # dt <- ms since clock was initialized or since last tick()
        dt = clock.tick()
        time_since_update += dt

        # if more than 200 ms have elapsed since last update, and if there is
        # more step data in the episode, then perform an update
        if time_since_update > UPDATE_INTERVAL and state_index < episode_length:

            # get state and inc state_index
            state = episode_states[state_index]
            state_index += 1

            # convert state to an analogous dict with 0-based indexing
            entity_map = {}
            entity_map['agent'] = (state['agent'][0]-1, state['agent'][1]-1)
            entity_map['goal'] = (state['goal'][0]-1, state['goal'][1]-1)

            # generate an updated representation of the grid
            grid = generate_grid(num_rows, num_cols, entity_map)

            # update the screen with the current contents of the grid
            update_screen(screen, grid)
            time_since_update = 0

        # test whether to exit the loop
        for event in pygame.event.get():
            # if the user clicked close, then exit the animation loop
            if event.type == pygame.QUIT:
                done = True

    # when done main loop, quit pygame
    pygame.quit()

# utility functions ------------------------------------------------------------

# get a list of episode states from the episode data
def get_episode_states(episode_data):
    episode_states = []
    num_steps = len(episode_data)
    for i in range(num_steps):
        # note that episode_data[i] is of form:
        # (state, action, reward, next_state)
        state = episode_data[i][0]
        episode_states.append(state)
    # the final state of the episode only appears as a 'next_state' value
    episode_states.append(episode_data[-1][3])
    return episode_states

# re-generate the grid given its size and the positions of various entities
def generate_grid(num_rows, num_cols, entity_map):
    (x,y) = agent_position
    grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    for entity,(x,y) in entity_map.items():
        grid[y][x] = entity
    return grid

# update the screen with the current contents of the grid
def update_screen(screen, grid):
    # set the screen background
    screen.fill(BLACK)
    num_rows = len(grid)
    num_cols = len(grid[0])
    # draw each square in the grid
    for row in range(num_rows):
        for column in range(num_cols):
            color = WHITE
            # color a goal square green
            if grid[row][column] == 'agent':
                color = BLUE
            elif grid[row][column] == 'goal':
                color = GREEN
            # define the square (rectangle) to draw
            rectangle_left_edge = (MARGIN + WIDTH) * column + MARGIN
            rectangle_top_edge = (MARGIN + HEIGHT) * row + MARGIN
            rectangle = [rectangle_left_edge, rectangle_top_edge, WIDTH, HEIGHT]
            # draw the square (rectangle)
            pygame.draw.rect(screen, color, rectangle)
    # update screen
    pygame.display.flip()
