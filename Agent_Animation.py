"""
Pygame program to animate the path taken by a Q-Learning agent at different
points in its learning process.

This code was adapted from here:
http://programarcadegames.com/index.php?chapter=array_backed_grids
"""

import pygame
import time

# set HEIGHT and WIDTH of the screen
WINDOW_SIZE = [255, 255]

# set title of screen
TITLE = "A Q-Learning Agent Tries to Find its Way Home..."

# define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# this sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40

# this sets the margin between each cell
MARGIN = 5

# utility method to re-generate a black grid before each frame is drawn
def generate_black_grid(num_rows, num_cols):
    grid = []
    for row in range(num_rows):
        grid.append([])
        for column in range(num_cols):
            grid[row].append(0)
    return grid


def run_animation(num_cols, num_rows, episode_data):
    # some initializations
    pygame.init()
    pygame.display.set_caption(TITLE)
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # loop until the user clicks the close button.
    done = False

    # used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # time elapsed since last action was shown
    time_elapsed_since_action_update = 0
    state_index = 0

    # length of the episode
    episode_length = len(episode_data)

    # main loop
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # dt <- ms since clock was initialized or since last tick()
        dt = clock.tick()
        time_elapsed_since_action_update += dt

        # if more than 200 ms have elapsed since last action update, then update
        if time_elapsed_since_action_update > 200 and \
           state_index < episode_length:

            # get state and inc state_index
            state = episode_data[state_index][0]
            state_index += 1

            # extract x, y, and convert to 0-based indexing
            (x, y) = (state[0]-1, state[1]-1)

            # set next action
            grid = generate_black_grid(num_rows, num_cols)
            grid[y][x] = 1

            time_elapsed_since_action_update = 0

            # set the screen background
            screen.fill(BLACK)

            # draw the grid
            for row in range(num_rows):
                for column in range(num_cols):
                    color = WHITE
                    if grid[row][column] == 1:
                        color = GREEN
                    pygame.draw.rect(screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])
        # freeze on the last state of the last episode
        elif state_index >= episode_length:
            time.sleep(.200)
            state = episode_data[-1][-1]
            # extract x, y, and convert to 0-based indexing
            (x, y) = (state[0]-1, state[1]-1)

            # set next action
            grid = generate_black_grid(num_rows, num_cols)
            grid[y][x] = 1

            time_elapsed_since_action_update = 0

            # set the screen background
            screen.fill(BLACK)

            # draw the grid
            for row in range(num_rows):
                for column in range(num_cols):
                    color = WHITE
                    if grid[row][column] == 1:
                        color = GREEN
                    pygame.draw.rect(screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])

        # update screen
        pygame.display.flip()

    # when done main loop, quit the simulation
    pygame.quit()
