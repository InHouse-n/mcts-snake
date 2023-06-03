from collections import namedtuple
from enum import Enum

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

Point = namedtuple("Point", "x, y")


class SnakeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=30):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "snake": spaces.Box(0, size - 1, dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"snake": self._snake_location, "target": self._target_location, "score": self.score}

    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # start of in the middle going to the right
        self._head = Point(self.size / 2, self.size / 2)
        self._snake_location = [
            self._head,
            Point(self._head.x - 1, self._head.y),
            Point(self._head.x - 2, self._head.y),
        ]

        self._place_food()
        self.score = 0
        observation = self._get_obs()
        

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # move the snake
        self._head = Point(self._head.x + direction[0], self._head.y + direction[1])
        self._snake_location.insert(0, self._head)

        # check if game is over
        reward = 0
        done = False
        if self._is_collision():
            done = True
            reward = -1

        # place new food or move
        if self._head == self._target_location:
            self.score += 1
            reward = 1
            self._place_food()
        else:
            self._snake_location.pop()

        observation = self._get_obs()
        

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done

    def _is_collision(self, pt=None):
        collision = False
        if pt is None:
            pt = self._head
        # hits boundary
        if pt.x > self.size - 1 or pt.x < 0 or pt.y > self.size - 1 or pt.y < 0:
            collision = True
        # hits itself
        if pt in self._snake_location[1:]:
            collision = True
        return collision

    def _place_food(self):
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = Point(
            self.np_random.integers(0, self.size, dtype=int),
            self.np_random.integers(0, self.size, dtype=int),
        )
        while self._target_location in self._snake_location:
            self._target_location = Point(
                self.np_random.integers(0, self.size, dtype=int),
                self.np_random.integers(0, self.size, dtype=int),
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * np.array(self._target_location),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        for snake in self._snake_location:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (np.array(snake) + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            # add score
            font = pygame.font.Font("src/mcts_snake/arial.ttf", 25)
            text = font.render("Score: " + str(self.score), True, (255, 255, 255))
            self.window.blit(canvas, canvas.get_rect())
            self.window.blit(text, [0, 0])
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    
    def next_state(self, state, action):
        """
        Given the current state of the environment and the action that is
        performed in that state, returns the resulting state.
        :param state: Current state of the environment.
        :param action: Action that is performed in that state.
        :return: Resulting state.
        """
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]


        # get snake and food location
        snake = state['snake']
        food = state['target']
        score = state['score']

        #move the snake
        snake.insert(0, snake[0].x + direction[0], snake[0].y + direction[1])

        # if snake eat the food place new one
        if snake[0] == food:
            score =+ 1
            while food in snake:
                food = Point(self.np_random.integers(0, self.size, dtype=int),
                self.np_random.integers(0, self.size, dtype=int),)
        
        return {"snake": snake, "target": food, "score": score}
    
    def is_done_state(self,state):
        """
        Given the state and the index of the current step, returns whether
        that state is the end of an episode, i.e. a done state.
        :param state: Current state.
        :param step_idx: Index of the step at which the state occurred.
        :return: True, if the step is a done state, False otherwise.
        """
        collision = False
        
        
        # get snake location
        snake = state['snake']
        snake_head = snake[0]
        
        # check if it hit anything
        
        # hits boundary
        if snake_head.x > self.size - 1 or snake_head.x < 0 or snake_head.y > self.size - 1 or snake_head.y < 0:
            collision = True
        # hits itself
        if snake_head in snake[1:]:
            collision = True
        return collision
    
    
    def initial_state(self):
        """
        Returns the initial state of the environment.
        """
        snake_head = Point(self.size / 2, self.size / 2)
        snake = [snake_head,
                 Point(snake_head.x - 1, snake_head.y),
                 Point(snake_head.x - 2, snake_head.y),
                 ]

        food = Point(self.np_random.integers(0, self.size, dtype=int),
                self.np_random.integers(0, self.size, dtype=int),)
        while food in snake:
                food = Point(self.np_random.integers(0, self.size, dtype=int),
                self.np_random.integers(0, self.size, dtype=int),)
        score = 0

        return {"snake": snake, "target": food, "score": score}

    @staticmethod
    def get_obs_for_states(states):
        """
        Some environments distinguish states and observations. An observation
        can be a subset (e.g. in Poker, state is all cards in game, observation
        is cards on player's hand) or superset of the state (i.e. observations
        add additional information).
        :param states: List of states.
        :return: Numpy array of observations.
        """
        return np.array(states)

    
    @staticmethod
    def get_return(state):
        """
        Returns the return that the agent has achieved so far when he is in
        a given state after a given number of steps.
        :param state: Current state that the agent is in.
        :param step_idx: Index of the step at which the agent reached that
        state.
        :return: Return the agent has achieved so far.
        """
        return state['score']
        
