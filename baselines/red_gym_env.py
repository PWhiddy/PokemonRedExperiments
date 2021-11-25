
import sys
import uuid 

import numpy as np
from pyboy import PyBoy, botsupport
import hnswlib

import gym
from gym import spaces
from pyboy.utils import WindowEvent

class RedGymEnv(gym.Env):


    def __init__(
        self, config=None):

        if config is None:
            self.config = {
                'headless':True, 
                'action_freq': 5, 'init_state':'init.state', 'max_steps': 100,  'print_rewards': False,
                'gb_path': './PokemonRed.gb', 'debug': False, 'sim_frame_dist': 1500000.0
            }

        self.debug = config['debug']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.downsample_factor = 4
        self.similar_frame_dist = config['sim_frame_dist']
        self.episode_count = 1
        self.instance_id = str(uuid.uuid4())[:8]


        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (-0.5, 1.5)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PASS
        ]

        self.release_arrow = [            
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(40, 36, 3), dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0 if config['headless'] else 2)
        self.reset()

    def reset(self):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

        self.step_count = 0

        return self.render()

    def render(self):
        game_pixels_render = self.screen.screen_ndarray() # (160, 144, 3)
        output_size = (40, 36)
        bin_size = (4, 4)
        small_image = game_pixels_render.reshape(
            (output_size[0], bin_size[0], 
            output_size[1], bin_size[1], 3)).max(3).max(1)
        return small_image
        

    def step(self, action):

        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == self.act_freq - 1:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
            self.pyboy.tick()

        obs = self.render()

        obs_flat = obs.flatten().astype(np.float)

        if self.knn_index.get_current_count() == 0:
            self.knn_index.add_items(
                obs_flat, np.array([self.knn_index.get_current_count()])
            )

        # Query dataset, k - number of closest elements
        labels, distances = self.knn_index.knn_query(obs_flat, k = 1)
        if distances[0] > self.similar_frame_dist:
            self.knn_index.add_items(
                obs_flat, np.array([self.knn_index.get_current_count()])
            )

        reward = (self.knn_index.get_current_count() / 100) + self.reward_range[0]

        if self.debug:
            print(frame)
            print(
                f'{self.knn_index.get_current_count()} '
                f'total frames indexed, current closest is: {distances[0]}'
            )

        self.step_count += 1
        if self.print_rewards and self.step_count % 20 == 0:
            steps = 15
            r_get = int((reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0]) * steps)
            r_not_get = steps - r_get
            for i in range(r_get):
                print('-', end ='', flush = True)
            for i in range(r_not_get):
                print(' ', end ='', flush = True)
            print('|', end ='', flush = True)
        if self.print_rewards and self.step_count == self.max_steps:
            print(f' {reward:.3f}', flush=True)

        return obs, reward, self.step_count >= self.max_steps, {}


