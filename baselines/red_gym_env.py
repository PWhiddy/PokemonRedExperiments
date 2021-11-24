
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
        self, headless=True, 
        action_freq=5, init_state='init.state', max_steps=100, 
        gb_path='./PokemonRed.gb', debug=False):

        self.debug = debug
        self.vec_dim = 4320 #1000
        self.num_elements = 20000 # max
        self.init_state = init_state
        self.act_freq = action_freq
        self.max_steps = max_steps
        self.downsample_factor = 4
        self.similar_frame_dist = 1500000.0
        self.episode_count = 1
        self.instance_id = str(uuid.uuid4())[:8]


        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 100)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PASS
        ]

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4320,), dtype=np.float)

        head = 'headless' if headless else 'SDL2'

        self.pyboy = PyBoy(
                gb_path,
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0)
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
        game_pixels_render = self.screen.screen_ndarray().astype(np.float) # (160, 144, 3)
        output_size = (40, 36)
        bin_size = (4, 4)
        small_image = game_pixels_render.reshape(
            (output_size[0], bin_size[0], 
            output_size[1], bin_size[1], 3)).max(3).max(1).flatten()
        return small_image
        

    def step(self, action):

        action = self.valid_actions[action]

        self.pyboy.send_input(action)
        for i in range(self.act_freq):
            self.pyboy.tick()

        obs = self.render()
                
        if self.knn_index.get_current_count() == 0:
            self.knn_index.add_items(
                obs, np.array([self.knn_index.get_current_count()])
            )

        # Query dataset, k - number of closest elements
        labels, distances = self.knn_index.knn_query(obs, k = 1)
        if distances[0] > self.similar_frame_dist:
            self.knn_index.add_items(
                obs, np.array([self.knn_index.get_current_count()])
            )

        reward = self.knn_index.get_current_count() / 100

        if self.debug:
            print(frame)
            print(
                f'{self.knn_index.get_current_count()} '
                f'total frames indexed, current closest is: {distances[0]}'
            )

        self.step_count += 1

        return obs, reward, self.step_count >= self.max_steps, {}


