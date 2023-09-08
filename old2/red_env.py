import sys
import uuid 

import numpy as np
from PIL.Image import init
from numpy.core.numeric import roll
from pyboy import PyBoy, WindowEvent
import hnswlib

from rollout import Rollout

class RedEnv:

    def __init__(
        self, headless=True, rollout_dir='rollouts',
        action_freq=5, init_state='init.state', 
        gb_path='./PokemonRed.gb', debug=False):

        self.debug = debug
        self.vec_dim = 1080 #4320 #1000
        self.num_elements = 20000 # max
        self.init_state = init_state
        self.act_freq = action_freq
        self.downsample_factor = 8
        self.similar_frame_dist = 1500000.0
        self.episode_count = 1
        self.rollout_dir = rollout_dir
        self.instance_id = str(uuid.uuid4())[:8]

        head = 'headless' if headless else 'SDL2'

        self.pyboy = PyBoy(
                gb_path,
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.reset_game()

        self.pyboy.set_emulation_speed(0)

    def reset_game(self):
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

    def play_episode(self, agent, max_episode_steps):

        frame = 0
        self.reset_game()
        agent.reset()

        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)

        rollout = Rollout(
            f'{self.instance_id}_r{self.episode_count}',
            agent.get_name(), basepath=self.rollout_dir
        )

        while not self.pyboy.tick() and frame < max_episode_steps:

            if frame % self.act_freq == 0:

                game_pixels_render = self.pyboy.screen_image()
                x, y = game_pixels_render.size
                state_np = np.array(game_pixels_render)

                next_action = agent.get_action(state_np, rollout)
                self.pyboy.send_input(next_action)

                rollout.add_state_action_pair(state_np, next_action)

                state = np.array(game_pixels_render.resize(
                    (x//self.downsample_factor, 
                    y//self.downsample_factor)
                    )).reshape(-1)

                if self.knn_index.get_current_count() == 0:
                    self.knn_index.add_items(
                        state, np.array([self.knn_index.get_current_count()])
                    )

                # Query dataset, k - number of closest elements
                labels, distances = self.knn_index.knn_query(state, k = 1)
                if distances[0] > self.similar_frame_dist:
                    self.knn_index.add_items(
                        state, np.array([self.knn_index.get_current_count()])
                    )

                rollout.add_reward(self.knn_index.get_current_count())

                if self.debug:
                    print(frame)
                    print(
                        f'{self.knn_index.get_current_count()} '
                        f'total frames indexed, current closest is: {distances[0]}'
                    )

            frame += 1

        rollout.save_to_file()
        self.episode_count += 1

        return self.knn_index.get_current_count()

        #tile_map = pyboy.get_window_tile_map() # Get the TileView object for the window.

        #index_map = tile_map.get_tile_matrix()

