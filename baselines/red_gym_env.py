
import sys
import uuid 
import os
from math import floor
import json

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib

import gym
from gym import spaces
from pyboy.utils import WindowEvent

class RedGymEnv(gym.Env):


    def __init__(
        self, config=None):

        '''
        if config is None:
            config = {
                'headless': True, 'save_final_state': False,
                'action_freq': 5, 'init_state':'init.state', 'max_steps': 100,  'print_rewards': False,
                'gb_path': './PokemonRed.gb', 'debug': False, 'sim_frame_dist': 1500000.0
            }
        '''

        self.debug = config['debug']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.downsample_factor = 4
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8]
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 2500)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
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

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 4
        self.output_full = (
            self.output_shape[0] + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

        self.recent_memory = np.zeros(self.output_shape[1]*self.memory_height, dtype=np.uint8)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0 if config['headless'] else 4)
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
        self.reset_count += 1
        return self.render()

    def render(self, reduce_res=True, add_memory=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(), 
                        pad,
                        self.create_recent_memory(),
                        pad,
                        game_pixels_render
                    ),
                    axis=0)
        return game_pixels_render

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
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            self.pyboy.tick()

        obs_memory = self.render()

        obs_flat = obs_memory[
            2 * (self.memory_height + self.mem_padding):, ...].flatten().astype(np.float32)

        reward = 0

        if self.knn_index.get_current_count() == 0:
            reward = 1
            self.knn_index.add_items(
                obs_flat, np.array([self.knn_index.get_current_count()])
            )

        # Query dataset, k - number of closest elements
        labels, distances = self.knn_index.knn_query(obs_flat, k = 1)
        if distances[0] > self.similar_frame_dist:
            reward = 1
            #if self.print_rewards:
            #    print('-', end='', flush=True)
            self.knn_index.add_items(
                obs_flat, np.array([self.knn_index.get_current_count()])
            )

        self.recent_memory = np.roll(self.recent_memory, 1)
        self.recent_memory[0] = reward * 255
        #reward = (self.knn_index.get_current_count() / 100) + self.reward_range[0]

        if self.debug:
            print(frame)
            print(
                f'{self.knn_index.get_current_count()} '
                f'total frames indexed, current closest is: {distances[0]}'
            )

        self.step_count += 1

        #done = self.step_count >= self.max_steps
        done = False
        if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
            done = True

        '''
        if self.print_rewards and self.step_count % 20 == 0:
            steps = 15
            r_get = int((reward - self.reward_range[0]) / (self.reward_range[1] - self.reward_range[0]) * steps)
            r_not_get = steps - r_get
            for i in range(r_get):
                print('-', end ='', flush = True)
            for i in range(r_not_get):
                print(' ', end ='', flush = True)
            print('|', end ='', flush = True)
        '''

        if self.print_rewards and done:
            raw_r = self.knn_index.get_current_count()
            print(f'\nenv: {self.instance_id} - {raw_r}', flush=True)
            if self.save_final_state:
                os.makedirs('final_states', exist_ok=True)
                plt.imsave(
                    f'final_states/frame_r{raw_r}_{self.reset_count}_small.jpeg', 
                    obs_memory)
                plt.imsave(
                    f'final_states/frame_r{raw_r}_{self.reset_count}_full.jpeg', 
                    self.render(reduce_res=False))

        if done:
            self.all_runs.append(self.knn_index.get_current_count())
            with open(f'all_runs_{self.instance_id}.json', 'w') as f:
                json.dump(self.all_runs, f)

        return obs_memory, reward, done, {}

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        total_reward = self.knn_index.get_current_count()
        col_steps = self.col_steps
        row = floor(total_reward / (h * col_steps))
        memory = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        memory[:, :row, :] = 255
        row_covered = row * h * col_steps
        col = floor((total_reward - row_covered) / col_steps)
        memory[:col, row, :] = 255
        col_covered = col * col_steps
        last_pixel = floor(total_reward - row_covered - col_covered) 
        memory[col, row, :] = last_pixel * (255 // col_steps)
        return memory

    def create_recent_memory(self):
        return np.stack((self.recent_memory.reshape(
            self.output_shape[1], 
            self.memory_height).T,)*3, axis=-1)