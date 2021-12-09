
import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media

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
        self.session_name = config['session_name']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.vec_dim = 4320 #1000
        self.headless = config['headless']
        self.num_elements = 20000 # max
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.video_interval = 2048 * self.act_freq
        self.downsample_factor = 4
        self.similar_frame_dist = config['sim_frame_dist']
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8]
        self.s_path = Path(f'session_{self.session_name}')
        self.s_path.mkdir(exist_ok=True)
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
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] + 2 * (self.mem_padding + self.memory_height),
                            self.output_shape[1],
                            self.output_shape[2]
        )

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

        self.pyboy.set_emulation_speed(0 if config['headless'] else 6)
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

        self.recent_memory = np.zeros(self.output_shape[1]*self.memory_height, dtype=np.uint8)

        self.run_frames = []
        self.progress_reward = 1
        self.explore_reward = 1
        self.total_reward = 1
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

        self.run_action_on_emulator(action)

        obs_memory = self.render()

        # trim off memory from frame for knn index
        obs_flat = obs_memory[
            2 * (self.memory_height + self.mem_padding):, ...].flatten().astype(np.float32)

        self.update_frame_knn_index(obs_flat)

        new_reward = self.update_reward()

        # shift over short term reward memory 1 slot
        self.recent_memory = np.roll(self.recent_memory, 1)
        self.recent_memory[0] = min(new_reward * 8, 255)

        done = self.check_if_done()

        self.save_and_print_info(done, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward, done, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video:
                self.run_frames.append(self.render(reduce_res=False))
            self.pyboy.tick()

    def update_frame_knn_index(self, frame_vec):
        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0] > self.similar_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        self.explore_reward = self.knn_index.get_current_count()

        new_total = self.explore_reward + self.progress_reward #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        total_reward = self.total_reward
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

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            print(
                f'\rstep: {self.step_count} explore reward: {self.explore_reward} \
                prog reward: {self.progress_reward} total: {self.total_reward}\
                    ', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video and len(self.run_frames) % self.video_interval == 0:
            # save frames as video
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            out_frames = np.array(self.run_frames)
            f_path = base_dir / Path(f'run_r_{self.total_reward}_{self.reset_count}_s{self.step_count}').with_suffix('.mp4')
            media.write_video(f_path, out_frames, fps=60)
            self.run_frames = []

        if done:
            self.all_runs.append(self.total_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
    
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        num_poke = self.pyboy.get_memory_value(int(0xD163))
        poke_levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        level_sum = max(sum(poke_levels) - 5, 0) # subtract starting pokemon level
        poke_xps = [self.poke_xp(a) for a in [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]]
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in range(0xD30A, 0xD31D)])
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            print(f'seen_poke_count : {seen_poke_count}')
        return (0.333*sum(poke_xps) + level_sum * 100 + seen_poke_count * 150) + 1

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")

    def poke_xp(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
