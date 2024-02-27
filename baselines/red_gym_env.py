import sys
import uuid
from math import floor
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import mediapy as media
import pandas as pd
from rewards import Reward
from reader_pyboy import ReaderPyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


class RedGymEnv(Env):

    def __init__(
            self, config=None):

        self.debug = config['debug']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.headless = config['headless']
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']

        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        if self.extra_buttons:
            self.valid_actions.extend([
                WindowEvent.PRESS_BUTTON_START,
                WindowEvent.PASS
            ])

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
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        # log_level("ERROR")
        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )
        self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)

        self.reader = ReaderPyBoy(self.pyboy)

        # Rewards
        self.print_rewards = config['print_rewards']
        self.reward_service = Reward(config, self.reader, self.save_screenshot)

        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.reward_service.reset()

        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)

        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0],
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.agent_stats = []

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()

        self.step_count = 0

        self.reset_count += 1
        return self.render(), {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
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
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0)
        return game_pixels_render

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)

        # OBSERVATION

        obs_memory = self.render()
        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[frame_start:frame_start + self.output_shape[0], ...].flatten().astype(np.float32)

        # REWARD

        reward_delta, new_prog = self.reward_service.update_rewards(obs_flat, self.step_count)

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # DONE

        step_limit_reached = self.check_if_done()
        self.save_and_print_info(step_limit_reached, obs_memory, reward_delta)
        self.step_count += 1

        return obs_memory, reward_delta * 0.1, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(self.release_button[action - 4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))

    def append_agent_stats(self, action):
        x_pos = self.reader.read_x_pos()
        y_pos = self.reader.read_y_pos()
        map_n = self.reader.read_map_n()
        levels = self.reader.read_levels()
        if self.use_screen_explore:
            expl = ('frames', self.reward_service.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.reward_service.seen_coords))
        self.agent_stats.append({
            'step': self.step_count, 'x': x_pos, 'y': y_pos, 'map': map_n,
            'map_location': self.reader.get_map_location(),
            'last_action': action,
            'pcount': self.reader.read_party_size_address(),
            'levels': levels,
            'levels_sum': sum(levels),
            'ptypes': self.reader.read_party(),
            'hp': self.reader.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.reward_service.died_count,
            'badge': self.reader.get_badges(),
            'event': self.reward_service.max_event_rew,
            'healr': self.reward_service.total_healing_rew
        })

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w - 1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, hp, explore = self.reward_service.group_rewards_lvl_hp_explore(self.reward_service.get_game_state_rewards())
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)

        if self.reader.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory,
            '(w h) c -> h w c',
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            rewards_state = self.reward_service.get_game_state_rewards()
            for key, val in rewards_state.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.reward_service.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)

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
                    fs_path / Path(f'frame_r{self.reward_service.total_reward:.4f}_{self.reset_count}_small.jpeg'),
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.reward_service.total_reward:.4f}_{self.reset_count}_full.jpeg'),
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.reward_service.get_game_state_rewards())
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

    def save_screenshot(self, name):
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(
                f'frame{self.instance_id}_r{self.reward_service.total_reward:.4f}_{self.reset_count}_{name}.jpeg'),
            self.render(reduce_res=False)
        )
