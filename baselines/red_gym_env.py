import sys

import numpy as np
from pyboy import PyBoy
import uuid
import json
import pandas as pd
from pathlib import Path
from renderer import Renderer
from rewards import Reward
from reader_pyboy import ReaderPyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


class RedGymEnv(Env):

    def __init__(self, config=None):

        self.debug = config['debug']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path = config['session_path']
        self.save_final_state = config['save_final_state']
        self.save_video = config['save_video']
        self.headless = config['headless']
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.fast_video = config['fast_video']
        self.video_interval = 256 * self.act_freq
        self.print_rewards = config['print_rewards']
        self.use_screen_explore = True if 'use_screen_explore' not in config else config['use_screen_explore']

        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
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

        head = 'headless' if config['headless'] else 'SDL2'

        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)

        self.reader = ReaderPyBoy(self.pyboy)

        # Rewards
        self.reward_service = Reward(config, self.reader)
        self.renderer = Renderer(self.s_path, self.pyboy, self.reward_service, self.instance_id)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.renderer.output_full, dtype=np.uint8)

        self.reset()

    def render(self):
        return self.renderer.render()

    def reset(self, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.reward_service.reset()

        self.renderer.reset()
        if self.save_video:
            self.renderer.save_video(self.reset_count)

        self.agent_stats = []

        self.step_count = 0

        self.reset_count += 1
        return self.render(), {}

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        self.renderer.recent_frames = np.roll(self.renderer.recent_frames, 1, axis=0)

        # OBSERVATION

        obs_memory = self.render()
        obs_flat = self.renderer.get_obs_flat(obs_memory)

        # REWARD

        reward_delta, new_prog = self.reward_service.update_rewards(obs_flat, self.step_count)
        if self.print_rewards and self.step_count % 100 == 0:
            self.reward_service.print_rewards(self.step_count)
        if reward_delta < 0 and self.reader.read_hp_fraction() > 0:
            self.renderer.save_screenshot('neg_reward', self.reward_service.total_reward, self.reset_count)

        # shift over short term reward memory
        self.renderer.recent_memory = np.roll(self.renderer.recent_memory, 3)
        self.renderer.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.renderer.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.renderer.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        # DONE

        done = self.check_if_done()
        if self.step_count % 50 == 0:
            self.renderer.save_and_print_info()

        if done:
            self.all_runs.append(self.reward_service.get_game_state_rewards())
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')

            if self.print_rewards:
                print('', flush=True)
                if self.save_final_state:
                    self.renderer.save_final_state(obs_memory, self.reset_count, self.reward_service.total_reward)

            if self.save_video and done:
                self.renderer.full_frame_writer.close()
                self.renderer.model_frame_writer.close()

        self.step_count += 1
        return obs_memory, reward_delta * 0.1, False, done, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.renderer.save_video and self.headless:
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
                self.renderer.add_video_frame()
            if i == self.act_freq - 1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.renderer.add_video_frame()

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
            'final_total_reward': self.reward_service.total_reward,
            'party_size': self.reader.read_party_size_address(),
            'levels': levels,
            'levels_sum': sum(levels),
            'seen_pokemons': self.reward_service.seen_pokemons,
            'ptypes': self.reader.read_party(),
            'hp': self.reader.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.reward_service.died_count,
            'badge': self.reader.get_badges(),
            'event': self.reward_service.max_event,
            'healr': self.reward_service.total_healing
        })

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.renderer.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        return done
