
import sys
import uuid 
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

from stream_agent_wrapper import StreamWrapper
from memory_addresses import *

class RedGymEnv(Env):
    def __init__(
        self, config=None):

        self.s_path = config['session_path']
        self.headless = config['headless']
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.extra_buttons = False if 'extra_buttons' not in config else config['extra_buttons']
        self.instance_id = str(uuid.uuid4())[:8] if 'instance_id' not in config else config['instance_id']
        self.s_path.mkdir(exist_ok=True)

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

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


        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

        head = 'headless' if config['headless'] else 'SDL2'

        #log_level("ERROR")
        self.pyboy = PyBoy(
                config['gb_path'],
                debugging=False,
                disable_input=False,
                window_type=head,
                hide_window='--quiet' in sys.argv,
            )

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not config['headless']:
            self.pyboy.set_emulation_speed(6)
            
        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.step_count = 0

        self.best_levels = 0
        self.best_coords = 0

        self.seen_coords = {}
        
        return self.render(), {}

    def update_seen_coords(self):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_N_ADDRESS)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[coord_string] = self.step_count

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        return np.array([1.0], dtype=np.uint8)
    
    def step(self, action):

        self.run_action_on_emulator(action)

        level_rew = 0
        levels = [self.read_m(a) for a in LEVELS_ADDRESSES]
        if levels > self.best_levels:
            level_rew = (levels - self.best_levels) * 1.0
            self.best_levels = levels

        coord_rew = 0
        coord_count = len(self.seen_coords)
        if coord_count > self.best_coords:
            coord_rew = (coord_count - self.best_coords) * 0.04
            self.best_coords = coord_count

        total_rew = level_rew + coord_rew

        self.step_count += 1

        return self.render(), total_rew, False, False, {}

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if self.headless:
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
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "pw-random", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#332299", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_{sess_id}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = 32  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    for step in range(2**30):
        #action = env.action_space.sample()  # agent policy that uses the observation and info
        actions = np.random.randint(low=0, high=5, size=num_cpu)
        #print(action)
        things = env.step(actions)
        #print(f"\r{step}")