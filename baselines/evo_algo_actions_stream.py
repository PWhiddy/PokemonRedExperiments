
import sys
import uuid 
from pathlib import Path

import numpy as np
from einops import reduce
from pyboy import PyBoy

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

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
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,72,80), dtype=np.float32)

        head = 'headless' if config['headless'] else 'SDL2'

        #log_level("ERROR")
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

        self.best_levels = 0
        self.best_coords = 0

        self.reset_count = 0
            
        self.reset()
    
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def reset(self, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        print(f"last round - levels: {self.best_levels} seen_coords: {self.best_coords}")

        self.step_count = 0

        self.best_levels = 0
        self.best_coords = 0

        self.seen_coords = {}

        self.reset_count += 1
        
        return self.render(), {}

    def update_seen_coords(self):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_N_ADDRESS)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[coord_string] = self.step_count

    def render(self):
        return reduce(self.screen.screen_ndarray().astype(np.float32) / 255.0, '(h 2) (w 2) c -> h w c', 'mean').transpose(2,0,1)
    
    def step(self, action):

        self.run_action_on_emulator(action)

        level_rew = 0
        levels = sum([self.read_m(a) for a in LEVELS_ADDRESSES]) - 6
        if levels > self.best_levels:
            level_rew = (levels - self.best_levels) * 1.0
            self.best_levels = levels

        self.update_seen_coords()

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
                "user": "pw-evo", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#662299", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        return env
    set_random_seed(seed)
    return _init

# Define the CNN architecture
class CNNPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNPolicy, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_layers(x).view(x.size()[0], -1)
        return self.fc(conv_out)


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


    checkpoint_dir = "evo_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    num_cpu = 32  # Also sets the number of episodes per training iteration
    envs = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    models = [CNNPolicy((3, 72, 80), envs.action_space.n) for _ in range(num_cpu)]  # Adjust input shape based on the actual env


    def mutate_model(model, strength):
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * strength
                param.add_(noise)

    def run_episodes(envs, models, steps_to_run):
        obs = envs.reset()
        total_rewards = np.zeros(num_cpu)
        for _ in range(steps_to_run):
            actions = []
            for model, ob in zip(models, obs):
                ob_tensor = torch.tensor(ob).unsqueeze(0).float()
                action_prob = model(ob_tensor)
                action = torch.argmax(action_prob, dim=1).item()
                actions.append(action)
            obs, rewards, dones, _ = envs.step(actions)
            total_rewards += rewards
        return total_rewards

    num_episodes = 100000  # Define the number of episodes

    ep_len = 10

    for episode in range(num_episodes):
        print(f"starting episode {episode}")
        rewards = run_episodes(envs, models, ep_len)
        ep_len += 2
        print(f"all rewards: {rewards}")
        bests = sorted(rewards)[-2:]
        print(f"getting best models - rewards {bests}")
        # Selection and reproduction logic as before, with necessary adjustments
        sorted_indices = np.argsort(rewards)[-2:]  # Get indices of the top 2 models
        top_models = [models[i] for i in sorted_indices]
        print("mutating models...")
        mutation_strengths = np.linspace(0.0001, 0.1, 15)
        new_models = []
        for i, strength in enumerate(mutation_strengths):
            for top_model in top_models:
                model_copy = copy.deepcopy(top_model)
                mutate_model(model_copy, strength)
                new_models.append(model_copy)
        
        # Replace the models except for the top 2
        models = top_models + new_models[:15] + new_models[15:]
        # completely replace one model
        models[-1] = CNNPolicy((3, 72, 80), envs.action_space.n)

         # Checkpointing every 10 episodes
        if (episode + 1) % 20 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_ep_{episode+1}.pt")
            torch.save(top_models[0].state_dict(), checkpoint_path)
            print(f"Checkpointed best model at episode {episode+1} with reward: {bests[0]}")