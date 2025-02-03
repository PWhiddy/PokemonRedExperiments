import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import matplotlib.pyplot as plt

"""
Our dataset
"""
X = []

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
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None  # Return None if no zip files are found
    
    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    
    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # Convert seconds to hours
    
    return most_recent_zip, age_in_hours

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    #env_checker.check_env(env)
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs")
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
        print(f"using checkpoint: {file_name}, which is {time_since} hours old")
    
    # could optionally manually specify a checkpoint here
    #file_name = "runs/poke_41943040_steps.zip"
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    counter = 0
    x = np.zeros((3, 73, 80))
    
    while True:
        if counter > 2:
            counter = 0
            # Combine the last 3 frames
            combined = np.hstack((x[0], x[1], x[2]))
            X.append(combined)
            
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        game_pixels = env.render()
        # Combine game_pxiels with action
        action = np.full((1, 80), action)
        pixel_image = game_pixels.squeeze()
        action_image = action.squeeze()
        combined = np.vstack((pixel_image, action_image))
        x[counter] = combined
        
        counter += 1
        
        
        if truncated:
            break
    env.close()


