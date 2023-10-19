from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from argparse_pokemon import *
import os
import glob
import re

# TensorBoard log directory
log_dir = "./tensorboard/"
# Create log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
        return env
    set_random_seed(seed)
    return _init

def find_latest_session_and_poke():
    all_folders = os.listdir()
    session_folders = [folder for folder in all_folders if re.match(r'session_[0-9a-fA-F]{8}', folder)]

    most_recent_time = 0
    most_recent_session = None
    most_recent_poke_file = None

    for session_folder in session_folders:
        poke_files = glob.glob(f"{session_folder}/poke_*_steps.zip")
        for poke_file in poke_files:
            mod_time = os.path.getmtime(poke_file)
            if mod_time > most_recent_time:
                most_recent_time = mod_time
                most_recent_session = session_folder
                most_recent_poke_file = poke_file[:-4]  # Remove '.zip' from the filename

    return most_recent_session, most_recent_poke_file

if __name__ == '__main__':


    ep_length = 2048 * 8
    #ep_length = 2048 
    sess_path = f'session_{str(uuid.uuid4())[:8]}'
    args = get_args('run_baseline_parallel.py', ep_length=ep_length, sess_path=sess_path)

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
            }
    
    env_config = change_env(env_config, args)
    
    num_cpu = os.cpu_count()

    #num_cpu = 44 #64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 40
    #file_name = 'session_e41c9eff/poke_38207488_steps' #'session_e41c9eff/poke_250871808_steps'
    session_folder, latest_poke_file = find_latest_session_and_poke()
    print('\n' + latest_poke_file)
    #'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    #if exists(file_name + '.zip'):
    if latest_poke_file:
        print('\nloading checkpoint')
        #model = PPO.load(latest_poke_file, env=env)
        model = PPO.load(latest_poke_file, env=env, tensorboard_log=log_dir)

        #print('\nloading checkpoint')
        #model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print('No valid session or poke file found.')
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999, tensorboard_log=log_dir)

    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=[checkpoint_callback])
