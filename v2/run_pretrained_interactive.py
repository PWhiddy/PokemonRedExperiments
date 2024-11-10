import os
from os.path import exists
from pathlib import Path
import uuid
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

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

def get_most_recent_checkpoint():
    max_steps = -1
    most_recent_zip = None
    runs_folder = "runs"
    for file_name in os.listdir(runs_folder):
        if file_name.endswith("_steps.zip") and file_name.startswith("poke_"):
            try:
                steps = int(file_name.split("_")[1])
                if steps > max_steps:
                    max_steps = steps
                    most_recent_zip = file_name
            except ValueError:
                continue 

    if most_recent_zip:
        return os.path.join(runs_folder, most_recent_zip)
    else:
        return None

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
    most_recent_checkpoint = get_most_recent_checkpoint()
    if most_recent_checkpoint is not None:
        file_name = most_recent_checkpoint
    
    # could optionally manually specify a checkpoint here
    # file_name = "runs/poke_124800_steps.zip"
    
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
        
    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 7 # pass action
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()


