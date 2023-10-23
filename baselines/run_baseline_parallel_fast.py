from os.path import exists
import os
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
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
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

#Allows user to select checkpoint to use
def choose_checkpoint():
    session_list = []
    base_dir = Path(__file__).resolve().parent #__name__?
    for x in os.listdir(base_dir):
        if 'session' in x:
            session_list.append({"session": x, "modified": os.path.getmtime(f"{base_dir}/{x}")})
    ordered_session_list = sorted(session_list, key=lambda x: x["modified"], reverse=True)
    ordered_session_list.append({"session": "None", "modified": "0"})

    print("Sessions (1 being the most recent):")
    for index, each in enumerate(ordered_session_list):
        print(f'{index + 1}: {each["session"]}')

    session_selection = int(input("Pick a session number to use (default is none): ") or 0)

    session = ordered_session_list[session_selection - 1]["session"]

    step_list = []
    for x in os.listdir(f"{base_dir}/{session}"):
        if "_steps.zip" in x:
            step_list.append(x)
    step_list = sorted(step_list, key=lambda x:x[5:x.index("_steps")], reverse=True)
    if step_list:
        step = step_list[0][0:-4]
        path_name = f"{session}/{step}"
    else:
        print("\n\n\nNo checkpoint found, starting training from scratch.\n\n\n")
        path_name = ""
    return(path_name)


if __name__ == '__main__':

    # put a checkpoint here you want to start from
    file_name = choose_checkpoint() 

    ep_length = 2048 * 10
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
                'explore_weight': 3 # 2.5
            }
    
    print(env_config)
    
    num_cpu = 16  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 40
    
    
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=checkpoint_callback)
