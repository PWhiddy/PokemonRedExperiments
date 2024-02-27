from os.path import exists
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from red_gym_env import RedGymEnv


def load_or_create_model(model_to_load_path, env_config, total_timesteps, num_cpu):

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    if exists(model_to_load_path + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(model_to_load_path, env=env)
        model.n_steps = total_timesteps
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = total_timesteps
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=total_timesteps, batch_size=512, n_epochs=1, gamma=0.999)

    return model


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

from datetime import datetime


def get_formatted_timestamp():
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
    return formatted_timestamp