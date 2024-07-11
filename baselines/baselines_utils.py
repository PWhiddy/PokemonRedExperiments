from os.path import exists
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from stream_agent_wrapper import StreamWrapper
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
        model = PPO('CnnPolicy', env, verbose=1, n_steps=total_timesteps, batch_size=512, n_epochs=1, gamma=0.999, tensorboard_log=env_config['session_path'])

    return model


def make_env(rank, env_config, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_config)
        env.reset(seed=(seed + rank))
        if env_config['stream'] is True:
            env = StreamWrapper(
                env,
                stream_metadata = { # All of this is part is optional
                    "user": "MATHIEU", # choose your own username
                    "env_id": env_config['instance_id'], # environment identifier
                    "color": "#d900ff", # choose your color :)
                    "extra": "", # any extra text you put here will be displayed
                }
            )
        return env
    set_random_seed(seed)
    return _init