from os.path import exists
from red_gym_env import RedGymEnv
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

env_config = {
                'headless': True, 'save_final_state': True,
                'action_freq': 5, 'init_state': '../init.state', 'max_steps': 4*2048, 'print_rewards': True,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 10_000_000.0
            }


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_config)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    #env_checker.check_env(env)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    learn_steps = 40
    file_name = 'poke_'
    if exists(file_name+'.zip'):
        print('loading checkpoint')
        model = PPO.load(file_name, env=env)
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=2048*2*8, batch_size=128, n_epochs=2, gamma=0.99)

    for i in range(learn_steps):
        model.learn(total_timesteps=480000)
        model.save(file_name+str(i))