from ._red_gym_env import RedGymEnv
from stable_baselines3.common.utils import set_random_seed


def make_env(env_conf, rank=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)

        if rank is not None:
            env.reset(seed=rank)

        return env
    set_random_seed(0)
    return _init
