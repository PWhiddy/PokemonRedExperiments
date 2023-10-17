from os.path import exists
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from util import get_args, change_env


run_steps = 2048
runs_per_update = 6
updates_per_checkpoint = 4

if __name__ == '__main__':

    args = get_args()

    env_config = change_env(args)
    env = RedGymEnv(config=env_config)

    env_checker.check_env(env)

    learn_steps = 40
    file_name = 'poke_' #'best_12-7/poke_12_b'
    inference_only = True

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')

        if inference_only:
            custom_objects = {
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                    "n_steps": 10*20*2048
                }
        else:
            custom_objects = None
        model = PPO.load(file_name, env=env, custom_objects=custom_objects)
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=run_steps*runs_per_update, batch_size=128, n_epochs=3, gamma=0.98)

    for i in range(learn_steps):
        model.learn(total_timesteps=run_steps*runs_per_update*updates_per_checkpoint)
        model.save(args.session_path / Path(file_name+str(i)))

