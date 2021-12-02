from os.path import exists
from red_gym_env import RedGymEnv
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker

env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../init.state', 'max_steps': 2*2048, 'print_rewards': True,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 7_000_000.0
            }

env = RedGymEnv(config=env_config)

env_checker.check_env(env)

learn_steps = 40
file_name = 'poke_'
if exists(file_name+'.zip'):
    print('loading checkpoint')
    model = PPO.load(file_name, env=env)
else:
    model = PPO('CnnPolicy', env, verbose=1, n_steps=2048*2*12, batch_size=128, n_epochs=3, gamma=0.995)

for i in range(learn_steps):
    model.learn(total_timesteps=2048*1*128)
    model.save(file_name+str(i))

