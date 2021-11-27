from os.path import exists
from red_gym_env import RedGymEnv
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker

env_config = {
                'headless': False, 'save_final_state': True,
                'action_freq': 5, 'init_state': '../init.state', 'max_steps': 2*768, 'print_rewards': True,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2500000.0
            }

env = RedGymEnv(env_config)

env_checker.check_env(env)

# env = gym.make('CartPole-v1')
learn_steps = 40
file_name = 'poke_1'
if exists(file_name+'.zip'):
    print('loading checkpoint')
    model = PPO.load(file_name, env=env)
else:
    model = PPO('CnnPolicy', env, verbose=1, n_steps=2048*6, batch_size=128, n_epochs=20, gamma=0.99)

for i in range(learn_steps):
    model.learn(total_timesteps=480000)
    model.save(file_name+str(i))

