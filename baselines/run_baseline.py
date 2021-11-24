from os.path import exists
from red_gym_env import RedGymEnv
import gym
from stable_baselines3 import A2C, PPO

env = RedGymEnv(headless=False,
        action_freq=5, init_state='../init.state', max_steps=100, 
        gb_path='../PokemonRed.gb', debug=False)

# env = gym.make('CartPole-v1')

file_name = 'poke_1'
if exists(file_name+'.zip'):
    print('loading checkpoint')
    model = PPO.load(file_name, env=env)
else:
    model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=100000)

model.save(file_name)

