from os.path import exists
import uuid
from red_gym_env import RedGymEnv
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker

sess_name = str(uuid.uuid4())[:8]

env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False, 
                'action_freq': 24, 'init_state': '../init.state', 'max_steps': 2048*20,
                'print_rewards': True, 'save_video': True, 'session_name': sess_name,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 7_000_000.0
            }

env = RedGymEnv(config=env_config)

env_checker.check_env(env)

learn_steps = 40
file_name = 'best_12-7/poke_12_b'
inference_only = True 

if exists(file_name + '.zip'):
    print('loading checkpoint')
    custom_objects = None
    if inference_only:
        custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
                "n_steps": 20*2048
            }
    model = PPO.load(file_name, env=env, custom_objects=custom_objects)
else:
    model = PPO('CnnPolicy', env, verbose=1, n_steps=2048*3*12, batch_size=128, n_epochs=3, gamma=0.995)

for i in range(learn_steps):
    model.learn(total_timesteps=2048*1*128)
    model.save(file_name+str(i))

