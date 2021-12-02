
from baselines.red_gym_env import RedGymEnv
from pyboy.utils import WindowEvent
import time
import random

env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': './init.state', 'max_steps': 16*2048, 'print_rewards': True,
                'gb_path': './PokemonRed.gb', 'debug': False, 'sim_frame_dist': 10_000_000.0
            }

env = RedGymEnv(config=env_config)

step = 0
while True:
    #time.sleep(0.1)
    #if step < 3:
    obs_memory, reward, done, info = env.step(random.choice([0,1,2,3])) 
    #else:
    #    obs_memory, reward, done, info = env.step(7) # 7 pass
    step += 1