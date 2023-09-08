
from baselines.red_gym_env import RedGymEnv
from pyboy.utils import WindowEvent
import time
import uuid
from pathlib import Path
import random

sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

run_steps = 1024*20

env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False, 
                'action_freq': 24, 'init_state': './has_pokedex_nballs.state', 'max_steps': run_steps,
                'print_rewards': True, 'save_video': True, 'session_path': sess_path,
                'gb_path': './PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
            }

env = RedGymEnv(config=env_config)

step = 0
while True:
    #time.sleep(0.1)
    #if step < 3:
    #obs_memory, reward, done, info = env.step(random.choice([0,1,2,3])) 
    #else:
    obs_memory, reward, done, info = env.step(7) # 7 pass
    step += 1
    '''
    if step % 50 == 0:
        with open('has_pokedex_nballs.state', 'wb') as f:
            env.pyboy.save_state(f)
    '''