import uuid
from pathlib import Path
import ray
from ray.rllib.algorithms import ppo
from red_gym_env_ray import RedGymEnv

ep_length = 2048 * 2 # 8
sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

env_config = {
            'headless': True, 'save_final_state': True, 'early_stop': False,
            'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
            'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
            'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
        }

ray.init()
algo = ppo.PPO(env=RedGymEnv, config={
    "framework": "torch",
    "env_config": env_config,  # config to pass to env class
})

while True:
    print(algo.train())