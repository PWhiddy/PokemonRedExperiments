from pathlib import Path
import pandas as pd
import numpy as np
from red_gym_env import RedGymEnv

def run_recorded_actions_on_emulator_and_save_video(sess_id, instance_id, run_index):
    sess_path = Path(f'session_{sess_id}')
    tdf = pd.read_csv(f"session_{sess_id}/agent_stats_{instance_id}.csv.gz", compression='gzip')
    tdf = tdf[tdf['map'] != 'map'] # remove unused 
    action_arrays = np.array_split(tdf, np.array((tdf["step"].astype(int) == 0).sum()))
    action_list = [int(x) for x in list(action_arrays[run_index]["last_action"])]
    max_steps = len(action_list) - 1

    env_config = {
            'headless': True, 'save_final_state': True, 'early_stop': False,
            'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': max_steps, #ep_length, 
            'print_rewards': False, 'save_video': True, 'fast_video': False, 'session_path': sess_path,
            'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'instance_id': f'{instance_id}_recorded'
    }
    env = RedGymEnv(env_config)
    env.reset_count = run_index

    obs = env.reset()
    for action in action_list:
        obs, rewards, term, trunc, info = env.step(action)
        env.render()
