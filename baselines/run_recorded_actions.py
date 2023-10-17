from pathlib import Path
import pandas as pd
import numpy as np
from red_gym_env import RedGymEnv
from util import get_args, change_env
from baselines.constants import GB_FILENAME

def run_recorded_actions_on_emulator_and_save_video(sess_id, instance_id, run_index):
    sess_path = Path(f'session_{sess_id}')
    tdf = pd.read_csv(f"session_{sess_id}/agent_stats_{instance_id}.csv.gz", compression='gzip')
    tdf = tdf[tdf['map'] != 'map'] # remove unused 
    action_arrays = np.array_split(tdf, np.array((tdf["step"].astype(int) == 0).sum()))
    action_list = [int(x) for x in list(action_arrays[run_index]["last_action"])]
    max_steps = len(action_list) - 1

    args = get_args()

    env_config = change_env(args)
    env = RedGymEnv(env_config)
    env.reset_count = run_index

    obs = env.reset()
    for action in action_list:
        obs, rewards, term, trunc, info = env.step(action)
        env.render()
