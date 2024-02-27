from pathlib import Path
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from baselines_utils import load_or_create_model

if __name__ == '__main__':

    ep_length = 2048 * 8
    sess_path = Path(f'session_{datetime.now().strftime("%Y%m%d_%H%M")}')
    #pretrained_session = "session_4da05e87_main_good/poke_439746560_steps"
    model_to_load_path = 'session_e41c9eff/poke_38207488_steps' #'session_e41c9eff/poke_250871808_steps'
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length,
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
                'use_screen_explore': True, 'extra_buttons': False
            }

    num_cpu = 44 #64 #46  # Also sets the number of episodes per training iteration
    model = load_or_create_model(model_to_load_path, env_config, ep_length, num_cpu)

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')
    learn_steps = 40

    for i in range(learn_steps):
        model.learn(total_timesteps=ep_length * num_cpu * 1000, callback=checkpoint_callback)
