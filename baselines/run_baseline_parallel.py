from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback
from baselines_utils import load_or_create_model, get_formatted_timestamp

def get_config():
    return {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': total_timesteps,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False
    }


if __name__ == '__main__':

    total_timesteps = 2048 * 8
    num_cpu = 1  # Also sets the number of episodes per training iteration
    sess_path = Path(f'session_{get_formatted_timestamp()}')
    #pretrained_session = "session_4da05e87_main_good/poke_439746560_steps"
    model_to_load_path = 'session_20240227_1328/poke_16384_steps.zip'
    env_config = get_config()

    model = load_or_create_model(model_to_load_path, env_config, total_timesteps, num_cpu)

    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps, save_path=sess_path, name_prefix='poke')
    learn_steps = 2

    for i in range(learn_steps):
        model.learn(total_timesteps=(total_timesteps) * num_cpu * 1000, callback=checkpoint_callback, progress_bar=False)