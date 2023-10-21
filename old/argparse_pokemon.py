# argparse_pokemon.py - Has two functions: one for getting arguments and another for changing the environment based on those arguments
# This is meant to help config your file without needing to go into the code
# Ryan Peruski, 10-13-2023

import argparse
from pathlib import Path
import uuid

def get_args(usage_string=None, ep_length=None, sess_path=None, headless=True):
    if sess_path == None:
        sess_path = f'session_{str(uuid.uuid4())[:8]}'
    description='Argument parser for env_config',
    usage=f'python {usage_string} [--headless HEADLESS] [--save_final_state SAVE_FINAL_STATE] ...' #usage different depending on the file
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('--headless', type=bool, default=headless, help='Whether to run the environment in headless mode')
    parser.add_argument('--save_final_state', type=bool, default=True, help='Whether to save the final state of the environment')
    parser.add_argument('--early_stop', type=bool, default=False, help='Whether to stop the environment early')
    parser.add_argument('--action_freq', type=int, default=24, help='Frequency of actions')
    parser.add_argument('--init_state', type=str, default='../has_pokedex_nballs.state', help='Initial state of the environment')
    parser.add_argument('--max_steps', type=int, default=ep_length, help='Maximum number of steps in the environment')
    parser.add_argument('--print_rewards', type=bool, default=True, help='Whether to print rewards')
    parser.add_argument('--save_video', type=bool, default=True, help='Whether to save a video of the environment')
    parser.add_argument('--fast_video', type=bool, default=False, help='Whether to save a fast video of the environment')
    parser.add_argument('--session_path', type=str, default=sess_path, help='Path to the session')
    parser.add_argument('--gb_path', type=str, default='../PokemonRed.gb', help='Path to the gameboy ROM')
    parser.add_argument('--debug', type=bool, default=False, help='Whether to run the environment in debug mode')
    parser.add_argument('--sim_frame_dist', type=float, default=2_000_000.0, help='Simulation frame distance')
    args, unknown_args= parser.parse_known_args() # Parses only the known args to fix an issue with argv[1] being used as a save path
    return args

def change_env(env_config, args):
    #Changes the environment based on the arguments given a env_config dictionary and args
    env_config = {
        'headless': args.headless,
        'save_final_state': args.save_final_state,
        'early_stop': args.early_stop,
        'action_freq': args.action_freq,
        'init_state': args.init_state,
        'max_steps': args.max_steps,
        'print_rewards': args.print_rewards,
        'save_video': args.save_video,
        'fast_video': args.fast_video,
        'session_path': Path(args.session_path), 
        'gb_path': args.gb_path,
        'debug': args.debug,
        'sim_frame_dist': args.sim_frame_dist
    }
    return env_config
