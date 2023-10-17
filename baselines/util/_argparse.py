# _argparse.py - Has two functions: one for getting arguments and another for changing the environment based on those arguments
# This is meant to help config your file without needing to go into the code
# Ryan Peruski, 10-13-2023

from argparse import ArgumentParser, BooleanOptionalAction, ArgumentTypeError
from pathlib import Path
from baselines.constants import GB_FILENAME, DEFAULT_CPU_COUNT, DEFAULT_EP_LENGTH
from baselines.util import get_new_session


def get_args():
    description = 'Argument parser for env_config'
    parser = ArgumentParser(description=description)
    parser.add_argument('--headless', action=BooleanOptionalAction, default=True, help='Whether to run the environment in headless mode')
    parser.add_argument('--save_final_state', type=bool, default=True, help='Whether to save the final state of the environment')
    parser.add_argument('--early_stop', type=bool, default=False, help='Whether to stop the environment early')
    parser.add_argument('--action_freq', type=int, default=24, help='Frequency of actions')
    parser.add_argument('--init_state', type=str, default='../has_pokedex_nballs.state', help='Initial state of the environment')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_EP_LENGTH, help='Maximum number of steps in the environment')
    parser.add_argument('--print_rewards', type=bool, default=True, help='Whether to print rewards')
    parser.add_argument('--save_video', type=bool, default=True, help='Whether to save a video of the environment')
    parser.add_argument('--fast_video', type=bool, default=False, help='Whether to save a fast video of the environment')
    parser.add_argument('--session_path', type=str, default=get_new_session(), help='Path at which to save session data')
    parser.add_argument('--gb_path', type=str, default=GB_FILENAME, help='Path to the gameboy ROM')
    parser.add_argument('--debug', type=bool, default=False, help='Whether to run the environment in debug mode')
    parser.add_argument('--sim_frame_dist', type=float, default=2_000_000.0, help='Simulation frame distance')
    parser.add_argument('--cpu_count', type=int, default=DEFAULT_CPU_COUNT, help='Number of CPUs to use')

    args, unknown_args = parser.parse_known_args()  # Parses only the known args to fix an issue with argv[1] being used as a save path

    if args.cpu_count <= 0:
        raise ArgumentTypeError("Number of CPUs must be 1 or more.")

    if args.max_steps <= 0:
        raise ArgumentTypeError("Max steps must be 1 or more.")

    return args


def change_env(args):
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
