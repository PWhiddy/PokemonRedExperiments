# Agent Research & Training Log

## Goal
Research the `PokemonRedExperiments` project and train an agent to play Pokemon Red using Reinforcement Learning.

## Project Overview
This project leverages **PyBoy** for GameBoy emulation and **Stable Baselines 3** for Reinforcement Learning. The environment is a custom OpenAI Gym wrapper around the Pokemon Red ROM.

### Key Components
- **Game ROM**: `PokemonRed.gb` (sha1: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)
- **Environment**: `v2/red_gym_env_v2.py` defines the state, actions, and reward function.
- **Training Script**: `v2/baseline_fast_v2.py` is the main entry point for training using PPO.
- **Pre-saved States**: `.state` files in the root directory allow starting the game from specific points (e.g., after getting the Pokedex).

## Research Notes

### Reward Function
(To be explored in `v2/red_gym_env_v2.py`)

### Exploration Strategy
V2 uses a coordinate-based exploration reward instead of the frame KNN used in V1. This is intended to be more efficient and help the agent reach Cerulean City.

## To-Do List
- [ ] Analyze `v2/red_gym_env_v2.py` to understand observation space and rewards.
- [ ] Create a Jupyter Notebook in `research/` for interactive environment exploration.
- [ ] Verify dependencies are installed.
- [ ] Run `v2/baseline_fast_v2.py` and monitor progress via Tensorboard.
