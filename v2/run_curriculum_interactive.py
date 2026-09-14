"""
run_curriculum_interactive.py -- watch the curriculum env (random or trained policy).

Useful for sanity-checking the structured observation, milestone detection, and
reward components, and for visually evaluating a trained model.

Examples
--------
    # random policy in a window
    python v2/run_curriculum_interactive.py --mode visual --speed 3

    # evaluate a trained model
    python v2/run_curriculum_interactive.py --mode visual --model runs_curriculum/poke_fullgame_best

    # headless quick smoke (no window), prints reward components + milestones
    python v2/run_curriculum_interactive.py --mode headless --steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from curriculum.config import CurriculumConfig
from curriculum_env import CurriculumRedGymEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/curriculum.yaml")
    p.add_argument("--mode", choices=["headless", "visual"], default="visual")
    p.add_argument("--speed", type=int, default=3)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--model", default="", help="trained model path (without .zip)")
    p.add_argument("--target-milestone", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = (CurriculumConfig.from_file(args.config)
           if Path(args.config).exists() else CurriculumConfig())
    cfg.headless = (args.mode == "headless")
    cfg.emulation_speed = args.speed
    cfg.target_milestone = args.target_milestone

    env = CurriculumRedGymEnv(cfg.to_env_config())
    obs, _ = env.reset()
    print("observation keys:", {k: getattr(v, "shape", None) for k, v in obs.items()})

    model = None
    if args.model and Path(args.model + ".zip").exists():
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"loaded model {args.model}.zip")

    total = 0.0
    for i in range(args.steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if reward != 0.0 or i % 200 == 0:
            comps = {k: round(v, 3) for k, v in info["reward_components"].items()}
            print(f"step {i:5d} r={reward:7.3f} total={total:9.2f} "
                  f"milestones={info['milestones_completed']} comps={comps}")
        if terminated or truncated:
            print(f"episode end @ step {i}: milestones={info['milestones_completed']}")
            obs, _ = env.reset()

    print(f"done. total reward={total:.2f}")


if __name__ == "__main__":
    main()
