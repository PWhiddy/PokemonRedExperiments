"""
train_full_game_curriculum.py -- staged / curriculum PPO training driver.

Builds ``CurriculumRedGymEnv`` (structured obs + milestone rewards + anti-loop),
vectorizes it, and trains a Stable-Baselines3 PPO policy. Supports two execution
modes and two training shapes.

Execution modes
---------------
* ``--mode headless`` (default): no emulator window, ``SubprocVecEnv``, many parallel
  envs at unlimited speed -- the fast path for real training.
* ``--mode visual``: a single SDL2 window via ``DummyVecEnv`` so you can watch the
  agent. ``--speed`` throttles the emulator (0 = unlimited).

Training shapes
---------------
* Default (no ``--staged``): one shared PPO policy trained on the *whole* curriculum
  reward at once. Every milestone still grants its one-time bonus, so this is already
  curriculum-by-reward-shaping and a good baseline.
* ``--staged``: walk an ordered list of milestones. Each stage warm-starts episodes
  from the previous milestone's saved success states, trains the shared policy until a
  per-stage timestep budget (or success-rate threshold) is hit, saves the best model
  per milestone (``<session>/poke_<key>.zip``), and advances. Successful emulator
  snapshots are collected automatically by the env into ``state_store_root``.

Examples
--------
    # fast headless training, 16 parallel envs, unlimited speed (whole-game reward)
    python v2/train_full_game_curriculum.py --mode headless --num-envs 16 --speed 0

    # watch a single agent
    python v2/train_full_game_curriculum.py --mode visual --speed 3

    # explicit staged curriculum across all milestones
    python v2/train_full_game_curriculum.py --staged --num-envs 16

    # train one specific milestone, warm-starting from earlier success states
    python v2/train_full_game_curriculum.py --target-milestone beat_brock --num-envs 8

ROM note: this script never includes or downloads a ROM. Place your legally obtained
``PokemonRed.gb`` at the repository root (one level above ``v2/``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Run with cwd = v2/ so the base env's relative opens (events.json, ../init.state,
# ../PokemonRed.gb) resolve regardless of where the script was launched from.
_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from curriculum.config import CurriculumConfig
from curriculum.milestones import build_default_milestones
from curriculum.tb_callback import CurriculumTensorboardCallback
from curriculum.milestone_display import MilestoneDisplayCallback
from curriculum_env import CurriculumRedGymEnv


# --------------------------------------------------------------------------- #
# Resume helpers                                                               #
# --------------------------------------------------------------------------- #
_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def find_latest_checkpoint(sess_path: Path, label: str) -> str:
    """Return the newest resumable checkpoint path (without ``.zip``) for ``label``.

    Looks for ``CheckpointCallback`` snapshots named ``poke_<label>_<steps>_steps.zip``
    and picks the one with the highest step count; falls back to ``poke_<label>_best``.
    Returns "" when nothing is found.
    """
    sess_path = Path(sess_path)
    best_steps, best_path = -1, ""
    for p in sess_path.glob(f"poke_{label}_*_steps.zip"):
        m = _STEPS_RE.search(p.name)
        if m and int(m.group(1)) > best_steps:
            best_steps, best_path = int(m.group(1)), str(p)[:-4]  # strip .zip
    if best_path:
        return best_path
    final = sess_path / f"poke_{label}_best.zip"
    return str(final)[:-4] if final.exists() else ""


def _stage_progress_path(sess_path: Path) -> Path:
    return Path(sess_path) / "stage_progress.json"


def load_completed_stages(sess_path: Path) -> set:
    p = _stage_progress_path(sess_path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).get("completed_stages", []))
    except Exception:
        return set()


def mark_stage_complete(sess_path: Path, label: str, model_path: str) -> None:
    p = _stage_progress_path(sess_path)
    data = {"completed_stages": [], "last_model": ""}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            pass
    done = set(data.get("completed_stages", []))
    done.add(label)
    data["completed_stages"] = sorted(done)
    data["last_model"] = model_path
    p.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Env factory                                                                  #
# --------------------------------------------------------------------------- #
def make_env(rank: int, env_config: dict, seed: int = 0):
    def _init():
        env = CurriculumRedGymEnv(env_config)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(env_config: dict, num_envs: int, mode: str):
    if mode == "visual":
        # single windowed env to actually watch the agent
        return DummyVecEnv([make_env(0, env_config)])
    if num_envs <= 1:
        return DummyVecEnv([make_env(0, env_config)])
    return SubprocVecEnv([make_env(i, env_config) for i in range(num_envs)])


# --------------------------------------------------------------------------- #
# Callback: log milestone progress + optional stage early-stop                 #
# --------------------------------------------------------------------------- #
class MilestoneCallback(BaseCallback):
    """Logs milestone progress and (optionally) signals stage completion.

    Reads the ``info`` dicts emitted by ``CurriculumRedGymEnv.step``. When training a
    specific ``target_milestone``, tracks the rolling fraction of finished episodes
    that reached it and stops the stage once it clears ``success_threshold``.
    """

    def __init__(self, target_milestone=None, success_threshold: float = 0.8,
                 window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.target = target_milestone
        self.success_threshold = success_threshold
        self.window = window
        self.recent_success = []
        self.stage_done = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            self.logger.record("curriculum/milestones_completed",
                               info.get("milestones_completed", 0))
            self.logger.record("curriculum/milestone_progress",
                               info.get("milestone_progress", 0.0))
            if self.target is not None:
                reached = 1.0 if info.get("reached_target", False) else 0.0
                self.recent_success.append(reached)
                self.recent_success = self.recent_success[-self.window:]
                rate = sum(self.recent_success) / max(len(self.recent_success), 1)
                self.logger.record("curriculum/target_success_rate", rate)
                if (len(self.recent_success) >= self.window
                        and rate >= self.success_threshold):
                    self.stage_done = True
        # returning False stops training -> ends this stage early
        return not self.stage_done


# --------------------------------------------------------------------------- #
# Model helpers                                                                #
# --------------------------------------------------------------------------- #
def linear_schedule(initial: float, final_frac: float = 0.1):
    """SB3 schedule: decay ``initial`` linearly to ``initial*final_frac`` over training.

    ``progress_remaining`` goes 1 -> 0, so this returns initial at the start and
    ``initial*final_frac`` at the end. A decaying LR (and clip range) stabilizes the
    long PPO runs needed to push past mid-game walls.
    """
    def f(progress_remaining: float) -> float:
        return initial * (final_frac + (1.0 - final_frac) * progress_remaining)
    return f


def make_or_load_model(vec_env, sess_path: Path, n_steps: int, resume: str = "",
                       seed: int = 0, lr: float = 3e-4, ent_coef: float = 0.01):
    if resume and Path(resume + ".zip").exists():
        print(f"[train] loading model from {resume}.zip")
        model = PPO.load(resume, env=vec_env)
        model.n_steps = n_steps
        model.n_envs = vec_env.num_envs
        model.rollout_buffer.buffer_size = n_steps
        model.rollout_buffer.n_envs = vec_env.num_envs
        model.rollout_buffer.reset()
        return model
    return PPO(
        "MultiInputPolicy", vec_env, verbose=1,
        n_steps=n_steps, batch_size=512, n_epochs=3,
        gamma=0.998, gae_lambda=0.95, ent_coef=ent_coef,
        learning_rate=linear_schedule(lr),
        clip_range=linear_schedule(0.2),
        vf_coef=0.5, max_grad_norm=0.5,
        tensorboard_log=str(sess_path), seed=seed,
    )


def train_one_stage(cfg: CurriculumConfig, args, target_milestone, resume_path,
                    timesteps, sess_path: Path):
    """Build env, train (optionally to a success threshold), return best model path."""
    cfg.target_milestone = target_milestone
    if target_milestone is not None:
        cfg.end_on_target = True  # focus episodes on hitting the target

    env_config = cfg.to_env_config()
    label = target_milestone or "fullgame"

    # Resolve resume target. Priority:
    #   1. this stage's own newest checkpoint (resumes an interrupted stage exactly),
    #   2. an explicit resume path passed in (e.g. previous stage's best model),
    #   3. nothing -> fresh start.
    own_ckpt = find_latest_checkpoint(sess_path, label)
    if own_ckpt:
        print(f"[resume] resuming stage '{label}' from its own checkpoint {own_ckpt}.zip")
        resume_path = own_ckpt
    elif resume_path == "auto":
        print(f"[resume] no existing checkpoint for stage '{label}'; starting fresh")
        resume_path = ""
    elif resume_path:
        print(f"[resume] warm-starting stage '{label}' from {resume_path}.zip")

    print(f"\n=== Stage: {label} | timesteps={timesteps} | envs={args.num_envs} "
          f"| mode={args.mode} ===")
    print(env_config)

    vec_env = build_vec_env(env_config, args.num_envs, args.mode)
    n_steps = args.n_steps  # per-env steps per update (SB3 multiplies by n_envs)

    model = make_or_load_model(vec_env, sess_path, n_steps, resume=resume_path,
                               seed=args.seed, lr=args.lr, ent_coef=args.ent_coef)
    print(model.policy)

    # Checkpoint often enough that an interruption loses little progress. save_freq is
    # per-env steps, so multiply target frequency back out by env count.
    save_every = max(args.save_freq // max(vec_env.num_envs, 1), 1)
    ckpt = CheckpointCallback(save_freq=save_every,
                              save_path=str(sess_path),
                              name_prefix=f"poke_{label}")
    milestone_cb = MilestoneCallback(target_milestone=target_milestone,
                                     success_threshold=args.success_threshold,
                                     window=args.success_window)
    display_cb = MilestoneDisplayCallback(sess_path)
    callbacks = CallbackList([ckpt, CurriculumTensorboardCallback(sess_path),
                              milestone_cb, display_cb])

    model.learn(total_timesteps=timesteps, callback=callbacks,
                tb_log_name=f"ppo_{label}", reset_num_timesteps=False)

    best_path = sess_path / f"poke_{label}_best"
    model.save(str(best_path))
    print(f"[train] saved stage model -> {best_path}.zip")
    vec_env.close()
    mark_stage_complete(sess_path, label, str(best_path))
    return str(best_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Curriculum PPO trainer for Pokemon Red")
    p.add_argument("--config", default="configs/curriculum.yaml",
                   help="path to YAML/JSON config (default: configs/curriculum.yaml)")
    p.add_argument("--mode", choices=["headless", "visual"], default="headless")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--speed", type=int, default=None,
                   help="emulation speed for visual mode (0 = unlimited)")
    p.add_argument("--timesteps", type=int, default=20_000_000,
                   help="total timesteps for the (single) run")
    p.add_argument("--steps-per-stage", type=int, default=2_000_000,
                   help="timesteps budget per milestone in --staged mode")
    p.add_argument("--n-steps", type=int, default=2048,
                   help="PPO per-env rollout length")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="initial learning rate (linearly decayed over the run)")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO entropy coefficient (higher = more exploration)")
    p.add_argument("--staged", action="store_true",
                   help="walk all milestones one stage at a time (shared policy)")
    p.add_argument("--target-milestone", default=None,
                   help="train a single milestone (warm-start from earlier states)")
    p.add_argument("--milestones", default=None,
                   help="comma-separated subset of milestone keys for --staged")
    p.add_argument("--resume", default="auto",
                   help="'auto' (default) finds the newest checkpoint in the session "
                        "and continues; '' starts fresh; or pass a path (without .zip)")
    p.add_argument("--save-freq", type=int, default=200_000,
                   help="save a resumable checkpoint every N total timesteps")
    p.add_argument("--session", default=None, help="override session_path")
    p.add_argument("--success-threshold", type=float, default=0.8)
    p.add_argument("--success-window", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # --- adaptive / planner toggles (override the YAML) ---
    p.add_argument("--planner", choices=["none", "rule", "ollama"], default=None,
                   help="enable the high-level planner + subgoal shaping")
    p.add_argument("--planner-model", default=None,
                   help="Ollama model for the planner (e.g. nemotron-mini)")
    p.add_argument("--intrinsic", choices=["none", "count", "rnd"], default=None,
                   help="intrinsic-motivation kind (overrides reward.intrinsic_kind)")
    p.add_argument("--no-adaptive", action="store_true",
                   help="disable the homeostatic reward controller")
    p.add_argument("--advice", action="store_true",
                   help="reward the policy for following the planner's advised skill "
                        "(and execute field-move macros like Cut)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = (CurriculumConfig.from_file(args.config)
           if Path(args.config).exists() else CurriculumConfig())

    # apply CLI overrides
    cfg.headless = (args.mode == "headless")
    if args.speed is not None:
        cfg.emulation_speed = args.speed
    if args.session:
        cfg.session_path = args.session
    # adaptive / planner overrides
    if args.planner is not None:
        cfg.use_planner = (args.planner != "none")
        cfg.planner_kind = args.planner
    if args.planner_model is not None:
        cfg.planner_model = args.planner_model
    if args.intrinsic is not None:
        cfg.reward.intrinsic_kind = args.intrinsic
    if args.no_adaptive:
        cfg.reward.adaptive_enabled = False
    if args.advice:
        cfg.use_advice = True
        if not cfg.use_planner:          # advice needs a planner to advise skills
            cfg.use_planner = True
            if cfg.planner_kind == "none":
                cfg.planner_kind = "rule"
    sess_path = Path(cfg.session_path)
    sess_path.mkdir(parents=True, exist_ok=True)

    if not Path(cfg.gb_path).exists():
        print(f"!! ROM not found at '{cfg.gb_path}'. Place your legally obtained "
              f"PokemonRed.gb there (it is never bundled). Continuing will fail.")

    if args.staged:
        # ordered milestone keys
        all_keys = [m.key for m in build_default_milestones()]
        if args.milestones:
            wanted = [k.strip() for k in args.milestones.split(",") if k.strip()]
            keys = [k for k in all_keys if k in wanted]
        else:
            keys = all_keys

        # Resume-aware staging: skip stages already finished in a prior run and
        # warm-start the next stage from the last completed stage's model.
        completed = load_completed_stages(sess_path)
        if completed:
            print(f"[resume] staged run: {len(completed)} stage(s) already complete: "
                  f"{sorted(completed)}")
        resume = args.resume
        last_done_model = ""
        if completed:
            prev = load_completed_stages  # noqa: F841 (kept for clarity)
            data_path = _stage_progress_path(sess_path)
            try:
                last_done_model = json.loads(data_path.read_text()).get("last_model", "")
            except Exception:
                last_done_model = ""
        for key in keys:
            if key in completed:
                print(f"[resume] skipping completed stage '{key}'")
                if last_done_model:
                    resume = last_done_model
                continue
            resume = train_one_stage(cfg, args, key, resume,
                                     args.steps_per_stage, sess_path)
        print("\n[train] staged curriculum complete.")
    else:
        # single run: either a focused target or the whole-game shared policy
        train_one_stage(cfg, args, args.target_milestone, args.resume,
                        args.timesteps, sess_path)


if __name__ == "__main__":
    main()
