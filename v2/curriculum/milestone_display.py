"""
milestone_display.py -- live, human-readable progress display for curriculum training.

``MilestoneDisplayCallback`` reads the ``info`` dicts emitted by
``CurriculumRedGymEnv.step`` (specifically the ``new_milestones`` list added there)
and turns reaching a significant checkpoint into:

  * a one-time console **banner** the first time *any* env reaches that milestone in
    the run ("DEFEATED BROCK", "CLEARED ROCKET HIDEOUT", "BEAT THE CHAMPION", ...),
    with the tier (LEGENDARY / MAJOR / KEY / step) chosen from the milestone reward;
  * an append-only ``milestones_log.jsonl`` recording *every* reach (timestep, env id,
    badges, levels) so the full history survives restarts;
  * a ``progress.json`` snapshot of the furthest milestone reached, per-milestone reach
    counts, and best badges/levels -- a durable "where is the agent now" dashboard;
  * TensorBoard scalars (``milestones/<key>_first_step`` and ``progress/furthest_index``)
    so milestone timing shows up alongside the reward curves.

The callback is read-only w.r.t. the env and safe with any number of parallel envs.
It dedupes the *banner* (first reach only) but counts *all* reaches, so you see both
the headline event and how reliably the fleet is clearing it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from stable_baselines3.common.callbacks import BaseCallback

from curriculum.milestones import build_default_milestones


# Tier thresholds by a milestone's one-time reward (see milestones.py).
TIER_LEGENDARY = 300.0   # Elite Four, Champion
TIER_MAJOR = 120.0       # gym badges, major dungeons (Rocket Hideout, Silph, ...)
TIER_KEY = 70.0          # key items / Pokedex / SS Ticket


def _tier(reward: float) -> str:
    if reward >= TIER_LEGENDARY:
        return "LEGENDARY"
    if reward >= TIER_MAJOR:
        return "MAJOR"
    if reward >= TIER_KEY:
        return "KEY"
    return "STEP"


# ASCII-only markers: the Windows console defaults to cp1252 and would raise
# UnicodeEncodeError on emoji, which must never crash training.
_TIER_STYLE = {
    "LEGENDARY": ("*** ", "#"),
    "MAJOR":     (">>> ", "="),
    "KEY":       ("[+] ", "-"),
    "STEP":      (" -> ", None),
}


class MilestoneDisplayCallback(BaseCallback):
    """Announce significant checkpoints and persist a durable progress dashboard."""

    def __init__(self, session_path, milestones=None, summary_every: int = 50_000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.session_path = Path(session_path)
        self.session_path.mkdir(parents=True, exist_ok=True)
        ms = milestones if milestones is not None else build_default_milestones()
        self.order = {m.key: i for i, m in enumerate(ms)}
        self.total_milestones = len(ms)

        self.log_file = self.session_path / "milestones_log.jsonl"
        self.progress_file = self.session_path / "progress.json"

        self.summary_every = summary_every
        self._next_summary = summary_every

        # run-level state (also rehydrated from progress.json so banners stay
        # one-time across resumes)
        self.announced: Set[str] = set()
        self.reach_counts: Dict[str, int] = {}
        self.first_step: Dict[str, int] = {}
        self.furthest_index = -1
        self.furthest_name = "(none yet)"
        self.best_badges = 0
        self.best_levels = 0
        self._load_progress()

    # ------------------------------------------------------------------ #
    def _load_progress(self) -> None:
        if not self.progress_file.exists():
            return
        try:
            data = json.loads(self.progress_file.read_text())
        except Exception:
            return
        self.announced = set(data.get("announced", []))
        self.reach_counts = {k: int(v) for k, v in data.get("reach_counts", {}).items()}
        self.first_step = {k: int(v) for k, v in data.get("first_step", {}).items()}
        self.furthest_index = int(data.get("furthest_index", -1))
        self.furthest_name = data.get("furthest_name", "(none yet)")
        self.best_badges = int(data.get("best_badges", 0))
        self.best_levels = int(data.get("best_levels", 0))
        if self.announced:
            print(f"[milestones] resumed progress: furthest = "
                  f"'{self.furthest_name}' ({self.furthest_index + 1}/"
                  f"{self.total_milestones}), {len(self.announced)} milestones "
                  f"already reached this campaign.")

    def _save_progress(self) -> None:
        data = {
            "updated_timestep": int(self.num_timesteps),
            "furthest_index": self.furthest_index,
            "furthest_name": self.furthest_name,
            "furthest_fraction": (self.furthest_index + 1) / max(self.total_milestones, 1),
            "best_badges": self.best_badges,
            "best_levels": self.best_levels,
            "announced": sorted(self.announced),
            "first_step": self.first_step,
            "reach_counts": self.reach_counts,
        }
        tmp = self.progress_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.progress_file)  # atomic-ish; survives a kill mid-write

    # ------------------------------------------------------------------ #
    def _banner(self, m: dict, env_id: int) -> None:
        tier = _tier(float(m.get("reward", 0.0)))
        icon, rule = _TIER_STYLE[tier]
        name = m["name"]
        step = m.get("step", 0)
        levels = m.get("party_levels_sum", 0)
        badges = m.get("badges", 0)
        if rule is None:
            print(f"{icon} reached: {name}  "
                  f"[t={self.num_timesteps:,} env{env_id} badges={badges} lv={levels}]")
            return
        line = rule * 72
        print(f"\n{line}")
        print(f"{icon}  {tier} CHECKPOINT REACHED:  {name.upper()}")
        print(f"     first reach at training step {self.num_timesteps:,} "
              f"(env {env_id}, episode step {step})")
        print(f"     badges={badges}  party_levels_sum={levels}  "
              f"progress={self.order.get(m['key'], 0) + 1}/{self.total_milestones}")
        print(f"{line}\n")

    def _record(self, m: dict, env_id: int) -> None:
        key = m["key"]
        self.reach_counts[key] = self.reach_counts.get(key, 0) + 1
        idx = self.order.get(key, -1)
        if idx > self.furthest_index:
            self.furthest_index = idx
            self.furthest_name = m["name"]
        self.best_badges = max(self.best_badges, int(m.get("badges", 0)))
        self.best_levels = max(self.best_levels, int(m.get("party_levels_sum", 0)))

        first = key not in self.announced
        if first:
            self.announced.add(key)
            self.first_step[key] = int(self.num_timesteps)
            self._banner(m, env_id)
            # TB: when each milestone was first cleared + how far we've gotten
            self.logger.record(f"milestones/{key}_first_step", int(self.num_timesteps))
            self.logger.record("progress/furthest_index", self.furthest_index)
            self.logger.record("progress/furthest_fraction",
                               (self.furthest_index + 1) / max(self.total_milestones, 1))

        # append-only history of every reach (durable across restarts)
        rec = {
            "timestep": int(self.num_timesteps),
            "wall_time": time.time(),
            "env": env_id,
            "first_reach": first,
            **m,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dirty = False
        for env_id, info in enumerate(infos):
            # Banner + log the milestones reached *this step* (the interesting moments).
            for m in info.get("new_milestones", []) or []:
                self._record(m, env_id)
                dirty = True
            # Also advance "furthest" from the full completed set so milestones that
            # were already satisfied at a warm-start snapshot are reflected too.
            for key in info.get("completed_milestone_keys", []) or []:
                idx = self.order.get(key, -1)
                if idx > self.furthest_index:
                    self.furthest_index = idx
                    self.furthest_name = key
                    dirty = True

        if dirty:
            self._save_progress()

        if self.num_timesteps >= self._next_summary:
            self._next_summary += self.summary_every
            self._print_summary()
            self._save_progress()
        return True

    def _print_summary(self) -> None:
        reached = len(self.announced)
        print(f"\n[milestones] --- progress @ step {self.num_timesteps:,} --- "
              f"furthest: '{self.furthest_name}' "
              f"({self.furthest_index + 1}/{self.total_milestones}), "
              f"{reached} milestones cleared, "
              f"best badges={self.best_badges}, best levels={self.best_levels}")
        # show the next few unreached objectives so it's clear what's being worked on
        upcoming = [k for k, i in sorted(self.order.items(), key=lambda kv: kv[1])
                    if k not in self.announced][:3]
        if upcoming:
            print(f"[milestones] next objectives: {', '.join(upcoming)}")

    def _on_training_end(self) -> None:
        self._save_progress()
        self._print_summary()
