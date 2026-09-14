"""
config.py -- typed configuration for the curriculum env + training driver.

Loadable from YAML or JSON. A single :class:`CurriculumConfig` bundles:

  - emulator/env settings (ROM path, init state, headless, action_freq, ...)
  - observation settings (structured vector on/off, keep pixel screens, history len)
  - the :class:`curriculum.rewards.RewardConfig` (all reward weights)
  - curriculum/staging settings (state-store location, start-state strategy, ...)
  - scripted-helper toggles

The env consumes a plain dict (for backwards compatibility with ``RedGymEnv``), so
:meth:`CurriculumConfig.to_env_config` flattens everything into the dict the env
reads in ``__init__``.

YAML support requires PyYAML (``pip install pyyaml``); JSON works with the stdlib.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from curriculum.rewards import RewardConfig


@dataclass
class CurriculumConfig:
    # --- emulator / env ---
    gb_path: str = "../PokemonRed.gb"
    init_state: str = "../init.state"
    headless: bool = True
    action_freq: int = 24
    max_steps: int = 2048 * 16
    save_video: bool = False
    fast_video: bool = True
    save_final_state: bool = False
    print_rewards: bool = False
    session_path: str = "runs_curriculum"
    emulation_speed: int = 0          # 0 = unlimited (headless); >0 throttles visual

    # --- observation ---
    use_structured_obs: bool = True
    use_screens: bool = True          # keep base pixel stream as auxiliary input
    history_len: int = 8

    # --- curriculum / staging ---
    state_store_root: str = "curriculum_states"
    target_milestone: Optional[str] = None   # which milestone this stage trains
    start_state_strategy: str = "random"      # random | latest | best
    save_success_states: bool = True
    success_state_min_reward: float = 0.0     # only snapshot if episode reward >= this
    end_on_target: bool = False               # end episode when target milestone done
    end_on_blackout: bool = False             # end episode on a blackout

    # --- Go-Explore warm-restart (works without a target milestone too) ---
    start_state_prob: float = 0.0             # P(restart from a saved frontier state)
    frontier_window: int = 4                  # sample among the furthest N saved states

    # --- efficiency ---
    max_steps_without_progress: int = 0       # truncate stalled episodes (0 = off)

    # --- scripted helpers ---
    use_scripted_helpers: bool = False
    auto_advance_dialogue: bool = False
    dialogue_taps: int = 8
    auto_use_cut: bool = False                # deterministic Cut when stuck (experimental)
    cut_trigger_bumps: int = 6                # wall-bumps in a row before trying Cut

    # --- high-level planner (the "think, then learn" layer) ---
    use_planner: bool = False                 # master switch for planner + subgoal shaping
    planner_kind: str = "rule"                # none | rule | ollama
    planner_model: str = "nemotron-mini"      # ollama model id (when planner_kind=ollama)
    planner_host: str = "http://localhost:11434"
    planner_timeout: float = 20.0             # seconds before falling back to rules
    replan_interval: int = 512                # steps between routine replans
    replan_on_stuck: bool = True              # also replan when stuck/loop signals spike
    planner_verbose: bool = True              # print a CLI panel on every genuine LLM decision
    subgoal_shaping_weight: float = 1.0       # scale of potential-based subgoal shaping
    subgoal_reach_bonus: float = 8.0          # one-time bonus for reaching the subgoal map

    # --- advice controller: make the policy FOLLOW the planner's decisions ---
    use_advice: bool = False                  # reward acting consistently with advised skill
    advice_w_align: float = 0.02              # per-step nudge for following the advised skill
    advice_trigger_bumps: int = 4             # bumps at an advised obstacle before macro fires
    advice_success_bonus: float = 5.0         # one-time bonus when a field-move opens progress
    advice_require_planner: bool = True       # only act on advice when a planner is active

    # --- reward weights (nested) ---
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumConfig":
        d = dict(d or {})
        reward_d = d.pop("reward", {}) or {}
        known = {f.name for f in fields(cls) if f.name != "reward"}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = set(d) - known
        if unknown:
            print(f"[CurriculumConfig] ignoring unknown keys: {sorted(unknown)}")
        reward_known = {f.name for f in fields(RewardConfig)}
        reward_kwargs = {k: v for k, v in reward_d.items() if k in reward_known}
        return cls(reward=RewardConfig(**reward_kwargs), **kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "CurriculumConfig":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "Loading a .yaml config requires PyYAML: pip install pyyaml "
                    "(or provide a .json config instead)."
                ) from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ #
    def to_env_config(self) -> dict:
        """Flatten into the dict ``CurriculumRedGymEnv`` / ``RedGymEnv`` expect.

        Keeps every key the base ``RedGymEnv`` reads, plus the curriculum extras.
        """
        return {
            # base RedGymEnv keys
            "session_path": Path(self.session_path),
            "gb_path": self.gb_path,
            "init_state": self.init_state,
            "headless": self.headless,
            "action_freq": self.action_freq,
            "max_steps": self.max_steps,
            "save_video": self.save_video,
            "fast_video": self.fast_video,
            "save_final_state": self.save_final_state,
            "print_rewards": self.print_rewards,
            "reward_scale": self.reward.reward_scale,
            "explore_weight": 1.0,   # exploration handled by RewardManager instead
            "early_stop": False,
            "debug": False,
            # curriculum extras
            "emulation_speed": self.emulation_speed,
            "use_structured_obs": self.use_structured_obs,
            "use_screens": self.use_screens,
            "history_len": self.history_len,
            "state_store_root": self.state_store_root,
            "target_milestone": self.target_milestone,
            "start_state_strategy": self.start_state_strategy,
            "save_success_states": self.save_success_states,
            "success_state_min_reward": self.success_state_min_reward,
            "end_on_target": self.end_on_target,
            "end_on_blackout": self.end_on_blackout,
            "start_state_prob": self.start_state_prob,
            "frontier_window": self.frontier_window,
            "max_steps_without_progress": self.max_steps_without_progress,
            "use_scripted_helpers": self.use_scripted_helpers,
            "auto_advance_dialogue": self.auto_advance_dialogue,
            "dialogue_taps": self.dialogue_taps,
            "auto_use_cut": self.auto_use_cut,
            "cut_trigger_bumps": self.cut_trigger_bumps,
            # planner / subgoal layer
            "use_planner": self.use_planner,
            "planner_kind": self.planner_kind,
            "planner_model": self.planner_model,
            "planner_host": self.planner_host,
            "planner_timeout": self.planner_timeout,
            "replan_interval": self.replan_interval,
            "replan_on_stuck": self.replan_on_stuck,
            "planner_verbose": self.planner_verbose,
            "subgoal_shaping_weight": self.subgoal_shaping_weight,
            "subgoal_reach_bonus": self.subgoal_reach_bonus,
            # advice controller
            "use_advice": self.use_advice,
            "advice_w_align": self.advice_w_align,
            "advice_trigger_bumps": self.advice_trigger_bumps,
            "advice_success_bonus": self.advice_success_bonus,
            "advice_require_planner": self.advice_require_planner,
            "reward_config": self.reward,
        }
