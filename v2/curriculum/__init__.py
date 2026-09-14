"""
Curriculum / Hierarchical RL extension for PokemonRedExperiments (v2).

This package layers a practical, incremental hierarchical-RL / curriculum-learning
system on top of the existing ``RedGymEnv`` (v2) environment **without modifying it**.
Everything here is additive: the original ``baseline_fast_v2.py`` and
``red_gym_env_v2.py`` keep working unchanged.

High level pieces
-----------------
- ``ram_map.GameState``       : single source of truth for reading structured game
                                state out of PyBoy RAM (position, party, battle,
                                badges, events, money, items, HMs/TMs, ...).
- ``structured_obs``          : turns ``GameState`` + episode history into a flat
                                float32 observation vector (and the matching space).
- ``milestones``              : the curriculum graph (Start Game -> ... -> Champion),
                                per-milestone success detectors, timeouts, start
                                states and transitions, plus a ``MilestoneManager``.
- ``rewards``                 : a fully configurable, logged reward system with
                                one-time / max-based milestone rewards and anti-loop
                                / anti-farming penalties.
- ``scripted_helpers``        : deterministic helpers (advance dialogue, heal at a
                                Poke Center, ...) so menus never hard-block progress.
- ``skills``                  : option-policy / skill scaffolding + a high-level
                                controller stub for later hierarchical control.
- ``state_store``             : save/load successful emulator snapshots per milestone
                                so later curriculum stages can start from them.
- ``config``                  : dataclasses + YAML loading for the whole system.

The glue that wires these into a Gymnasium env lives one level up in
``v2/curriculum_env.py`` (``CurriculumRedGymEnv``), and the staged training driver
is ``v2/train_full_game_curriculum.py``.
"""

from curriculum.ram_map import GameState
from curriculum.milestones import Milestone, MilestoneManager, build_default_milestones
from curriculum.rewards import RewardConfig, RewardManager
from curriculum.structured_obs import StructuredObservationBuilder
from curriculum.state_store import StateStore
from curriculum.intrinsic import build_novelty, HashCountNovelty
from curriculum.reward_controller import AdaptiveRewardController, AdaptiveRewardConfig
from curriculum.subgoal import Subgoal, SubgoalShaper, SKILL_VOCAB
from curriculum.llm_planner import build_planner, RuleBasedPlanner, OllamaPlanner
from curriculum.skill_bridge import AdviceController, AdviceConfig

__all__ = [
    "GameState",
    "Milestone",
    "MilestoneManager",
    "build_default_milestones",
    "RewardConfig",
    "RewardManager",
    "StructuredObservationBuilder",
    "StateStore",
    "build_novelty",
    "HashCountNovelty",
    "AdaptiveRewardController",
    "AdaptiveRewardConfig",
    "Subgoal",
    "SubgoalShaper",
    "SKILL_VOCAB",
    "build_planner",
    "RuleBasedPlanner",
    "OllamaPlanner",
    "AdviceController",
    "AdviceConfig",
]
