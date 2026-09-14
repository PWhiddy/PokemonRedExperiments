"""
skills.py -- option-policy / skill scaffolding for hierarchical control.

The brief asks us to *start* with a single PPO policy + curriculum rewards, but to
structure the code so reusable "option policies" (skills) and a high-level
controller can be added later without rework.

This module defines:

  - :class:`Skill`            : abstract option interface (init/act/terminated).
  - :class:`ScriptedSkill`    : a skill backed by deterministic logic (wraps the
                                scripted helpers) -- e.g. healing or simple dialogue.
  - :class:`PolicySkill`      : a skill backed by a trained SB3 policy (e.g. a
                                navigation or battle PPO model loaded from disk).
  - :class:`SkillRegistry`    : name -> skill factory lookup.
  - :class:`HighLevelController` : a stub manager policy that, given game state,
                                selects which skill/option should be active. It
                                currently uses simple rules (in battle -> battle
                                skill, etc.) and is the natural place to later drop
                                in a learned manager or an LLM-guided manager.

None of this is required for single-policy curriculum training; it exists so the
training/runtime code can grow into hierarchical RL incrementally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from curriculum.ram_map import GameState


class Skill(ABC):
    """An option/skill: a temporally-extended sub-policy with its own termination."""

    name: str = "skill"

    def on_start(self, gs: GameState) -> None:
        """Called when this skill becomes active."""

    @abstractmethod
    def act(self, observation, gs: GameState) -> int:
        """Return a primitive action index for the current step."""

    def terminated(self, gs: GameState) -> bool:
        """Return True when the skill has finished its sub-goal."""
        return False


class ScriptedSkill(Skill):
    """A skill whose behaviour is a deterministic callback over the game state."""

    def __init__(self, name: str, policy_fn: Callable[[object, GameState], int],
                 done_fn: Optional[Callable[[GameState], bool]] = None):
        self.name = name
        self._policy_fn = policy_fn
        self._done_fn = done_fn or (lambda gs: False)

    def act(self, observation, gs: GameState) -> int:
        return self._policy_fn(observation, gs)

    def terminated(self, gs: GameState) -> bool:
        return self._done_fn(gs)


class PolicySkill(Skill):
    """A skill backed by a trained Stable-Baselines3 policy loaded from disk.

    Loading is lazy so importing this module never requires torch/SB3.
    """

    def __init__(self, name: str, model_path: str,
                 done_fn: Optional[Callable[[GameState], bool]] = None):
        self.name = name
        self.model_path = model_path
        self._model = None
        self._done_fn = done_fn or (lambda gs: False)

    def _ensure_loaded(self):
        if self._model is None:
            from stable_baselines3 import PPO
            self._model = PPO.load(self.model_path)
        return self._model

    def act(self, observation, gs: GameState) -> int:
        model = self._ensure_loaded()
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    def terminated(self, gs: GameState) -> bool:
        return self._done_fn(gs)


@dataclass
class SkillRegistry:
    """Name -> factory registry so skills can be declared once and reused."""
    _factories: Dict[str, Callable[[], Skill]] = None

    def __post_init__(self):
        if self._factories is None:
            self._factories = {}

    def register(self, name: str, factory: Callable[[], Skill]) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> Skill:
        if name not in self._factories:
            raise KeyError(f"Unknown skill '{name}'. Registered: {list(self._factories)}")
        return self._factories[name]()

    def names(self):
        return list(self._factories)


# Suggested skill names (see brief). Register factories for these as they are built.
SUGGESTED_SKILLS = [
    "navigation",     # move toward a target map/coord
    "battle",         # fight a wild/trainer battle
    "healing",        # restore HP at a Poke Center
    "shopping",       # buy items at a mart
    "grinding",       # level up the party
    "story",          # advance scripted story beats
    "unstuck",        # escape loops / stuck states
]


class HighLevelController:
    """Rule-based manager that selects which skill should be active.

    This is a deliberately simple stub: it demonstrates the option-selection
    interface that a learned (or LLM-guided) manager would later implement. It does
    not run any skill by itself -- the runtime decides whether to honour its choice.
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.active: Optional[Skill] = None
        self.active_name: Optional[str] = None

    def select(self, gs: GameState) -> Optional[str]:
        """Return the name of the skill that *should* be active, or None.

        Simple heuristics; replace with a trained policy / LLM manager later.
        """
        names = set(self.registry.names())
        if gs.in_battle() and "battle" in names:
            return "battle"
        if gs.total_hp_fraction() < 0.25 and "healing" in names:
            return "healing"
        if "navigation" in names:
            return "navigation"
        return None

    def step(self, observation, gs: GameState) -> Optional[int]:
        """Pick/maintain a skill and return its primitive action (or None)."""
        desired = self.select(gs)
        if desired is None:
            return None
        if self.active_name != desired or (self.active and self.active.terminated(gs)):
            self.active = self.registry.create(desired)
            self.active_name = desired
            self.active.on_start(gs)
        return self.active.act(observation, gs)


class PlannerManager(HighLevelController):
    """A manager whose skill choice is driven by the high-level planner's subgoal.

    This is the explicit "manager selects a specialist agent" layer (the hierarchical
    / multi-policy control the project is reaching for). It maps the planner's
    ``subgoal.skill`` to a registered specialist option (a :class:`PolicySkill` loaded
    from a trained model, or a :class:`ScriptedSkill`). Until a specialist is
    registered for the requested skill, ``step`` returns ``None`` and the base
    subgoal-conditioned PPO policy stays in control -- so this composes with, rather
    than replaces, the single-policy trainer.

    The env already feeds the chosen skill + target into the observation, so the PPO
    policy is *subgoal-conditioned* even when no specialist overrides it; registering
    specialists here lets you hand specific situations (e.g. battles) to dedicated
    models without retraining the whole stack.
    """

    def __init__(self, registry: SkillRegistry):
        super().__init__(registry)
        self._subgoal_skill: Optional[str] = None

    def set_subgoal_skill(self, skill_name: Optional[str]) -> None:
        self._subgoal_skill = skill_name

    def select(self, gs: GameState) -> Optional[str]:
        names = set(self.registry.names())
        # battles always go to the battle specialist if one exists (safety override)
        if gs.in_battle() and "battle" in names:
            return "battle"
        if self._subgoal_skill in names:
            return self._subgoal_skill
        return super().select(gs)
