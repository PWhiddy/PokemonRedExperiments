"""
skill_bridge.py -- make the RL policy ACT ON the planner's (LLM's) decisions.

The planner (rules or a local LLM) decides *what kind of thing* to do next -- a skill:
``navigation``, ``battle``, ``healing``, ``grinding``, ``story``, or a field move
(``cut``/``surf``/``strength``). That decision is already placed in the observation
(the skill one-hot), so the policy can *see* it. This module is what makes the policy
*follow* it: the :class:`AdviceController` turns "the LLM advised skill X" into a small,
dense, per-step reward whenever the agent's behaviour is **consistent with X**. Trained
under it, the policy learns to do what the planner says -- i.e. the LLM's decisions get
implemented in the model, across every skill, not just Cut.

Two distinct cases:

* **Learnable skills** (navigation/battle/healing/grinding/story/unstuck): PPO *can*
  produce the behaviour; we just bias it toward the advised one with an *alignment*
  reward (e.g. when ``navigation`` is advised, reward actually moving; when ``battle``
  is advised, reward being in / winning a battle; when ``healing`` is advised, reward HP
  going up). The policy, conditioned on the advised skill, learns to comply.

* **Unlearnable field moves** (cut/surf/strength): PPO essentially never discovers the
  menu button sequence, so on top of the alignment reward the controller *executes a
  deterministic macro* (``scripted_helpers.field_use_cut``) once the policy has committed
  (kept engaging the advised obstacle). The policy still learns the learnable part
  (go to the obstacle and trigger), the macro handles the unlearnable buttons.

Reward design is deliberately farm-safe: every alignment bonus is small and gated on a
genuinely desirable event (moving, leveling, healing, advancing a flag, winning a
battle), and the big payoffs still come from the real reward terms (new map / event /
badge / subgoal reach). The macro earns nothing for merely opening a menu -- the env
attributes the large "advice success" bonus only when an execution is immediately
followed by real progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from curriculum.ram_map import GameState
from curriculum.subgoal import Subgoal, MACRO_SKILLS


@dataclass
class AdviceConfig:
    enabled: bool = False
    # general alignment nudge (per step, small) for following the advised skill
    w_align: float = 0.02
    align_cap: float = 0.05          # hard cap on alignment reward per step
    # field-move execution
    execute_field_moves: bool = True
    trigger_bumps: int = 4           # bumps at the advised obstacle before firing the macro
    cooldown: int = 12               # steps between macro fires
    dialogue_taps: int = 10
    follow_cap_steps: int = 6        # stop paying the engage nudge after this many bumps
    require_advice: bool = True      # if False, still rescue Cut heuristically (no planner)


class AdviceController:
    """Rewards the policy for following the planner's advised skill (and executes the
    field-move macros the policy cannot learn)."""

    def __init__(self, cfg: Optional[AdviceConfig] = None):
        self.cfg = cfg or AdviceConfig()
        self.reset()

    def reset(self) -> None:
        self._cooldown = 0
        self._follow_run = 0
        self._battle_credited = False
        self.advice_active = False        # field move advised AND usable right now
        self.following = False            # the agent acted consistently with advice this step
        self.last_executed: Optional[str] = None
        self.advised_skill: str = ""

    # ------------------------------------------------------------------ #
    def _can_use_field(self, gs: GameState, skill: str) -> bool:
        if skill == "cut":
            return gs.can_use_cut()
        if skill == "surf":
            return gs.can_use_surf()
        if skill == "strength":
            return gs.can_use_strength()
        if skill == "flute":
            return gs.has_poke_flute()
        return False

    def _execute_field(self, helpers, skill: str) -> bool:
        if helpers is None:
            return False
        if skill == "cut":
            return bool(helpers.field_use_cut(self.cfg.dialogue_taps))
        if skill == "surf":
            return bool(helpers.field_use_surf(self.cfg.dialogue_taps))
        if skill == "strength":
            return bool(helpers.field_use_strength(self.cfg.dialogue_taps))
        if skill == "flute":
            return bool(helpers.use_poke_flute(self.cfg.dialogue_taps))
        return False

    # ------------------------------------------------------------------ #
    def _alignment(self, gs: GameState, skill: str, comp: Dict[str, float],
                   sig: Dict[str, float]) -> float:
        """Small reward when the agent's behaviour matches the advised learnable skill.

        Reads already-computed per-step signals (``comp`` reward components, ``sig``
        behavioural signals) so it never recomputes game state and never double-pays the
        underlying term -- it adds a *separate* small "you did what was advised" bonus.
        """
        w = self.cfg.w_align
        moved = bool(sig.get("moved", False))
        in_battle = gs.in_battle()
        r = 0.0
        if skill == "navigation":
            if moved and not sig.get("wall_bump", False):
                r += w
        elif skill == "battle":
            if in_battle:
                # credit being in the advised battle once, plus winning (event/level)
                if not self._battle_credited:
                    r += w
                    self._battle_credited = True
                if "event" in comp or "level" in comp:
                    r += w
        elif skill == "grinding":
            if "level" in comp:
                r += w
        elif skill == "healing":
            if "heal" in comp:
                r += w
        elif skill == "story":
            if "event" in comp or "new_map" in comp:
                r += w
        elif skill == "unstuck":
            if moved and not sig.get("stuck", False) and sig.get("loop_score", 0.0) < 0.5:
                r += w
        # shopping: no robust per-step signal yet -> 0
        if not in_battle:
            self._battle_credited = False
        return min(r, self.cfg.align_cap)

    # ------------------------------------------------------------------ #
    def step(self, gs: GameState, anti, subgoal: Optional[Subgoal], helpers,
             comp: Dict[str, float], sig: Dict[str, float]) -> float:
        """Advance one step. Returns the per-step advice-following reward and may fire a
        field-move macro as a side effect. ``last_executed`` / ``advice_active`` /
        ``following`` are updated for the env (obs cue, success attribution, logging)."""
        self.last_executed = None
        self.advice_active = False
        self.following = False
        if not self.cfg.enabled:
            return 0.0
        if self._cooldown > 0:
            self._cooldown -= 1

        skill = (subgoal.skill if subgoal else "") or ""
        self.advised_skill = skill

        # --- macro skill: alignment by engaging + execute the macro --------- #
        if skill in MACRO_SKILLS or (not self.cfg.require_advice and skill == ""):
            eff = skill if skill in MACRO_SKILLS else "cut"
            if self._can_use_field(gs, eff):
                self.advice_active = True
                reward = 0.0
                bumping = getattr(anti, "wall_bump_run", 0) >= 1
                if bumping and self._follow_run < self.cfg.follow_cap_steps:
                    reward += self.cfg.w_align
                    self.following = True
                self._follow_run = self._follow_run + 1 if bumping else 0
                if (self.cfg.execute_field_moves
                        and getattr(anti, "wall_bump_run", 0) >= self.cfg.trigger_bumps
                        and self._cooldown == 0):
                    if self._execute_field(helpers, eff):
                        self.last_executed = eff
                        self._cooldown = self.cfg.cooldown
                return min(reward, self.cfg.align_cap)
            self._follow_run = 0
            return 0.0

        # --- learnable skills: alignment reward ----------------------------- #
        r = self._alignment(gs, skill, comp, sig)
        self.following = r > 0.0
        return r
