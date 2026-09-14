"""
reward_controller.py -- homeostatic, self-adjusting reward weights.

The static reward weights in :class:`curriculum.rewards.RewardConfig` are correct
for *some* phase of the game and wrong for others. Worse, a fixed weight vector is
exploitable: once exploration is exhausted, the densest renewable term (grinding
levels, repeat-healing) dominates the return and PPO settles into farming it instead
of progressing. This is reward hacking, and no single static weight vector prevents
it across all 33 milestones.

:class:`AdaptiveRewardController` makes the *farmable* reward terms homeostatic. Each
step it:

1. tracks an exponential moving average of every reward component's contribution,
2. computes each **farmable** component's share of recent positive reward,
3. notices when *true* progress (a milestone / badge / event / new map) has stalled,
4. and multiplicatively **decays** the weight of any farmable component whose share
   has run away while progress is stalled -- starving the exploit -- then **relaxes**
   the weight back toward 1.0 once progress resumes or the share normalizes.

Crucially, the *objective* components (milestone, badge, event, new_map) are
**protected**: they are never decayed, so the controller can only ever redirect the
agent away from busy-work and toward real progress -- it cannot learn to suppress the
goal itself. Every multiplier is exposed for TensorBoard so the adaptation is visible.

This is the "automatically adjust the rewards to adapt to new challenges" layer: the
weights are no longer hand-tuned constants but a closed control loop around them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


# Components that represent *real* progress and must never be throttled.
PROTECTED: Set[str] = {
    "milestone", "badge", "event", "new_map", "subgoal", "subgoal_reach",
}
# Components that are farmable busy-work and therefore adaptive.
FARMABLE_DEFAULT: Set[str] = {
    "explore", "level", "heal", "pokedex", "intrinsic",
}


@dataclass
class AdaptiveRewardConfig:
    enabled: bool = True
    # EMA smoothing for per-component contribution tracking (0..1, higher = slower).
    ema: float = 0.999
    # A farmable component is "running away" when its share of recent positive
    # reward exceeds this budget.
    share_budget: float = 0.5
    # Steps of no real progress before the controller is allowed to start decaying
    # (so it never punishes a legitimate exploration/grind burst that is paying off).
    stall_patience: int = 1200
    # Multiplicative decay/relax applied per step when (de)activating.
    decay_rate: float = 0.999      # weight *= decay_rate while starving a runaway term
    relax_rate: float = 1.0005     # weight *= relax_rate while restoring toward 1.0
    floor: float = 0.05            # a weight never decays below this fraction
    ceil: float = 1.0              # and never relaxes above this
    farmable: List[str] = field(default_factory=lambda: sorted(FARMABLE_DEFAULT))


class AdaptiveRewardController:
    """Closed-loop controller over farmable reward-component weights."""

    def __init__(self, cfg: AdaptiveRewardConfig | None = None):
        self.cfg = cfg or AdaptiveRewardConfig()
        self.farmable: Set[str] = set(self.cfg.farmable)
        self.reset()

    def reset(self) -> None:
        self.mult: Dict[str, float] = {}        # component -> current weight multiplier
        self._ema_pos: Dict[str, float] = {}    # EMA of positive contribution per comp
        self._ema_total_pos = 1e-6              # EMA of total positive reward
        self.steps_since_progress = 0
        self._last_shares: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def note_progress(self) -> None:
        """Call when a *protected* progress event happened this step."""
        self.steps_since_progress = 0

    # ------------------------------------------------------------------ #
    def adjust(self, components: Dict[str, float], progressed: bool) -> Dict[str, float]:
        """Apply (and update) adaptive multipliers to ``components`` in place-safe.

        Parameters
        ----------
        components:
            The per-step incremental reward dict from :class:`RewardManager.step`.
        progressed:
            True if a protected progress event fired this step.

        Returns
        -------
        The (possibly re-scaled) components dict. Protected components are untouched.
        """
        cfg = self.cfg
        if not cfg.enabled:
            return components

        if progressed:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1

        # --- update EMAs of positive contributions ---
        a = cfg.ema
        step_total_pos = sum(v for v in components.values() if v > 0)
        self._ema_total_pos = a * self._ema_total_pos + (1 - a) * max(step_total_pos, 0.0)
        for comp in self.farmable:
            v = max(components.get(comp, 0.0), 0.0)
            self._ema_pos[comp] = a * self._ema_pos.get(comp, 0.0) + (1 - a) * v

        stalled = self.steps_since_progress > cfg.stall_patience

        # --- update each farmable multiplier ---
        shares: Dict[str, float] = {}
        for comp in self.farmable:
            share = self._ema_pos.get(comp, 0.0) / max(self._ema_total_pos, 1e-6)
            shares[comp] = share
            cur = self.mult.get(comp, 1.0)
            runaway = share > cfg.share_budget
            if stalled and runaway:
                # this term is hogging reward while the agent makes no real progress
                cur *= cfg.decay_rate
            else:
                # restore toward neutral when behaving (or when progress resumes)
                cur *= cfg.relax_rate
            cur = min(max(cur, cfg.floor), cfg.ceil)
            self.mult[comp] = cur
        self._last_shares = shares

        # --- apply multipliers (objective terms are protected) ---
        for comp in list(components.keys()):
            if comp in PROTECTED or comp not in self.farmable:
                continue
            m = self.mult.get(comp, 1.0)
            if m != 1.0:
                components[comp] = components[comp] * m
        return components

    # ------------------------------------------------------------------ #
    def tb_stats(self) -> Dict[str, float]:
        """Multipliers + shares for TensorBoard (prefix added by the callback)."""
        out: Dict[str, float] = {}
        for comp in self.farmable:
            out[f"adapt_mult/{comp}"] = self.mult.get(comp, 1.0)
            out[f"adapt_share/{comp}"] = self._last_shares.get(comp, 0.0)
        out["adapt/steps_since_progress"] = float(self.steps_since_progress)
        return out
