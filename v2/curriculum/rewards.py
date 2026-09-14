"""
rewards.py -- a configurable, logged reward system for the curriculum env.

Design goals (from the project brief):

* Reward *meaningful* progress: new maps, story event flags, badges, gym/trainer
  wins, key items, HMs/TMs, Pokedex, milestone completion, sensible level growth.
* Penalize degenerate behaviour: blackouts, coordinate loops, excessive menuing,
  wall bumping, inactivity, repeated healing without progress, farming the same
  reward, getting stuck in battle/menu/dialogue loops, very long no-progress runs.
* Make milestone rewards **one-time / max-based** so they cannot be farmed.
* Expose every component for TensorBoard / logging.

The manager returns, each step, a dict mapping component name -> **incremental**
reward for that step (SB3 consumes the sum). It also keeps cumulative totals for
logging and derives the behavioural "signal" scores (stuck/loop/...) that feed the
structured observation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set

from curriculum.ram_map import GameState
from curriculum.milestones import MilestoneManager
from curriculum.structured_obs import EpisodeSignals
from curriculum.intrinsic import build_novelty
from curriculum.reward_controller import (
    AdaptiveRewardController, AdaptiveRewardConfig)

MOVEMENT_ACTIONS = {0, 1, 2, 3}   # down, left, right, up
START_ACTION = 6


@dataclass
class RewardConfig:
    """All reward weights and anti-abuse thresholds in one place.

    Loadable from YAML (see ``configs/curriculum.yaml``). Set any weight to 0 to
    disable that component.
    """
    # positive components
    w_milestone: float = 1.0          # multiplies each milestone's own reward
    w_new_map: float = 5.0            # first time a new map is entered
    w_new_coord: float = 0.02         # each newly seen (x,y,map) tile
    w_event: float = 4.0              # per newly set story event flag (max-based)
    w_badge: float = 25.0             # per badge gained (max-based)
    w_level: float = 1.0              # scaled party-level growth (max-based)
    w_pokedex: float = 1.0            # per newly owned Pokedex entry (max-based)
    w_heal: float = 3.0               # healing (squared fraction, like base env)

    # penalty components (positive numbers; applied as negatives)
    p_blackout: float = 25.0          # per blackout (whole party fainted)
    p_stuck: float = 0.05             # standing on an over-visited tile
    p_wall_bump: float = 0.02         # movement action with no position change
    p_menu_spam: float = 0.02         # opening menu (START) too frequently
    p_inactivity: float = 0.01        # per step once no-progress window exceeded
    p_heal_farm: float = 0.5          # healing repeatedly without progress
    p_loop: float = 0.1               # cycling within a tiny region

    # thresholds
    stuck_visit_threshold: int = 400      # tile visit count that counts as "stuck"
    inactivity_patience: int = 1500       # steps without progress before penalty
    loop_window: int = 200                # window for loop detection
    loop_unique_min: int = 12             # < this many unique tiles in window => loop
    menu_window: int = 50                 # window for menu-spam detection
    menu_spam_max: int = 12               # > this many START presses in window => spam
    heal_farm_patience: int = 2           # heals without progress before penalty
    level_explore_thresh: int = 22        # diminishing-returns knee (matches base env)
    level_scale_factor: float = 4.0

    # global scaling (parallels base env's reward_scale)
    reward_scale: float = 1.0

    # --- intrinsic motivation (non-saturating exploration) ---
    intrinsic_kind: str = "count"         # none | count | rnd
    w_intrinsic: float = 0.3              # bonus coefficient (coef at first visit)
    intrinsic_power: float = 0.5          # decay exponent: 0.5 => 1/sqrt(N)
    intrinsic_xy_bucket: int = 2          # tile coarseness for the count hash
    intrinsic_event_bucket: int = 4       # story-flag chunk that refreshes novelty

    # --- adaptive reward controller (homeostatic, anti-reward-hacking) ---
    adaptive_enabled: bool = True
    adaptive_ema: float = 0.999
    adaptive_share_budget: float = 0.5    # a farmable term hogging > this share is throttled
    adaptive_stall_patience: int = 1200   # steps of no real progress before throttling
    adaptive_decay_rate: float = 0.999
    adaptive_relax_rate: float = 1.0005
    adaptive_floor: float = 0.05


class AntiLoopTracker:
    """Detects stuck/looping/wall-bumping/inactivity and produces 0..1 scores."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.coord_visits: Dict[str, int] = {}
        self.last_pos = None
        self.recent_coords: Deque[str] = deque(maxlen=self.cfg.loop_window)
        self.recent_starts: Deque[int] = deque(maxlen=self.cfg.menu_window)
        self.steps_since_progress = 0
        self.heals_since_progress = 0
        self.wall_bump_run = 0

    def update(self, gs: GameState, action: int):
        """Update internal counters. Returns ``(moved, signals)`` where signals is
        a dict with: stuck (bool), wall_bump (bool), menu_spam (bool),
        loop_score (float), and the raw scores for the observation."""
        x, y, m = gs.position()
        coord = f"{x},{y},{m}"
        in_battle = gs.in_battle()

        # visit counting (overworld only, like base env)
        moved = self.last_pos is not None and (x, y, m) != self.last_pos
        if not in_battle:
            self.coord_visits[coord] = self.coord_visits.get(coord, 0) + 1
            self.recent_coords.append(coord)

        # wall bump: tried to move but position unchanged and not in battle/menu
        wall_bump = (action in MOVEMENT_ACTIONS and not in_battle
                     and self.last_pos is not None and (x, y, m) == self.last_pos)
        self.wall_bump_run = self.wall_bump_run + 1 if wall_bump else 0

        # menu spam
        self.recent_starts.append(1 if action == START_ACTION else 0)
        menu_presses = sum(self.recent_starts)
        menu_spam = menu_presses > self.cfg.menu_spam_max

        # stuck: standing on an over-visited tile
        stuck = self.coord_visits.get(coord, 0) > self.cfg.stuck_visit_threshold

        # loop: very few unique tiles across the recent window
        unique_recent = len(set(self.recent_coords))
        loop_score = 0.0
        if len(self.recent_coords) >= self.cfg.loop_window:
            loop_score = max(0.0, 1.0 - unique_recent / max(self.cfg.loop_unique_min, 1))
            loop_score = min(loop_score, 1.0)
        is_loop = loop_score > 0.0

        self.steps_since_progress += 1
        self.last_pos = (x, y, m)

        signals = {
            "moved": moved,
            "wall_bump": wall_bump,
            "menu_spam": menu_spam,
            "menu_presses": menu_presses,
            "stuck": stuck,
            "is_loop": is_loop,
            "loop_score": loop_score,
            "unique_recent": unique_recent,
        }
        return signals

    def note_progress(self) -> None:
        """Call whenever real progress happens (new map/event/badge/level/...)."""
        self.steps_since_progress = 0
        self.heals_since_progress = 0

    # derived 0..1 scores for the structured observation
    def stuck_score(self, coord_visits_for_current: int) -> float:
        return min(coord_visits_for_current / max(self.cfg.stuck_visit_threshold, 1), 1.0)

    def inactivity_score(self) -> float:
        return min(self.steps_since_progress / max(self.cfg.inactivity_patience, 1), 1.0)

    def wall_bump_score(self) -> float:
        return min(self.wall_bump_run / 10.0, 1.0)


class RewardManager:
    """Computes per-step incremental reward and exposes logging + obs signals."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.anti = AntiLoopTracker(cfg)
        # non-saturating intrinsic motivation (None if disabled). RND needs an
        # obs_fn (wired by the env); from here we can always build the count module.
        self.novelty = build_novelty(
            cfg.intrinsic_kind, coef=cfg.w_intrinsic,
            power=cfg.intrinsic_power, xy_bucket=cfg.intrinsic_xy_bucket,
            event_bucket=cfg.intrinsic_event_bucket,
        ) if cfg.intrinsic_kind != "rnd" else None
        # homeostatic controller over farmable component weights
        self.controller = AdaptiveRewardController(AdaptiveRewardConfig(
            enabled=cfg.adaptive_enabled, ema=cfg.adaptive_ema,
            share_budget=cfg.adaptive_share_budget,
            stall_patience=cfg.adaptive_stall_patience,
            decay_rate=cfg.adaptive_decay_rate, relax_rate=cfg.adaptive_relax_rate,
            floor=cfg.adaptive_floor,
        ))
        self.reset()

    def attach_novelty(self, novelty) -> None:
        """Inject an externally-built novelty module (e.g. RND needing an obs_fn)."""
        self.novelty = novelty

    def reset(self) -> None:
        self.anti.reset()
        if self.novelty is not None:
            self.novelty.reset()
        self.controller.reset()
        self.seen_coords: Set[str] = set()
        self.seen_maps: Set[int] = set()
        self.max_event_sum = 0
        self.badges_awarded = 0
        self.max_level_metric = 0.0
        self.max_pokedex = 0
        self.heal_total = 0.0
        self.blackout_count = 0
        self.heal_count = 0
        self.pokecenter_used_recently = 0
        self._last_hp_fraction = None
        self._last_party_count = None
        self.cumulative: Dict[str, float] = {}
        self.last_signals: Dict[str, float] = {}
        self.last_components: Dict[str, float] = {}

    def baseline(self, gs: GameState) -> None:
        """Anchor the max-based accumulators to the (possibly mid-game) start state.

        Call once right after loading a warm-start snapshot so already-satisfied
        progress (events, badges, levels, current map) is *not* re-rewarded on the
        first steps of the episode.
        """
        self.max_event_sum = gs.event_flags_sum()
        self.badges_awarded = gs.badge_count()
        self.max_level_metric = self._level_metric(gs)
        self.max_pokedex = gs.pokedex_owned_count()
        self.seen_maps = {gs.map_id()}
        self._last_hp_fraction = gs.total_hp_fraction()
        self._last_party_count = gs.party_count()
        self.anti.last_pos = gs.position()

    # ------------------------------------------------------------------ #
    def _level_metric(self, gs: GameState) -> float:
        """Diminishing-returns party-level metric (mirrors base env)."""
        level_sum = gs.party_levels_sum()
        thresh = self.cfg.level_explore_thresh
        if level_sum < thresh:
            return float(level_sum)
        return (level_sum - thresh) / self.cfg.level_scale_factor + thresh

    def _accumulate(self, comp: Dict[str, float], key: str, val: float) -> None:
        if val != 0.0:
            comp[key] = comp.get(key, 0.0) + val
            self.cumulative[key] = self.cumulative.get(key, 0.0) + val

    # ------------------------------------------------------------------ #
    def step(self, gs: GameState, action: int, mm: MilestoneManager) -> Dict[str, float]:
        """Return a dict of incremental reward components for this step."""
        cfg = self.cfg
        comp: Dict[str, float] = {}
        progressed = False
        # "real" progress = an objective event (milestone/map/event/badge). This is
        # stricter than ``progressed`` (which also counts level/pokedex) and is what
        # the adaptive controller uses to decide that the agent has *stalled*.
        real_progress = False

        sig = self.anti.update(gs, action)
        x, y, m = gs.position()
        coord = f"{x},{y},{m}"

        # --- milestone rewards (one-time) ---
        for key in mm.newly_completed:
            ms = mm.by_key[key]
            self._accumulate(comp, "milestone", cfg.w_milestone * ms.reward)
            progressed = True
            real_progress = True

        # --- new map ---
        if m not in self.seen_maps:
            self.seen_maps.add(m)
            self._accumulate(comp, "new_map", cfg.w_new_map)
            progressed = True
            real_progress = True

        # --- intrinsic novelty (non-saturating curiosity) ---
        if self.novelty is not None:
            self._accumulate(comp, "intrinsic", self.novelty.bonus(gs))

        # --- new coordinate (exploration breadcrumb) ---
        if not gs.in_battle() and coord not in self.seen_coords:
            self.seen_coords.add(coord)
            self._accumulate(comp, "explore", cfg.w_new_coord)

        # --- story events (max-based: only reward net-new flags) ---
        event_sum = gs.event_flags_sum()
        if event_sum > self.max_event_sum:
            gained = event_sum - self.max_event_sum
            self.max_event_sum = event_sum
            self._accumulate(comp, "event", cfg.w_event * gained)
            progressed = True
            real_progress = True

        # --- badges (max-based) ---
        badge_count = gs.badge_count()
        if badge_count > self.badges_awarded:
            gained = badge_count - self.badges_awarded
            self.badges_awarded = badge_count
            self._accumulate(comp, "badge", cfg.w_badge * gained)
            progressed = True
            real_progress = True

        # --- level growth (max-based, diminishing returns) ---
        lvl = self._level_metric(gs)
        if lvl > self.max_level_metric:
            self._accumulate(comp, "level", cfg.w_level * (lvl - self.max_level_metric))
            self.max_level_metric = lvl
            progressed = True

        # --- pokedex (max-based) ---
        owned = gs.pokedex_owned_count()
        if owned > self.max_pokedex:
            self._accumulate(comp, "pokedex", cfg.w_pokedex * (owned - self.max_pokedex))
            self.max_pokedex = owned
            progressed = True

        # --- healing / blackout ---
        self._healing_and_blackout(gs, comp)

        # --- penalties --------------------------------------------------- #
        if sig["stuck"]:
            self._accumulate(comp, "stuck_pen", -cfg.p_stuck)
        if sig["wall_bump"]:
            self._accumulate(comp, "wall_bump_pen", -cfg.p_wall_bump)
        if sig["menu_spam"]:
            self._accumulate(comp, "menu_spam_pen", -cfg.p_menu_spam)
        if sig["is_loop"]:
            self._accumulate(comp, "loop_pen", -cfg.p_loop * sig["loop_score"])
        if self.anti.steps_since_progress > cfg.inactivity_patience:
            self._accumulate(comp, "inactivity_pen", -cfg.p_inactivity)

        # progress bookkeeping (resets inactivity / heal-farm windows)
        if progressed:
            self.anti.note_progress()

        # --- homeostatic adaptation: throttle any farmable term that is running
        #     away while real progress has stalled (anti-reward-hacking). Protected
        #     objective terms (milestone/badge/event/new_map) are never touched. ---
        comp = self.controller.adjust(comp, real_progress)

        # cache signals for the observation builder + logging
        self.last_signals = {
            **sig,
            "stuck_score": self.anti.stuck_score(self.anti.coord_visits.get(coord, 0)),
            "inactivity_score": self.anti.inactivity_score(),
            "wall_bump_score": self.anti.wall_bump_score(),
        }
        self.last_components = comp
        return comp

    def _healing_and_blackout(self, gs: GameState, comp: Dict[str, float]) -> None:
        cfg = self.cfg
        cur_hp = gs.total_hp_fraction()
        party = gs.party_count()
        if self._last_hp_fraction is not None:
            if cur_hp > self._last_hp_fraction and party == self._last_party_count:
                if self._last_hp_fraction > 0:
                    heal_amt = cur_hp - self._last_hp_fraction
                    self.heal_total += heal_amt * heal_amt
                    self.heal_count += 1
                    self.pokecenter_used_recently = 1
                    self._accumulate(comp, "heal", cfg.w_heal * heal_amt * heal_amt)
                    # repeated healing without progress => discourage PC farming
                    self.anti.heals_since_progress += 1
                    if self.anti.heals_since_progress > cfg.heal_farm_patience:
                        self._accumulate(comp, "heal_farm_pen", -cfg.p_heal_farm)
                else:
                    # HP went up from 0 with same party size => a revive/blackout
                    self.blackout_count += 1
                    self._accumulate(comp, "blackout_pen", -cfg.p_blackout)
            else:
                self.pokecenter_used_recently = 0
        # explicit blackout: whole party fainted
        if gs.all_fainted():
            # only count the transition once per faint episode
            if self._last_hp_fraction is None or self._last_hp_fraction > 0:
                self.blackout_count += 1
                self._accumulate(comp, "blackout_pen", -cfg.p_blackout)
        self._last_hp_fraction = cur_hp
        self._last_party_count = party

    # ------------------------------------------------------------------ #
    def episode_signals(self, mm: MilestoneManager, recent_actions: List[int],
                        recent_maps: List[int], num_milestones: int) -> EpisodeSignals:
        """Bundle the behavioural signals for the structured observation."""
        s = self.last_signals
        bits = [1 if ms.key in mm.completed else 0 for ms in mm.milestones][:num_milestones]
        return EpisodeSignals(
            recent_actions=recent_actions,
            recent_maps=recent_maps,
            heal_count=self.heal_count,
            blackout_count=self.blackout_count,
            pokecenter_used=self.pokecenter_used_recently,
            stuck_score=float(s.get("stuck_score", 0.0)),
            loop_score=float(s.get("loop_score", 0.0)),
            wall_bump_score=float(s.get("wall_bump_score", 0.0)),
            inactivity_score=float(s.get("inactivity_score", 0.0)),
            milestone_bits=bits,
        )

    def scaled_total(self, comp: Dict[str, float]) -> float:
        return self.cfg.reward_scale * sum(comp.values())

    def adaptive_stats(self) -> Dict[str, float]:
        """Adaptive-controller multipliers + intrinsic stats for TensorBoard."""
        out = dict(self.controller.tb_stats())
        if self.novelty is not None:
            out.update(self.novelty.stats())
        return out
