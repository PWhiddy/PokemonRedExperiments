"""
structured_obs.py -- build a flat, normalized observation vector from game state.

The RL agent should not depend only on pixels. This module turns a
:class:`curriculum.ram_map.GameState` plus a small bundle of per-episode signals
into a single fixed-length ``float32`` vector (roughly 0..1 normalized) and exposes
the matching ``gymnasium`` Box space.

The pixel/screen observation from the base ``RedGymEnv`` can remain as an optional
auxiliary stream (``CurriculumRedGymEnv`` keeps it under the ``"screens"`` key);
this structured vector is the primary, sample-efficient signal.

Everything is schema-driven: :meth:`StructuredObservationBuilder.build` appends
features in a fixed order and asserts the final length equals :attr:`dim`, so the
observation space and the produced vectors can never silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from gymnasium import spaces

from curriculum.ram_map import GameState
from curriculum.subgoal import SKILL_VOCAB


@dataclass
class EpisodeSignals:
    """Dynamic, history-derived signals the env feeds into the observation.

    These cannot be read from a single RAM snapshot; the env maintains them across
    steps (recent actions, map transitions, loop/stuck detection, heal/blackout
    counters, and which milestones are done).
    """
    recent_actions: List[int] = field(default_factory=list)   # most-recent-first
    recent_maps: List[int] = field(default_factory=list)      # most-recent-first
    heal_count: int = 0
    blackout_count: int = 0
    pokecenter_used: int = 0          # 1 if a Poke Center heal happened recently
    stuck_score: float = 0.0          # 0..1, high => revisiting same coords
    loop_score: float = 0.0           # 0..1, high => cycling a small region
    wall_bump_score: float = 0.0      # 0..1, high => repeatedly bumping walls
    inactivity_score: float = 0.0     # 0..1, high => no new progress for a while
    milestone_bits: List[int] = field(default_factory=list)   # completed flags
    # --- high-level subgoal (from the planner) so the policy is subgoal-conditioned ---
    subgoal_active: float = 0.0       # 1 if a planner subgoal is currently set
    subgoal_target_map: int = 0       # target map id (0 if none)
    subgoal_hops: float = 0.0         # 0..1 normalized map-hop distance to target
    subgoal_skill_id: int = -1        # index into subgoal.SKILL_VOCAB (-1 = none)
    advice_active: float = 0.0        # 1 if a field move (cut/surf/..) is advised AND usable now


class StructuredObservationBuilder:
    """Construct the structured observation vector and its space."""

    def __init__(self, num_actions: int = 7, history_len: int = 8,
                 num_milestones: int = 0):
        self.num_actions = num_actions
        self.history_len = history_len
        self.num_milestones = num_milestones
        # Compute the fixed dimension once, from the schema below.
        self.dim = self._compute_dim()

    # ------------------------------------------------------------------ #
    def _compute_dim(self) -> int:
        d = 0
        d += 4                       # map_id, prev_map_id, x, y
        d += 8 + 1                   # badge bits + badge count
        d += 1                       # event flags sum (normalized)
        d += 6                       # party species
        d += 6                       # party levels
        d += 6                       # party hp fractions
        d += 6                       # party has-status flags
        d += 2                       # party size, total hp fraction
        d += 5                       # in_battle, battle_type, enemy species/level/hp
        d += 1                       # money (normalized)
        d += 1                       # pokeball count (normalized)
        d += 3                       # has cut / surf / strength
        d += 2                       # hm count, tm count (normalized)
        d += 4                       # parcel, ss ticket, town map, bicycle
        d += 2                       # pokedex owned count, has_pokedex
        d += 6                       # heal/blackout/pokecenter/stuck/loop/inactivity
        d += 1                       # wall bump score
        d += 3                       # subgoal: active, target_map, hops
        d += 1                       # advice_active (field move advised + usable now)
        d += len(SKILL_VOCAB)        # subgoal skill one-hot
        d += self.num_actions * self.history_len   # recent actions (one-hot history)
        d += self.history_len        # recent map ids (normalized history)
        d += self.num_milestones     # milestone completion bits
        return d

    def space(self) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(self.dim,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def build(self, gs: GameState, sig: EpisodeSignals) -> np.ndarray:
        f: List[float] = []

        # --- position / map ---
        f.append(gs.map_id() / 255.0)
        f.append(gs.prev_map_id() / 255.0)
        x, y, _ = gs.position()
        f.append(min(x, 255) / 255.0)
        f.append(min(y, 255) / 255.0)

        # --- badges / events ---
        f.extend(gs.badge_bits())
        f.append(gs.badge_count() / 8.0)
        f.append(min(gs.event_flags_sum(), 320) / 320.0)

        # --- party ---
        mons = gs.party_mons()
        species = [m.species for m in mons] + [0] * (6 - len(mons))
        levels = [m.level for m in mons] + [0] * (6 - len(mons))
        hpfr = [m.hp_fraction for m in mons] + [0.0] * (6 - len(mons))
        status = [1.0 if m.status != 0 else 0.0 for m in mons] + [0.0] * (6 - len(mons))
        f.extend([s / 255.0 for s in species[:6]])
        f.extend([min(l, 100) / 100.0 for l in levels[:6]])
        f.extend([float(np.clip(h, 0.0, 1.0)) for h in hpfr[:6]])
        f.extend(status[:6])
        f.append(gs.party_count() / 6.0)
        f.append(float(np.clip(gs.total_hp_fraction(), 0.0, 1.0)))

        # --- battle ---
        f.append(1.0 if gs.in_battle() else 0.0)
        f.append(min(gs.battle_type(), 255) / 255.0)
        f.append(gs.enemy_species() / 255.0)
        f.append(min(gs.enemy_level(), 100) / 100.0)
        f.append(float(np.clip(gs.enemy_hp_fraction(), 0.0, 1.0)))

        # --- resources / key items ---
        f.append(min(gs.money(), 999999) / 999999.0)
        f.append(min(gs.pokeball_count(), 99) / 99.0)
        f.append(1.0 if gs.has_cut() else 0.0)
        f.append(1.0 if gs.has_surf() else 0.0)
        f.append(1.0 if gs.has_strength() else 0.0)
        f.append(min(len(gs.hm_ids_owned()), 5) / 5.0)
        f.append(min(len(gs.tm_ids_owned()), 50) / 50.0)
        from curriculum.ram_map import (ITEM_OAKS_PARCEL, ITEM_SS_TICKET,
                                        ITEM_TOWN_MAP, ITEM_BICYCLE)
        bag = gs.bag_item_ids()
        f.append(1.0 if ITEM_OAKS_PARCEL in bag else 0.0)
        f.append(1.0 if ITEM_SS_TICKET in bag else 0.0)
        f.append(1.0 if ITEM_TOWN_MAP in bag else 0.0)
        f.append(1.0 if ITEM_BICYCLE in bag else 0.0)
        f.append(min(gs.pokedex_owned_count(), 151) / 151.0)
        f.append(1.0 if gs.has_pokedex() else 0.0)

        # --- behavioural signals ---
        f.append(min(sig.heal_count, 50) / 50.0)
        f.append(min(sig.blackout_count, 20) / 20.0)
        f.append(1.0 if sig.pokecenter_used else 0.0)
        f.append(float(np.clip(sig.stuck_score, 0.0, 1.0)))
        f.append(float(np.clip(sig.loop_score, 0.0, 1.0)))
        f.append(float(np.clip(sig.inactivity_score, 0.0, 1.0)))
        f.append(float(np.clip(sig.wall_bump_score, 0.0, 1.0)))

        # --- high-level subgoal (subgoal-conditioned policy) ---
        f.append(float(np.clip(sig.subgoal_active, 0.0, 1.0)))
        f.append(min(max(sig.subgoal_target_map, 0), 255) / 255.0)
        f.append(float(np.clip(sig.subgoal_hops, 0.0, 1.0)))
        f.append(float(np.clip(sig.advice_active, 0.0, 1.0)))
        skill_oh = [0.0] * len(SKILL_VOCAB)
        if 0 <= sig.subgoal_skill_id < len(SKILL_VOCAB):
            skill_oh[sig.subgoal_skill_id] = 1.0
        f.extend(skill_oh)

        # --- recent action history (one-hot per slot) ---
        actions = list(sig.recent_actions[:self.history_len])
        actions += [-1] * (self.history_len - len(actions))
        for a in actions:
            onehot = [0.0] * self.num_actions
            if 0 <= a < self.num_actions:
                onehot[a] = 1.0
            f.extend(onehot)

        # --- recent map history ---
        maps = list(sig.recent_maps[:self.history_len])
        maps += [0] * (self.history_len - len(maps))
        f.extend([m / 255.0 for m in maps])

        # --- milestone completion bits ---
        bits = list(sig.milestone_bits[:self.num_milestones])
        bits += [0] * (self.num_milestones - len(bits))
        f.extend([float(b) for b in bits])

        arr = np.asarray(f, dtype=np.float32)
        assert arr.shape[0] == self.dim, (
            f"structured obs length {arr.shape[0]} != declared dim {self.dim}; "
            "schema and _compute_dim are out of sync"
        )
        # Guard against NaN/inf leaking into the policy.
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
