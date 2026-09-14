"""
milestones.py -- the curriculum graph for completing Pokemon Red.

The full game is decomposed into an **ordered list of milestones**, each a small,
detectable, learnable objective ("Get starter", "Beat Brock", "Reach Cerulean",
"Beat Champion", ...).

Each :class:`Milestone` defines:
  - ``detector(gs, ctx)``     : returns True once the objective is achieved.
  - ``reward``                : one-time reward granted the first time it completes.
  - ``timeout_steps``         : episode/stage step budget when this is the target.
  - ``allowed_start_states``  : keys of saved emulator snapshots a stage may start
                                from (typically the previous milestone's snapshots).
  - ``next_key``              : the milestone that follows it (linear by default).
  - ``save_state``            : whether to snapshot successful emulator states here
                                so later stages can warm-start from them.

:class:`MilestoneManager` tracks which milestones are completed during an episode,
latches completion (a milestone, once done, stays done), maintains a small context
(visited maps, max badges) used by detectors, and exposes the "frontier" milestone
for curriculum staging.

Detectors rely only on :class:`curriculum.ram_map.GameState` plus the lightweight
:class:`MilestoneContext`, so they are cheap and side-effect free. Event-flag
addresses/indices come from this repo's ``events.json`` (MSB-first indices); map ids
come from ``map_data.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from curriculum.ram_map import GameState


@dataclass
class MilestoneContext:
    """Per-episode running context made available to milestone detectors."""
    visited_maps: Set[int] = field(default_factory=set)
    max_badges: int = 0
    step: int = 0

    def update(self, gs: GameState) -> None:
        self.visited_maps.add(gs.map_id())
        self.max_badges = max(self.max_badges, gs.badge_count())


# Detector signature: (GameState, MilestoneContext) -> bool
Detector = Callable[[GameState, MilestoneContext], bool]


@dataclass
class Milestone:
    key: str
    name: str
    detector: Detector
    reward: float = 50.0
    timeout_steps: int = 2048 * 16
    allowed_start_states: List[str] = field(default_factory=list)
    next_key: Optional[str] = None
    save_state: bool = True

    def is_complete(self, gs: GameState, ctx: MilestoneContext) -> bool:
        try:
            return bool(self.detector(gs, ctx))
        except Exception:
            # A detector referencing an address/flag that misbehaves should never
            # crash training -- treat as "not complete" and keep going.
            return False


# --------------------------------------------------------------------------- #
# Detector helpers                                                             #
# --------------------------------------------------------------------------- #
def _event(addr: int, idx: int) -> Detector:
    """Milestone completes when a named event flag (events.json key) is set."""
    return lambda gs, ctx: gs.event_flag(addr, idx)


def _reached_map(*map_ids: int) -> Detector:
    """Completes the first time any of ``map_ids`` has been entered this episode."""
    targets = set(map_ids)
    return lambda gs, ctx: bool(targets & ctx.visited_maps) or gs.map_id() in targets


def _badges_at_least(n: int) -> Detector:
    return lambda gs, ctx: gs.badge_count() >= n or ctx.max_badges >= n


def _any(*detectors: Detector) -> Detector:
    return lambda gs, ctx: any(d(gs, ctx) for d in detectors)


def _party_at_least(n: int) -> Detector:
    return lambda gs, ctx: gs.party_count() >= n


# --------------------------------------------------------------------------- #
# The default Pokemon Red curriculum                                          #
# --------------------------------------------------------------------------- #
# Map ids are from map_data.json; event (addr, idx) pairs from events.json.
# Where a precise "completed the dungeon/building" event was not obviously
# available, we use "reached the deepest known map" as a robust proxy and flag it
# in the comment. Detectors are intentionally permissive (event OR badge OR map)
# so they fire reliably even if one signal lags by a frame.

def build_default_milestones() -> List[Milestone]:
    M = Milestone
    milestones = [
        M("start_game", "Start Game",
          _event(0xD747, 0),  # Followed Oak Into Lab (real event, not a no-op)
          reward=10.0, timeout_steps=2048 * 4, save_state=True),

        M("get_starter", "Get Starter Pokemon",
          _any(_event(0xD74B, 2), _party_at_least(1)),  # Got Starter
          reward=60.0),

        M("first_rival_battle", "Complete First Rival Battle",
          _event(0xD74B, 3),                            # Battled Rival In Oaks Lab
          reward=60.0),

        M("deliver_parcel", "Deliver Oak's Parcel",
          _event(0xD74E, 0),                            # Oak Got Parcel
          reward=60.0),

        M("get_pokedex", "Get Pokedex",
          _any(_event(0xD74B, 5), lambda gs, c: gs.has_pokedex()),  # Got Pokedex
          reward=80.0),

        M("reach_viridian_forest", "Reach Viridian Forest",
          _reached_map(51), reward=40.0),

        M("exit_viridian_forest", "Exit Viridian Forest",
          _reached_map(47, 2), reward=40.0),            # Route 2 Gate Pewter / Pewter

        M("reach_pewter", "Reach Pewter City",
          _reached_map(2), reward=50.0),

        M("beat_brock", "Beat Brock (Boulder Badge)",
          _any(_event(0xD755, 7), _badges_at_least(1)),  # Beat Brock
          reward=150.0),

        M("reach_mt_moon", "Reach Mt. Moon",
          _reached_map(59, 60, 61), reward=40.0),

        M("exit_mt_moon", "Exit Mt. Moon",
          _reached_map(15, 3), reward=50.0),            # Route 4 / Cerulean

        M("reach_cerulean", "Reach Cerulean City",
          _reached_map(3), reward=50.0),

        M("beat_misty", "Beat Misty (Cascade Badge)",
          _any(_event(0xD75E, 7), _badges_at_least(2)),  # Beat Misty
          reward=150.0),

        M("help_bill", "Help Bill / Get SS Ticket",
          _event(0xD7F2, 4), reward=80.0),              # Got Ss Ticket

        M("reach_vermilion", "Reach Vermilion City",
          _reached_map(5), reward=50.0),

        M("get_cut", "Get HM01 Cut",
          _any(_event(0xD803, 0), lambda gs, c: gs.has_cut()),  # Got Hm01
          reward=80.0),

        M("beat_lt_surge", "Beat Lt. Surge (Thunder Badge)",
          _any(_event(0xD773, 7), _badges_at_least(3)),  # Beat Lt Surge
          reward=150.0),

        M("reach_rock_tunnel", "Reach Rock Tunnel",
          _reached_map(82, 232), reward=40.0),

        M("reach_lavender", "Reach Lavender Town",
          _reached_map(4), reward=50.0),

        M("reach_celadon", "Reach Celadon City",
          _reached_map(6), reward=50.0),

        M("beat_erika", "Beat Erika (Rainbow Badge)",
          _any(_event(0xD77C, 1), _badges_at_least(4)),  # Beat Erika
          reward=150.0),

        M("complete_rocket_hideout", "Complete Rocket Hideout",
          # Proxy: reached the deepest hideout floor (B4F, map 202).
          _reached_map(202), reward=120.0),

        M("complete_pokemon_tower", "Complete Pokemon Tower",
          _any(_event(0xD7E0, 7), _event(0xD76C, 0)),    # Rescued Mr Fuji / Got Poke Flute
          reward=120.0),

        M("complete_silph_co", "Complete Silph Co.",
          # Proxy: reached Silph Co 11F (map 235, Giovanni's floor).
          _reached_map(235), reward=160.0),

        M("beat_koga", "Beat Koga (Soul Badge)",
          _any(_event(0xD792, 1), _badges_at_least(5)),  # Beat Koga
          reward=150.0),

        M("beat_sabrina", "Beat Sabrina (Marsh Badge)",
          _any(_event(0xD7B3, 1), _badges_at_least(6)),  # Beat Sabrina
          reward=150.0),

        M("reach_cinnabar", "Reach Cinnabar Island",
          _reached_map(8), reward=50.0),

        M("beat_blaine", "Beat Blaine (Volcano Badge)",
          _any(_event(0xD79A, 1), _badges_at_least(7)),  # Beat Blaine
          reward=150.0),

        M("beat_giovanni", "Beat Giovanni (Earth Badge)",
          _any(_event(0xD751, 1), _badges_at_least(8)),  # Beat Viridian Gym Giovanni
          reward=200.0),

        M("reach_victory_road", "Reach Victory Road",
          _reached_map(108, 194, 198), reward=60.0),

        M("exit_victory_road", "Exit Victory Road",
          _reached_map(9, 174), reward=80.0),           # Indigo Plateau / Lobby

        M("beat_elite_four", "Beat Elite Four",
          _reached_map(120), reward=300.0),             # Reached Champion's Room

        M("beat_champion", "Beat Champion",
          _event(0xD747, 3), reward=1000.0,             # Hall Of Fame Dex Rating
          save_state=True),
    ]

    # Wire up linear ``next_key`` and default ``allowed_start_states`` (each stage
    # may warm-start from the snapshots saved by the previous milestone).
    for i, m in enumerate(milestones):
        if i + 1 < len(milestones):
            m.next_key = milestones[i + 1].key
        if i > 0 and not m.allowed_start_states:
            m.allowed_start_states = [milestones[i - 1].key]
    return milestones


class MilestoneManager:
    """Tracks milestone completion across an episode and curriculum staging.

    Parameters
    ----------
    milestones:
        Ordered list of :class:`Milestone`. Defaults to ``build_default_milestones``.
    target_key:
        Optional milestone key designating the current curriculum *target*. When set,
        :meth:`reached_target` reports whether it has been completed this episode and
        :attr:`target` exposes the milestone (useful for stage-specific timeouts and
        for deciding when to snapshot a successful state).
    """

    def __init__(self, milestones: Optional[List[Milestone]] = None,
                 target_key: Optional[str] = None):
        self.milestones = milestones if milestones is not None else build_default_milestones()
        self.by_key: Dict[str, Milestone] = {m.key: m for m in self.milestones}
        self.order: Dict[str, int] = {m.key: i for i, m in enumerate(self.milestones)}
        self.target_key = target_key
        self.ctx = MilestoneContext()
        self.completed: Set[str] = set()
        self.newly_completed: List[str] = []  # filled each step()

    # ----- lifecycle ------------------------------------------------------- #
    def reset(self) -> None:
        self.ctx = MilestoneContext()
        self.completed = set()
        self.newly_completed = []

    def update(self, gs: GameState) -> List[str]:
        """Advance one step. Returns the keys completed *this* step (for rewards)."""
        self.ctx.step += 1
        self.ctx.update(gs)
        self.newly_completed = []
        for m in self.milestones:
            if m.key in self.completed:
                continue
            if m.is_complete(gs, self.ctx):
                self.completed.add(m.key)
                self.newly_completed.append(m.key)
        return self.newly_completed

    # ----- queries --------------------------------------------------------- #
    @property
    def target(self) -> Optional[Milestone]:
        return self.by_key.get(self.target_key) if self.target_key else None

    def reached_target(self) -> bool:
        return self.target_key is not None and self.target_key in self.completed

    def frontier_index(self) -> int:
        """Index of the first not-yet-completed milestone (the active frontier)."""
        for i, m in enumerate(self.milestones):
            if m.key not in self.completed:
                return i
        return len(self.milestones)

    def frontier(self) -> Optional[Milestone]:
        idx = self.frontier_index()
        return self.milestones[idx] if idx < len(self.milestones) else None

    def completion_count(self) -> int:
        return len(self.completed)

    def progress_fraction(self) -> float:
        return len(self.completed) / max(len(self.milestones), 1)
