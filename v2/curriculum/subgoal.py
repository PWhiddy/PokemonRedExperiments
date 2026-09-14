"""
subgoal.py -- subgoals and potential-based reward shaping toward them.

The milestones are sparse: between "Reach Cerulean" and "Beat Misty" there can be
thousands of steps with no reward signal, which is exactly where a flat policy stalls.
A *subgoal* is a short-horizon target (usually "get to map X", optionally "use skill
Y") proposed by the high-level planner (rules or an LLM). This module turns a subgoal
into a **dense, safe** learning signal via potential-based reward shaping (Ng et al.,
1999):

    F(s, s') = gamma * Phi(s') - Phi(s)

Because the shaping is the difference of a potential, it provably does **not** change
the optimal policy -- it only densifies the gradient, so it cannot introduce a new
exploit. The potential here is the negative *map-hop distance* to the target map,
measured on a graph the env learns online from observed map transitions (no static
map atlas needed). Reaching the target map also pays a one-time bonus.

If the target map is unknown or unreachable on the current graph, the shaper degrades
to "0 shaping, pay the reach bonus when/if the map is entered", so a bad LLM
suggestion can never destabilize training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# Canonical skill vocabulary shared by the planner, the observation (skill one-hot)
# and the HRL manager. Keep the order stable -- the index is fed to the policy.
# The trailing field-move skills (cut/surf/strength) are the ones the skill bridge
# can execute via deterministic macros when the planner advises them.
SKILL_VOCAB: List[str] = [
    "navigation", "battle", "healing", "story", "grinding", "shopping", "unstuck",
    "cut", "surf", "strength", "flute",
]
# HM field moves whose menu button sequence is unlearnable by PPO.
FIELD_MOVE_SKILLS = frozenset({"cut", "surf", "strength"})
# All skills the advice controller can EXECUTE via a deterministic macro (field moves
# + key-item uses like the Poke Flute on a Snorlax).
MACRO_SKILLS = FIELD_MOVE_SKILLS | frozenset({"flute"})
SKILL_INDEX: Dict[str, int] = {s: i for i, s in enumerate(SKILL_VOCAB)}


def skill_id(name: str) -> int:
    """Index of ``name`` in :data:`SKILL_VOCAB`, or -1 if unknown."""
    return SKILL_INDEX.get(name, -1)


@dataclass
class Subgoal:
    """A short-horizon target proposed by the planner."""
    target_map: Optional[int] = None        # map id to head for (None = no nav target)
    text: str = ""                          # human-readable goal ("Enter Pewter Gym")
    skill: str = "navigation"               # suggested specialist skill / option
    source: str = "rule"                    # "rule" | "llm" | "none"
    reasoning: str = ""                     # planner's short rationale (logged)

    def is_empty(self) -> bool:
        return self.target_map is None and not self.text


class MapGraph:
    """Online, undirected map-adjacency graph learned from (prev_map -> map) edges.

    BFS over it gives the hop distance between any two visited maps -- a cheap,
    monotone potential for navigation shaping that needs no precomputed atlas.
    """

    def __init__(self):
        self.adj: Dict[int, Set[int]] = {}

    def observe(self, prev_map: int, cur_map: int) -> None:
        if prev_map == cur_map:
            return
        self.adj.setdefault(prev_map, set()).add(cur_map)
        self.adj.setdefault(cur_map, set()).add(prev_map)

    def hops(self, src: int, dst: int, cap: int = 30) -> Optional[int]:
        """Shortest hop count ``src -> dst``; None if unconnected on the known graph."""
        if src == dst:
            return 0
        if src not in self.adj or dst not in self.adj:
            return None
        seen = {src}
        q: deque = deque([(src, 0)])
        while q:
            node, d = q.popleft()
            if d >= cap:
                continue
            for nb in self.adj.get(node, ()):  # noqa: B007
                if nb == dst:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None


class SubgoalShaper:
    """Potential-based shaping toward the active subgoal's target map."""

    def __init__(self, gamma: float = 0.998, weight: float = 1.0,
                 reach_bonus: float = 8.0, dist_cap: int = 20):
        self.gamma = gamma
        self.weight = weight              # scales the per-step shaping term
        self.reach_bonus = reach_bonus    # one-time payout for entering the target map
        self.dist_cap = dist_cap
        self.graph = MapGraph()
        self.reset()

    def reset(self) -> None:
        self._last_potential: Optional[float] = None
        self._last_target: Optional[int] = None
        self._reached_targets: Set[int] = set()

    def _potential(self, cur_map: int, target_map: Optional[int]) -> float:
        if target_map is None:
            return 0.0
        h = self.graph.hops(cur_map, target_map, cap=self.dist_cap)
        if h is None:
            # unknown route: a flat, mild "not there yet" potential so entering the
            # target (potential -> 0) still yields a positive shaping bump.
            return -float(self.dist_cap)
        return -float(min(h, self.dist_cap))

    def step(self, prev_map: int, cur_map: int, subgoal: Optional[Subgoal]) -> float:
        """Return the shaping reward for this transition (0.0 if no active subgoal)."""
        self.graph.observe(prev_map, cur_map)
        target = subgoal.target_map if subgoal is not None else None

        # reset the potential baseline when the target changes (avoids a spurious
        # one-step jump being credited/charged when the planner switches goals).
        if target != self._last_target:
            self._last_potential = self._potential(prev_map, target)
            self._last_target = target

        reward = 0.0
        phi = self._potential(cur_map, target)
        if self._last_potential is not None:
            reward += self.weight * (self.gamma * phi - self._last_potential)
        self._last_potential = phi

        # one-time reach bonus (protected progress-like signal toward the subgoal)
        if target is not None and cur_map == target and target not in self._reached_targets:
            self._reached_targets.add(target)
            reward += self.reach_bonus

        return reward

    def reached(self, target_map: Optional[int]) -> bool:
        return target_map is not None and target_map in self._reached_targets
