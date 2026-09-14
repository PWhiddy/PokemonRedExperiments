"""
llm_planner.py -- the "think before you learn" high-level planner.

When the RL agent stalls (loop/stuck/inactivity signals fire and milestone progress
flatlines), a flat policy has no way to reason *"I'm stuck because I need to leave
Cerulean heading east to find Bill"*. This module adds that reasoning layer: a planner
inspects the structured game state and proposes the next :class:`~curriculum.subgoal.Subgoal`
(a target map + a suggested skill + a short rationale). The env then shapes reward
toward that subgoal (see :mod:`curriculum.subgoal`) so the policy gets a dense gradient
out of the stall.

Three planners share one ``propose(ctx) -> Subgoal`` interface:

* :class:`RuleBasedPlanner` -- zero-dependency lookup from the current *frontier*
  milestone to a curated target map / skill (the bundled Kanto route knowledge below).
  Always available; used as the instant fallback.
* :class:`OllamaPlanner` -- queries a local Ollama model (e.g. ``gemma4:e2b``) with the
  game state + the route knowledge as grounding, and parses a JSON subgoal. This is the
  "agent figures out what to do" layer, kept fully local (no API key, no cloud).
* :class:`AsyncPlanner` -- wraps any slow planner so the env **never blocks**: it serves
  the last cached subgoal immediately and refreshes it on a background thread, keyed by
  (frontier milestone, current map). LLM latency is hidden behind the rule-based default.

The bundled route knowledge plays the role of "search the web for a walkthrough": it is
curated Pokemon Red progression facts the small local model would otherwise hallucinate.
Swap in a real web/RAG lookup by replacing ``ROUTE_KNOWLEDGE`` / ``knowledge_for``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

from curriculum.subgoal import Subgoal


# --------------------------------------------------------------------------- #
# Bundled Kanto route knowledge (stand-in for a web/walkthrough lookup)        #
# Maps the current *frontier* milestone key -> the next concrete target.       #
# target_map ids come from this repo's map_data.json (city ids 0-11, etc).     #
# --------------------------------------------------------------------------- #
ROUTE_KNOWLEDGE: Dict[str, Dict] = {
    "start_game":            {"target_map": 40, "skill": "story",      "text": "Leave the bedroom and follow Oak into his lab."},
    "get_starter":           {"target_map": 40, "skill": "story",      "text": "Pick a starter Pokemon from Oak's table."},
    "first_rival_battle":    {"target_map": 40, "skill": "battle",     "text": "Defeat your rival in Oak's lab."},
    "deliver_parcel":        {"target_map": 1,  "skill": "navigation", "text": "Go north to Viridian, buy/grab Oak's Parcel, return it to Oak."},
    "get_pokedex":           {"target_map": 40, "skill": "story",      "text": "Return to Oak's lab to receive the Pokedex."},
    "reach_viridian_forest": {"target_map": 51, "skill": "navigation", "text": "Head north through Viridian and Route 2 into Viridian Forest."},
    "exit_viridian_forest":  {"target_map": 2,  "skill": "navigation", "text": "Cross Viridian Forest north to Pewter City."},
    "reach_pewter":          {"target_map": 2,  "skill": "navigation", "text": "Reach Pewter City."},
    "beat_brock":            {"target_map": 2,  "skill": "battle",     "text": "Enter Pewter Gym and beat Brock for the Boulder Badge (use Water/Grass)."},
    "reach_mt_moon":         {"target_map": 59, "skill": "navigation", "text": "Go east on Route 3 to Mt. Moon."},
    "exit_mt_moon":          {"target_map": 3,  "skill": "navigation", "text": "Cross Mt. Moon to Route 4 and Cerulean City."},
    "reach_cerulean":        {"target_map": 3,  "skill": "navigation", "text": "Reach Cerulean City."},
    "beat_misty":            {"target_map": 3,  "skill": "battle",     "text": "Enter Cerulean Gym and beat Misty for the Cascade Badge (use Electric/Grass)."},
    "help_bill":             {"target_map": 3,  "skill": "navigation", "text": "Go north on Route 24/25 to Bill's house, then get the S.S. Ticket."},
    "reach_vermilion":       {"target_map": 5,  "skill": "navigation", "text": "Head south through Saffron's gates to Vermilion City."},
    "get_cut":               {"target_map": 5,  "skill": "story",      "text": "Board the S.S. Anne and get HM01 Cut from the captain."},
    "beat_lt_surge":         {"target_map": 5,  "skill": "cut",        "text": "Use Cut on the tree blocking Vermilion Gym, then beat Lt. Surge (use Ground)."},
    "reach_rock_tunnel":     {"target_map": 82, "skill": "navigation", "text": "Go east via Route 9 to Rock Tunnel (needs Flash ideally)."},
    "reach_lavender":        {"target_map": 4,  "skill": "navigation", "text": "Pass through Rock Tunnel to Lavender Town."},
    "reach_celadon":         {"target_map": 6,  "skill": "navigation", "text": "Go west from Saffron/Route 7 to Celadon City."},
    "beat_erika":            {"target_map": 6,  "skill": "battle",     "text": "Enter Celadon Gym, beat Erika for the Rainbow Badge (use Fire/Flying)."},
    "complete_rocket_hideout": {"target_map": 202, "skill": "battle",  "text": "Clear the Rocket Hideout under the Celadon Game Corner, beat Giovanni."},
    "complete_pokemon_tower":  {"target_map": 4,   "skill": "story",   "text": "Get the Silph Scope, climb Pokemon Tower, save Mr. Fuji, get the Poke Flute."},
    "complete_silph_co":     {"target_map": 10, "skill": "battle",     "text": "Go to Saffron, clear Silph Co. up to the 11F and beat Giovanni."},
    "beat_koga":             {"target_map": 7,  "skill": "flute",      "text": "Wake the Snorlax blocking the route with the Poke Flute, reach Fuchsia, then beat Koga for the Soul Badge (use Psychic)."},
    "beat_sabrina":          {"target_map": 10, "skill": "battle",     "text": "Beat Sabrina in Saffron Gym for the Marsh Badge (use Bug/Dark)."},
    "reach_cinnabar":        {"target_map": 8,  "skill": "surf",       "text": "Surf south from Pallet/Fuchsia across the water to Cinnabar Island."},
    "beat_blaine":           {"target_map": 8,  "skill": "battle",     "text": "Get the gym key from the Mansion, beat Blaine (use Water/Ground)."},
    "beat_giovanni":         {"target_map": 1,  "skill": "battle",     "text": "Return to Viridian Gym and beat Giovanni for the Earth Badge."},
    "reach_victory_road":    {"target_map": 108,"skill": "strength",   "text": "Go north past Route 22/23 to Victory Road; use Strength to move boulders (Surf also needed)."},
    "exit_victory_road":     {"target_map": 9,  "skill": "navigation", "text": "Climb Victory Road to Indigo Plateau."},
    "beat_elite_four":       {"target_map": 120,"skill": "battle",     "text": "Beat the Elite Four in sequence: Lorelei, Bruno, Agatha, Lance."},
    "beat_champion":         {"target_map": 120,"skill": "battle",     "text": "Beat your rival, the Champion, to enter the Hall of Fame."},
}


def knowledge_for(frontier_key: Optional[str]) -> Optional[Dict]:
    return ROUTE_KNOWLEDGE.get(frontier_key) if frontier_key else None


# --------------------------------------------------------------------------- #
# Human-readable map names (cities + a few landmarks) for the CLI panel        #
# --------------------------------------------------------------------------- #
MAP_NAMES: Dict[int, str] = {
    0: "Pallet Town", 1: "Viridian City", 2: "Pewter City", 3: "Cerulean City",
    4: "Lavender Town", 5: "Vermilion City", 6: "Celadon City", 7: "Fuchsia City",
    8: "Cinnabar Island", 9: "Indigo Plateau", 10: "Saffron City",
    40: "Oak's Lab", 51: "Viridian Forest", 59: "Mt. Moon", 82: "Rock Tunnel",
    108: "Victory Road", 120: "Champion's Room", 202: "Rocket Hideout B4F",
    235: "Silph Co 11F",
}


def map_name(map_id: int) -> str:
    return MAP_NAMES.get(map_id, f"map {map_id}")


def render_llm_decision(ctx: "PlannerContext", sub: Subgoal, latency: float,
                        count: int = 0, model: str = "") -> None:
    """Print a boxed CLI panel announcing a genuine LLM planner decision.

    Fires ONLY when the local LLM actually produced the subgoal (not on cached or
    rule-based decisions), so watching the console tells you exactly when the agent
    "thought" with the model versus ran on a default.
    """
    # ASCII-only box + arrows so it never crashes on a Windows cp1252 console;
    # ANSI colour codes are plain ASCII bytes, so they are safe to encode.
    tgt = f"{sub.target_map} ({map_name(sub.target_map)})" if sub.target_map is not None else "-"
    tag = f"#{count} " if count else ""
    title = f" LLM DECISION {tag}({model or 'local'}, {latency:.1f}s) "
    width = 72
    bar = title.center(width, "=")
    C, G, R = "\033[96m", "\033[92m", "\033[0m"
    lines = [
        f"{C}+{bar}+{R}",
        f"  when : {map_name(ctx.map_id)} ({ctx.map_id}) @({ctx.x},{ctx.y}) "
        f"| badges {ctx.badge_count}/8 | objective: {ctx.frontier_name or ctx.frontier_key}",
        f"  why  : stuck={ctx.stuck_score:.2f} loop={ctx.loop_score:.2f} "
        f"inactivity={ctx.inactivity_score:.2f}  (high => agent was stuck)",
        f"  {G}-> skill{R} : {sub.skill:10s}  {G}-> target{R} : {tgt}",
        f"  goal : {sub.text}",
        f"  model: {sub.reasoning}",
        f"{C}+{'=' * width}+{R}",
    ]
    try:
        print("\n".join(lines), flush=True)
    except UnicodeEncodeError:
        # last-resort: strip anything non-ASCII
        print("\n".join(l.encode('ascii', 'replace').decode('ascii') for l in lines),
              flush=True)


# --------------------------------------------------------------------------- #
# Planner context + interface                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class PlannerContext:
    """Everything a planner needs to reason about the next subgoal."""
    map_id: int
    x: int
    y: int
    badge_count: int
    party_levels: List[int] = field(default_factory=list)
    party_hp_frac: float = 1.0
    in_battle: bool = False
    frontier_key: Optional[str] = None
    frontier_name: str = ""
    recent_maps: List[int] = field(default_factory=list)
    stuck_score: float = 0.0
    loop_score: float = 0.0
    inactivity_score: float = 0.0

    def cache_key(self):
        return (self.frontier_key, self.map_id)


class Planner(Protocol):
    def propose(self, ctx: PlannerContext) -> Subgoal: ...


# --------------------------------------------------------------------------- #
# Rule-based planner (instant, always available)                               #
# --------------------------------------------------------------------------- #
class RuleBasedPlanner:
    """Look the frontier milestone up in the bundled route knowledge."""

    def propose(self, ctx: PlannerContext) -> Subgoal:
        k = knowledge_for(ctx.frontier_key)
        if not k:
            return Subgoal(source="rule", text="Explore for new areas and events.",
                           skill="navigation")
        return Subgoal(target_map=k.get("target_map"), text=k.get("text", ""),
                       skill=k.get("skill", "navigation"), source="rule",
                       reasoning=f"frontier={ctx.frontier_key}")


# --------------------------------------------------------------------------- #
# Ollama planner (local LLM reasoning)                                         #
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are the strategic planner for a reinforcement-learning agent playing "
    "Pokemon Red. Given the current game state and a verified hint, decide the single "
    "best next sub-goal. Think like a skilled human speedrunner: concrete, one step at "
    "a time. Respond with ONLY a compact JSON object and nothing else, of the form: "
    '{"target_map": <int or null>, '
    '"skill": "navigation|battle|healing|story|grinding|cut|surf|strength|flute", '
    '"goal": "<short imperative>", "why": "<one sentence>"}. '
    'Use skill "cut"/"surf"/"strength" when an HM field move is the thing blocking '
    'progress (small tree => cut, water => surf, boulder => strength), and "flute" when '
    'a sleeping Snorlax blocks the path and you hold the Poke Flute.'
)


class OllamaPlanner:
    """Query a local Ollama model for the next subgoal (grounded by route knowledge)."""

    def __init__(self, model: str = "gemma4:e2b",
                 host: str = "http://localhost:11434", timeout: float = 20.0,
                 on_decision: Optional[Callable] = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # called with (ctx, subgoal, latency_seconds) on every genuine LLM decision
        self.on_decision = on_decision
        self._rule = RuleBasedPlanner()

    def _prompt(self, ctx: PlannerContext) -> str:
        hint = knowledge_for(ctx.frontier_key) or {}
        lines = [
            f"Current map id: {ctx.map_id}  position: ({ctx.x},{ctx.y})",
            f"Badges: {ctx.badge_count}/8   Party levels: {ctx.party_levels}   "
            f"Party HP: {ctx.party_hp_frac:.2f}   In battle: {ctx.in_battle}",
            f"Current objective (frontier milestone): {ctx.frontier_name or ctx.frontier_key}",
            f"Recently visited maps: {ctx.recent_maps}",
            f"Stuck={ctx.stuck_score:.2f} loop={ctx.loop_score:.2f} "
            f"inactivity={ctx.inactivity_score:.2f} (high => agent is stuck).",
        ]
        if hint:
            lines.append(
                f"VERIFIED HINT: to progress, head toward map {hint.get('target_map')} "
                f"and {hint.get('text')} (suggested skill: {hint.get('skill')}).")
        lines.append("Return the JSON subgoal now.")
        return "\n".join(lines)

    def propose(self, ctx: PlannerContext) -> Subgoal:
        import requests  # local import so the module imports without requests
        payload = {
            "model": self.model,
            "system": _SYSTEM,
            "prompt": self._prompt(ctx),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 160},
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.host}/api/generate", json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            data = json.loads(text)
        except Exception as e:
            # any failure (server down, bad JSON, timeout) -> safe rule-based fallback
            sub = self._rule.propose(ctx)
            sub.reasoning = f"ollama_failed:{type(e).__name__}; {sub.reasoning}"
            return sub
        latency = time.time() - t0

        tgt = data.get("target_map")
        try:
            tgt = int(tgt) if tgt is not None else None
        except (TypeError, ValueError):
            tgt = None
        # ground the LLM: if it gave no/implausible map, borrow the verified hint's map
        hint = knowledge_for(ctx.frontier_key) or {}
        if tgt is None:
            tgt = hint.get("target_map")
        sub = Subgoal(
            target_map=tgt,
            text=str(data.get("goal", hint.get("text", "")))[:200],
            skill=str(data.get("skill", hint.get("skill", "navigation"))),
            source="llm",
            reasoning=str(data.get("why", ""))[:200],
        )
        if self.on_decision is not None:
            try:
                self.on_decision(ctx, sub, latency)
            except Exception:
                pass  # a logging hook must never break planning
        return sub


# --------------------------------------------------------------------------- #
# Async wrapper: never block the env on LLM latency                            #
# --------------------------------------------------------------------------- #
class AsyncPlanner:
    """Serve a cached subgoal instantly; refresh slow planners off-thread.

    ``propose`` returns immediately with the best subgoal known for the current
    (frontier, map) key -- the rule-based default until the LLM answers. When the key
    is new (or ``force`` is set) and no refresh is in flight, it launches a background
    thread to query the wrapped (slow) planner and caches the result for next time.
    """

    def __init__(self, slow: Planner, fallback: Optional[Planner] = None,
                 max_inflight: int = 1):
        self.slow = slow
        self.fallback = fallback or RuleBasedPlanner()
        self._cache: Dict[tuple, Subgoal] = {}
        self._inflight: set = set()
        self._lock = threading.Lock()
        self._max_inflight = max_inflight

    def _worker(self, key, ctx: PlannerContext) -> None:
        try:
            sub = self.slow.propose(ctx)
        except Exception:
            sub = self.fallback.propose(ctx)
        with self._lock:
            self._cache[key] = sub
            self._inflight.discard(key)

    def propose(self, ctx: PlannerContext, force: bool = False) -> Subgoal:
        key = ctx.cache_key()
        with self._lock:
            cached = self._cache.get(key)
            can_launch = (key not in self._inflight
                          and len(self._inflight) < self._max_inflight)
            need = force or cached is None
            if need and can_launch:
                self._inflight.add(key)
                launch = True
            else:
                launch = False
        if launch:
            t = threading.Thread(target=self._worker, args=(key, ctx), daemon=True)
            t.start()
        if cached is not None:
            return cached
        # nothing cached yet -> instant rule-based answer this step
        return self.fallback.propose(ctx)


# --------------------------------------------------------------------------- #
# Factory                                                                       #
# --------------------------------------------------------------------------- #
def build_planner(kind: str, *, model: str = "gemma4:e2b",
                  host: str = "http://localhost:11434",
                  timeout: float = 20.0, asy: bool = True,
                  on_decision: Optional[Callable] = None) -> Optional[Planner]:
    """Construct a planner. ``kind`` in {"none","rule","ollama"}.

    ``on_decision(ctx, subgoal, latency)`` is invoked on every genuine LLM decision
    (used by the env to print the CLI panel when ``planner_verbose`` is on).
    """
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "rule":
        return RuleBasedPlanner()
    if kind == "ollama":
        slow = OllamaPlanner(model=model, host=host, timeout=timeout,
                             on_decision=on_decision)
        return AsyncPlanner(slow) if asy else slow
    raise ValueError(f"unknown planner kind '{kind}' (use none|rule|ollama)")
