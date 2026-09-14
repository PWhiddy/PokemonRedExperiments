"""
intrinsic.py -- non-saturating intrinsic motivation (curiosity / novelty).

The static ``w_new_coord`` exploration bonus in :mod:`curriculum.rewards` is
*max-based*: each tile pays out exactly once, then the term goes to zero forever.
Once every reachable tile before a wall has been seen, exploration reward dries up
and PPO collapses onto whatever renewable reward is left (grinding levels) -- the
"saturate then exploit" failure the project is trying to fix.

Intrinsic motivation fixes this structurally: the bonus for a state is a function of
how *often that state has been visited*, so it

  * stays high at the frontier (rarely visited states keep paying),
  * decays smoothly as a region is mastered (no hard one-shot cliff),
  * automatically *revives* when the agent reaches a genuinely new region.

Two implementations share one :class:`NoveltyModule` interface:

* :class:`HashCountNovelty` -- pure-numpy pseudo-count bonus ``coef / sqrt(N(s))``
  over a discretized state hash. Cheap (no torch), per-env, the sensible default
  for many parallel ``SubprocVecEnv`` workers on CPU.
* :class:`RNDNovelty` -- Random Network Distillation (optional, needs torch). A
  fixed random target net vs. a trained predictor; prediction error is the bonus.
  Generalizes across similar states better than counts but costs a forward+backward
  per step. Enable only when a GPU is available.

Both expose ``bonus(gs) -> float`` and ``reset()``; the reward manager treats them
identically.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Protocol

from curriculum.ram_map import GameState


class NoveltyModule(Protocol):
    """Anything that scores how novel the current game state is (>= 0)."""

    def bonus(self, gs: GameState) -> float: ...
    def reset(self) -> None: ...
    def stats(self) -> Dict[str, float]: ...


# --------------------------------------------------------------------------- #
# Count-based pseudo-count novelty (default, no torch)                         #
# --------------------------------------------------------------------------- #
class HashCountNovelty:
    """Pseudo-count exploration bonus ``coef / (N(s) ** power)``.

    The state is discretized to a coarse, *progress-aware* key so that standing on
    the same tile but with a new badge / a chunk of new story flags counts as novel
    again. Visit counts are kept for the whole run (``lifelong``) and, optionally,
    just within the episode (``episodic``); the returned bonus is the larger decay
    of the two so both "never been here" and "haven't been here this episode" pay.

    Parameters
    ----------
    coef:
        Scale of the bonus at the first visit (``N=1`` gives ``coef``).
    power:
        Decay exponent. 0.5 => ``1/sqrt(N)`` (classic MBIE-EB), 1.0 => ``1/N``
        (faster decay, exploration calms down sooner).
    xy_bucket:
        Tile coordinates are floor-divided by this before hashing, so nearby tiles
        share a count (coarser => cheaper, less granular novelty).
    event_bucket:
        Story-flag count is bucketed by this so each chunk of new events refreshes
        novelty without making every single flag a brand-new state.
    episodic:
        Also track within-episode counts (reset every ``reset()``), encouraging the
        agent to cover fresh ground *each* episode rather than only globally-new ground.
    """

    def __init__(self, coef: float = 0.2, power: float = 0.5, xy_bucket: int = 2,
                 event_bucket: int = 4, episodic: bool = True):
        self.coef = float(coef)
        self.power = float(power)
        self.xy_bucket = max(int(xy_bucket), 1)
        self.event_bucket = max(int(event_bucket), 1)
        self.episodic = bool(episodic)
        self._lifelong: Dict[int, int] = {}
        self._episode: Dict[int, int] = {}
        self._last_bonus = 0.0
        self._unique_lifelong = 0

    # -- key -------------------------------------------------------------- #
    def _key(self, gs: GameState) -> int:
        x, y, m = gs.position()
        # progress tier so the *same* tile after real progress is novel again
        badges = gs.badge_count()
        ev = gs.event_flags_sum() // self.event_bucket
        battle = 1 if gs.in_battle() else 0
        return hash((m, x // self.xy_bucket, y // self.xy_bucket, badges, ev, battle))

    # -- api -------------------------------------------------------------- #
    def bonus(self, gs: GameState) -> float:
        k = self._key(gs)
        n_life = self._lifelong.get(k, 0) + 1
        self._lifelong[k] = n_life
        if n_life == 1:
            self._unique_lifelong += 1
        decay = self.coef / (n_life ** self.power)

        if self.episodic:
            n_ep = self._episode.get(k, 0) + 1
            self._episode[k] = n_ep
            decay = max(decay, self.coef / (n_ep ** self.power))

        self._last_bonus = decay
        return decay

    def reset(self) -> None:
        # lifelong counts persist across episodes on purpose (that is what makes
        # exploration globally non-saturating); only the episodic counts reset.
        self._episode.clear()

    def stats(self) -> Dict[str, float]:
        return {
            "intrinsic_last": self._last_bonus,
            "intrinsic_unique_states": float(self._unique_lifelong),
            "intrinsic_table_size": float(len(self._lifelong)),
        }


# --------------------------------------------------------------------------- #
# Random Network Distillation (optional, needs torch)                          #
# --------------------------------------------------------------------------- #
class RNDNovelty:
    """Random Network Distillation novelty over the structured observation vector.

    A fixed, randomly-initialized *target* network maps the (normalized) structured
    observation to an embedding; a *predictor* network is trained online to match it.
    States seen often are predicted well (low error => low bonus); novel states have
    high error => high bonus. Unlike counts, RND generalizes: a state similar to ones
    already mastered also gets a low bonus.

    This needs torch and does a forward+backward per step, so it is opt-in. The
    builder must pass an ``obs_fn`` that turns a :class:`GameState` into the same
    float vector the policy sees (the env provides one).
    """

    def __init__(self, obs_dim: int, obs_fn, coef: float = 0.5, lr: float = 1e-4,
                 emb_dim: int = 64, device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.coef = float(coef)
        self.obs_fn = obs_fn
        self.device = device

        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, emb_dim),
            ).to(device)

        self.target = mlp()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = mlp()
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        # running normalization of the error so coef stays meaningful as training
        # drives the average error down.
        self._err_mean = 1.0
        self._last_bonus = 0.0

    def bonus(self, gs: GameState) -> float:
        torch = self.torch
        vec = self.obs_fn(gs)
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        err = ((p - t) ** 2).mean()
        # train predictor toward target
        self.opt.zero_grad()
        err.backward()
        self.opt.step()
        e = float(err.detach())
        self._err_mean = 0.99 * self._err_mean + 0.01 * e
        bonus = self.coef * (e / max(self._err_mean, 1e-8))
        self._last_bonus = bonus
        return bonus

    def reset(self) -> None:
        pass  # RND carries its learned predictor across episodes by design

    def stats(self) -> Dict[str, float]:
        return {"intrinsic_last": self._last_bonus, "intrinsic_err_mean": self._err_mean}


# --------------------------------------------------------------------------- #
# Factory                                                                       #
# --------------------------------------------------------------------------- #
def build_novelty(kind: str, *, coef: float, obs_dim: int = 0, obs_fn=None,
                  **kw) -> Optional[NoveltyModule]:
    """Construct a novelty module by name. ``kind`` in {"none","count","rnd"}."""
    kind = (kind or "none").lower()
    if kind in ("none", "off", ""):
        return None
    if kind == "count":
        return HashCountNovelty(coef=coef,
                                power=kw.get("power", 0.5),
                                xy_bucket=kw.get("xy_bucket", 2),
                                event_bucket=kw.get("event_bucket", 4),
                                episodic=kw.get("episodic", True))
    if kind == "rnd":
        if obs_fn is None or obs_dim <= 0:
            raise ValueError("RND novelty needs obs_dim>0 and an obs_fn")
        return RNDNovelty(obs_dim=obs_dim, obs_fn=obs_fn, coef=coef,
                          lr=kw.get("lr", 1e-4), emb_dim=kw.get("emb_dim", 64),
                          device=kw.get("device", "cpu"))
    raise ValueError(f"unknown novelty kind '{kind}' (use none|count|rnd)")
