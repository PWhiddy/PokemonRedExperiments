"""
state_store.py -- persist successful emulator snapshots per milestone.

Curriculum learning needs to warm-start later stages from states where earlier
milestones are already satisfied. PyBoy serializes its full machine state via
``pyboy.save_state(file)`` / ``pyboy.load_state(file)``; this module organizes those
snapshots on disk by milestone key and offers simple selection (random/latest/best).

Layout::

    <root>/
      get_starter/  get_starter_r60.0_<uid>.state ...
      beat_brock/   beat_brock_r150.0_<uid>.state ...

Snapshot reward is encoded in the filename (``_r<reward>_``) so reads are a plain
directory scan with no shared index file. That makes the store safe to use from many
``SubprocVecEnv`` worker processes writing concurrently -- filenames are unique
(uuid) and selection never depends on a mutable shared index.

Snapshots are emulator save-states (.state), never ROMs -- nothing here downloads,
contains, or distributes copyrighted game data.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

_REWARD_RE = re.compile(r"_r(-?\d+(?:\.\d+)?)_")


class StateStore:
    def __init__(self, root: str | Path, max_per_milestone: int = 50):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_per_milestone = max_per_milestone

    # ------------------------------------------------------------------ #
    def _dir(self, milestone_key: str) -> Path:
        return self.root / milestone_key

    @staticmethod
    def _reward_of(path: Path) -> float:
        m = _REWARD_RE.search(path.name)
        return float(m.group(1)) if m else 0.0

    def _files(self, milestone_key: str) -> List[Path]:
        d = self._dir(milestone_key)
        if not d.exists():
            return []
        return sorted(d.glob("*.state"))

    # ------------------------------------------------------------------ #
    def save(self, milestone_key: str, pyboy, reward: float = 0.0,
             step: int = 0) -> Path:
        """Snapshot the current emulator state under ``milestone_key``."""
        d = self._dir(milestone_key)
        d.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        fname = d / f"{milestone_key}_r{reward:.1f}_{uid}.state"
        # Write to a temp name first, then atomically rename. With many SubprocVecEnv
        # workers writing while others read for Go-Explore warm-starts, a reader must
        # never observe a half-written ".state" (that would crash load_state and
        # deadlock the vec env). ".tmp" files are not matched by the "*.state" glob.
        tmp = d / f".{milestone_key}_{uid}.state.tmp"
        with open(tmp, "wb") as f:
            pyboy.save_state(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fname)  # atomic on the same filesystem
        self._prune(milestone_key)
        return fname

    def _prune(self, milestone_key: str) -> None:
        """Keep only the best ``max_per_milestone`` snapshots (by encoded reward)."""
        files = self._files(milestone_key)
        if len(files) <= self.max_per_milestone:
            return
        files.sort(key=self._reward_of, reverse=True)
        for stale in files[self.max_per_milestone:]:
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def has_states(self, milestone_key: str) -> bool:
        return len(self._files(milestone_key)) > 0

    def count(self, milestone_key: str) -> int:
        return len(self._files(milestone_key))

    def select(self, milestone_key: str, strategy: str = "random",
               rng=None) -> Optional[str]:
        """Return a snapshot file path for ``milestone_key`` (or None).

        strategy: ``"random"`` | ``"latest"`` | ``"best"``.
        """
        files = self._files(milestone_key)
        if not files:
            return None
        if strategy == "best":
            return str(max(files, key=self._reward_of))
        if strategy == "latest":
            return str(max(files, key=lambda p: p.stat().st_mtime))
        import random as _random
        r = rng or _random
        return str(r.choice(files))

    def select_from_any(self, milestone_keys: List[str], strategy: str = "random",
                        rng=None) -> Optional[str]:
        """Pick a snapshot from the first key (in order) that has any."""
        for key in milestone_keys:
            path = self.select(key, strategy=strategy, rng=rng)
            if path is not None:
                return path
        return None
