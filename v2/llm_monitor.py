"""
llm_monitor.py -- a live GUI dashboard for the agent's ACHIEVEMENTS and LLM decisions.

During training the env/callbacks write two JSONL files into the session folder:
  * ``llm_decisions.jsonl``  -- every genuine LLM planner decision
  * ``milestones_log.jsonl`` -- every milestone reach (Beat Brock, Exit Mt. Moon, ...)

This standalone window *tails* both and shows, live:
  * an ACHIEVEMENTS panel that banners each major checkpoint the first time it's reached,
  * the LATEST LLM decision, and a scrolling history of LLM decisions.

So you get one dedicated dashboard next to a (headless) training run -- no console
scrolling required.

Usage
-----
    python v2/llm_monitor.py --session runs_adaptive
    python v2/llm_monitor.py --session runs_adaptive_visual

Run it in its own terminal before/after launching training with a planner enabled
(e.g. ``--advice --planner ollama``). It auto-waits for the files, so order does not
matter. Pure standard-library Tkinter -- no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

try:
    from curriculum.llm_planner import map_name  # nice city names
except Exception:
    def map_name(m):  # fallback if import fails
        return f"map {m}"

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:  # pragma: no cover
    print("Tkinter is required for the GUI monitor and is not available:", e)
    print("Tip: you can still watch decisions in the console (planner_verbose: true) "
          "or tail the JSONL file directly.")
    sys.exit(1)


BG = "#0b1021"
FG = "#e6e6e6"
ACCENT = "#5ad1ff"
GREEN = "#7cf08a"
DIM = "#8a93b2"
YELLOW = "#ffd166"


class _Tailer:
    """Reads new whole lines appended to a growing file across polls."""

    def __init__(self, path: Path):
        self.path = path
        self._pos = 0
        self._buf = ""

    def new_lines(self):
        out = []
        try:
            if not self.path.exists():
                return out
            size = self.path.stat().st_size
            if size < self._pos:            # truncated/rotated -> restart
                self._pos, self._buf = 0, ""
            if size > self._pos:
                with open(self.path, "r", encoding="utf-8") as fh:
                    fh.seek(self._pos)
                    chunk = fh.read()
                    self._pos = fh.tell()
                self._buf += chunk
                *lines, self._buf = self._buf.split("\n")
                out = [ln.strip() for ln in lines if ln.strip()]
        except Exception:
            pass
        return out


class LLMMonitor:
    def __init__(self, root: "tk.Tk", session: Path, poll_ms: int = 500):
        self.root = root
        self.session = session
        self.poll_ms = poll_ms
        self._dec = _Tailer(session / "llm_decisions.jsonl")
        self._ms = _Tailer(session / "milestones_log.jsonl")
        self._count = 0
        self._ach_count = 0

        root.title("Pokemon Red RL — Live Dashboard")
        root.configure(bg=BG)
        root.geometry("720x760")

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=12, pady=(12, 4))
        tk.Label(header, text="POKEMON RED RL — LIVE DASHBOARD", bg=BG, fg=ACCENT,
                 font=("Consolas", 14, "bold")).pack(side="left")
        self.status = tk.Label(header, text="waiting for the run…", bg=BG, fg=DIM,
                               font=("Consolas", 10))
        self.status.pack(side="right")

        # --- ACHIEVEMENTS panel (milestones) ---
        ach = tk.LabelFrame(root, text=" ACHIEVEMENTS (major checkpoints reached) ",
                            bg=BG, fg=GREEN, font=("Consolas", 10, "bold"),
                            bd=2, relief="groove")
        ach.pack(fill="both", expand=True, padx=12, pady=(8, 4))
        self.ach = tk.Text(ach, bg="#06120a", fg=FG, font=("Consolas", 11), wrap="word",
                           height=10, bd=0)
        self.ach.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb1 = ttk.Scrollbar(ach, command=self.ach.yview)
        sb1.pack(side="right", fill="y")
        self.ach.configure(yscrollcommand=sb1.set, state="disabled")
        self.ach.tag_config("win", foreground=GREEN, font=("Consolas", 12, "bold"))
        self.ach.tag_config("dim", foreground=DIM)

        # --- latest LLM decision ---
        latest = tk.LabelFrame(root, text=" latest LLM decision ", bg=BG, fg=YELLOW,
                               font=("Consolas", 10, "bold"), bd=2, relief="groove")
        latest.pack(fill="x", padx=12, pady=4)
        self.latest_var = tk.StringVar(value="(no LLM decision yet)")
        tk.Label(latest, textvariable=self.latest_var, bg=BG, fg=FG, justify="left",
                 anchor="w", font=("Consolas", 10)).pack(fill="x", padx=10, pady=6)

        # --- LLM decision history ---
        hist = tk.LabelFrame(root, text=" LLM decision history ", bg=BG, fg=DIM,
                             font=("Consolas", 10, "bold"), bd=2, relief="groove")
        hist.pack(fill="both", expand=True, padx=12, pady=4)
        self.text = tk.Text(hist, bg="#070a16", fg=FG, font=("Consolas", 10), wrap="word",
                            height=12, bd=0)
        self.text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        sb2 = ttk.Scrollbar(hist, command=self.text.yview)
        sb2.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=sb2.set, state="disabled")
        self.text.tag_config("hdr", foreground=ACCENT, font=("Consolas", 10, "bold"))
        self.text.tag_config("skill", foreground=GREEN, font=("Consolas", 10, "bold"))
        self.text.tag_config("dim", foreground=DIM)

        tk.Label(root, text=f"tailing  {self.session}", bg=BG, fg=DIM,
                 font=("Consolas", 9)).pack(fill="x", padx=12, pady=(0, 8))

        self.root.after(200, self._poll)

    # ------------------------------------------------------------------ #
    def _poll(self):
        for line in self._ms.new_lines():
            self._handle_milestone(line)
        for line in self._dec.new_lines():
            self._handle_decision(line)
        self.root.after(self.poll_ms, self._poll)

    def _handle_milestone(self, line: str):
        try:
            rec = json.loads(line)
        except Exception:
            return
        if not rec.get("first_reach", False):
            return  # only banner the first time each checkpoint is reached
        self._ach_count += 1
        name = rec.get("name", rec.get("key", "?"))
        step = rec.get("timestep", 0)
        badges = rec.get("badges", "?")
        levels = rec.get("party_levels_sum", "?")
        self.ach.configure(state="normal")
        self.ach.insert("end", f"  #{self._ach_count}  {name}\n", "win")
        self.ach.insert("end",
                        f"        step {step:,} | badges {badges}/8 | party levels {levels}\n\n",
                        "dim")
        self.ach.see("end")
        self.ach.configure(state="disabled")
        self.status.config(text=f"{self._ach_count} achievements | {self._count} LLM calls")

    def _handle_decision(self, line: str):
        try:
            rec = json.loads(line)
        except Exception:
            return
        self._count += 1
        n = rec.get("n", self._count)
        skill = rec.get("skill", "?")
        tgt = rec.get("target_map")
        tgt_s = f"{tgt} ({map_name(tgt)})" if tgt is not None else "-"
        latency = rec.get("latency", 0.0)
        model = rec.get("model", "llm")
        where = f"{map_name(rec.get('map_id', 0))} @({rec.get('x')},{rec.get('y')})"
        frontier = rec.get("frontier", "")
        goal = rec.get("goal", "")
        why = rec.get("why", "")
        stuck, loop, ina = rec.get("stuck", 0), rec.get("loop", 0), rec.get("inactivity", 0)

        self.latest_var.set(
            f"#{n}  ({model}, {latency}s)\n"
            f"when : {where} | badges {rec.get('badges','?')}/8 | objective: {frontier}\n"
            f"why  : stuck={stuck} loop={loop} inactivity={ina}\n"
            f"-> skill: {skill}    -> target: {tgt_s}\n"
            f"goal : {goal}\n"
            f"model: {why}"
        )
        self.status.config(text=f"{self._ach_count} achievements | {self._count} LLM calls")

        self.text.configure(state="normal")
        self.text.insert("end", f"#{n} ", "hdr")
        self.text.insert("end", f"[{model} {latency}s]  ", "dim")
        self.text.insert("end", f"{skill} ", "skill")
        self.text.insert("end", f"-> {tgt_s}\n")
        self.text.insert("end", f"    @ {where} | {frontier}\n", "dim")
        if goal:
            self.text.insert("end", f"    goal: {goal}\n")
        self.text.insert("end", "\n")
        self.text.see("end")
        self.text.configure(state="disabled")


def main():
    ap = argparse.ArgumentParser(description="Live dashboard: achievements + LLM decisions")
    ap.add_argument("--session", default="runs_adaptive",
                    help="session folder containing the JSONL logs")
    ap.add_argument("--poll-ms", type=int, default=500)
    args = ap.parse_args()

    session = Path(args.session)
    session.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    LLMMonitor(root, session, poll_ms=args.poll_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
