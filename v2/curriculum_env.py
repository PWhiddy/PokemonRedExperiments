"""
curriculum_env.py -- CurriculumRedGymEnv: the hierarchical/curriculum environment.

This subclasses the existing v2 :class:`RedGymEnv` and wires together the curriculum
package (structured observations, milestone manager, configurable rewards, anti-loop
penalties, scripted helpers, and the per-milestone state store) **without modifying**
the base environment. ``baseline_fast_v2.py`` and ``red_gym_env_v2.py`` keep working
exactly as before.

What it changes relative to the base env:

* Observation: a primary structured RAM/event vector (``obs["structured"]``), with the
  pixel frame-stack kept as an optional auxiliary stream (``obs["screens"]``).
* Reward: computed by :class:`curriculum.rewards.RewardManager` (configurable, logged,
  one-time/max-based, with anti-loop/anti-farm penalties) instead of the base reward.
* Episodes: can start from saved success snapshots of a previous milestone and can end
  when a target milestone is reached or on a blackout.
* On milestone completion, the exact emulator state is snapshotted to the state store
  so later curriculum stages can warm-start from it.

The class degrades gracefully: with ``use_structured_obs=False`` and ``use_screens=True``
and no ``target_milestone`` it behaves much like the base env but with the new reward.
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import spaces

from red_gym_env_v2 import RedGymEnv

from curriculum.ram_map import GameState
from curriculum.milestones import MilestoneManager, build_default_milestones
from curriculum.rewards import RewardConfig, RewardManager
from curriculum.structured_obs import StructuredObservationBuilder
from curriculum.scripted_helpers import ScriptedHelpers
from curriculum.state_store import StateStore
from curriculum.subgoal import Subgoal, SubgoalShaper, skill_id
from curriculum.llm_planner import build_planner, PlannerContext
from curriculum.skill_bridge import AdviceController, AdviceConfig


class CurriculumRedGymEnv(RedGymEnv):
    def __init__(self, config=None):
        config = dict(config or {})

        # --- pull curriculum extras (with defaults) before base init ---
        self.use_structured_obs = config.get("use_structured_obs", True)
        self.use_screens = config.get("use_screens", True)
        self.history_len = int(config.get("history_len", 8))
        self.emulation_speed = int(config.get("emulation_speed", 0))
        self.target_milestone = config.get("target_milestone", None)
        self.start_state_strategy = config.get("start_state_strategy", "random")
        self.save_success_states = config.get("save_success_states", True)
        self.success_state_min_reward = float(config.get("success_state_min_reward", 0.0))
        self.end_on_target = config.get("end_on_target", False)
        self.end_on_blackout = config.get("end_on_blackout", False)
        self.use_scripted_helpers = config.get("use_scripted_helpers", False)
        self.auto_advance_dialogue = config.get("auto_advance_dialogue", False)
        self.dialogue_taps = int(config.get("dialogue_taps", 8))
        # --- Go-Explore: probability an episode restarts from a saved frontier
        #     success-state instead of init.state, even with no target milestone. ---
        self.start_state_prob = float(config.get("start_state_prob", 0.0))
        self.frontier_window = int(config.get("frontier_window", 4))
        # --- efficiency: truncate an episode that has made no progress for too long
        #     (0 disables). Recycles wasted samples toward productive states. ---
        self.max_steps_without_progress = int(config.get("max_steps_without_progress", 0))
        # --- experimental: deterministically use Cut when wall-bumping with Cut
        #     available (the classic post-SS-Anne bottleneck). Off by default. ---
        self.auto_use_cut = bool(config.get("auto_use_cut", False))
        self.cut_trigger_bumps = int(config.get("cut_trigger_bumps", 6))
        self._cut_cooldown = 0
        self._rng = random.Random()

        # --- high-level planner + subgoal shaping (the "think, then learn" layer) ---
        self.use_planner = bool(config.get("use_planner", False))
        self.planner_kind = config.get("planner_kind", "rule")  # none|rule|ollama
        self.planner_model = config.get("planner_model", "nemotron-mini")
        self.planner_host = config.get("planner_host", "http://localhost:11434")
        self.planner_timeout = float(config.get("planner_timeout", 20.0))
        self.replan_interval = int(config.get("replan_interval", 512))
        self.replan_on_stuck = bool(config.get("replan_on_stuck", True))
        self.planner_verbose = bool(config.get("planner_verbose", True))
        self.llm_decision_count = 0
        # append every LLM decision to a JSONL file so the live monitor window
        # (v2/llm_monitor.py) and post-hoc review can read them.
        self._llm_log_path = None
        try:
            sp = Path(config.get("session_path") or "runs_curriculum")
            sp.mkdir(parents=True, exist_ok=True)
            self._llm_log_path = sp / "llm_decisions.jsonl"
        except Exception:
            self._llm_log_path = None
        self.subgoal_shaping_weight = float(config.get("subgoal_shaping_weight", 1.0))
        self.subgoal_reach_bonus = float(config.get("subgoal_reach_bonus", 8.0))

        # --- advice controller: make the policy follow the planner's decisions ---
        self.use_advice = bool(config.get("use_advice", False))
        self.advice_w_align = float(config.get("advice_w_align", 0.02))
        self.advice_trigger_bumps = int(config.get("advice_trigger_bumps", 4))
        self.advice_success_bonus = float(config.get("advice_success_bonus", 5.0))
        self.advice_require_planner = bool(config.get("advice_require_planner", True))
        state_store_root = config.get("state_store_root", "curriculum_states")
        reward_config: RewardConfig = config.get("reward_config", None) or RewardConfig()

        # --- base env init (creates pyboy, base observation_space) ---
        super().__init__(config)

        # --- emulation speed (base sets 6 when windowed; honor our config) ---
        if not self.headless and self.emulation_speed > 0:
            self.pyboy.set_emulation_speed(self.emulation_speed)

        # --- curriculum components (persist across resets) ---
        self.gs = GameState(self.pyboy)
        self.milestones = build_default_milestones()
        self.milestone_mgr = MilestoneManager(self.milestones, target_key=self.target_milestone)
        self.reward_mgr = RewardManager(reward_config)
        self.state_store = StateStore(state_store_root)
        self.helpers = ScriptedHelpers(
            self.pyboy, act_freq=self.act_freq, headless=self.headless,
            save_video=self.save_video,
        ) if (self.use_scripted_helpers or self.auto_use_cut or self.use_advice) else None

        # planner (rule-based / local-LLM) + potential-based subgoal shaper. The
        # planner is only consulted on a cadence or when the agent stalls, and the
        # Ollama variant runs off-thread so the env step never blocks on the model.
        self.planner = build_planner(
            self.planner_kind if self.use_planner else "none",
            model=self.planner_model, host=self.planner_host,
            timeout=self.planner_timeout, asy=True,
            on_decision=self._on_llm_decision,
        )
        self.shaper = SubgoalShaper(
            gamma=0.998, weight=self.subgoal_shaping_weight,
            reach_bonus=self.subgoal_reach_bonus,
        )
        self.active_subgoal: Subgoal = Subgoal(source="none")
        self._steps_since_replan = 0
        self._prev_map = 0
        self._last_frontier_key = None

        # advice controller: rewards the policy for following the planner's advised
        # skill (any skill) and executes the field-move macros PPO can't learn.
        self.advice = AdviceController(AdviceConfig(
            enabled=self.use_advice, w_align=self.advice_w_align,
            trigger_bumps=self.advice_trigger_bumps,
            require_advice=self.advice_require_planner,
        ))
        self._advice_exec_pending = False

        self.num_actions = len(self.valid_actions)
        self.obs_builder = StructuredObservationBuilder(
            num_actions=self.num_actions,
            history_len=self.history_len,
            num_milestones=len(self.milestones),
        )

        # histories for the structured observation (most-recent-first)
        self._action_hist = deque(maxlen=self.history_len)
        self._map_hist = deque(maxlen=self.history_len)
        self.episode_reward = 0.0

        # --- rebuild observation space with our keys ---
        self.observation_space = self._build_observation_space()

    # ------------------------------------------------------------------ #
    def _build_observation_space(self) -> spaces.Dict:
        space = {}
        if self.use_structured_obs:
            space["structured"] = self.obs_builder.space()
        if self.use_screens or not self.use_structured_obs:
            # keep the pixel frame-stack (always keep at least one stream)
            space["screens"] = spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8)
        return spaces.Dict(space)

    # ------------------------------------------------------------------ #
    def _choose_start_state(self) -> Optional[str]:
        """Pick a warm-start snapshot for this episode, or None to use init.state.

        Two mechanisms:
        * Targeted curriculum: start from the target milestone's allowed predecessor
          snapshots (used by --staged / --target-milestone).
        * Go-Explore: with probability ``start_state_prob`` restart from a saved
          *frontier* success-state even without a target. As the agent reaches deeper
          milestones, more episodes begin near the frontier, so a far-from-start wall
          (e.g. just past the SS Anne) gets a real share of training samples instead
          of ~none. Falls back to init.state until any states have been saved.
        """
        if self.target_milestone:
            ms = self.milestone_mgr.by_key.get(self.target_milestone)
            if ms is None or not ms.allowed_start_states:
                return None
            return self.state_store.select_from_any(
                ms.allowed_start_states, strategy=self.start_state_strategy)

        if self.start_state_prob > 0.0 and self._rng.random() < self.start_state_prob:
            return self._select_frontier_state()
        return None

    def _select_frontier_state(self) -> Optional[str]:
        """Choose a saved snapshot biased toward the furthest reached milestones."""
        keys_with_states = [m.key for m in self.milestones
                            if self.state_store.has_states(m.key)]
        if not keys_with_states:
            return None
        # bias toward the frontier: sample among the last ``frontier_window`` keys
        candidates = keys_with_states[-max(self.frontier_window, 1):]
        key = self._rng.choice(candidates)
        return self.state_store.select(key, strategy=self.start_state_strategy)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        options = options or {}
        if seed is not None:
            self._rng.seed(seed)
        self._cut_cooldown = 0
        chosen = self._choose_start_state()
        original_init = self.init_state
        if chosen is not None:
            self.init_state = chosen

        # base reset loads init_state and resets base bookkeeping. A warm-start
        # snapshot can occasionally be unreadable (e.g. picked up the instant another
        # worker created it); never let that crash the worker -- fall back to the
        # original init.state so the vec env can't deadlock.
        try:
            _, info = super().reset(seed=seed, options=options)
        except Exception as e:
            if chosen is not None:
                print(f"[CurriculumEnv] warm-start load failed ({e}); "
                      f"falling back to {original_init}")
                self.init_state = original_init
                _, info = super().reset(seed=seed, options=options)
            else:
                raise

        self.init_state = original_init

        # (re)initialize curriculum trackers against the freshly loaded state
        self._action_hist.clear()
        self._map_hist.clear()
        self.episode_reward = 0.0
        self.reward_mgr.reset()
        self.reward_mgr.baseline(self.gs)
        self.milestone_mgr.reset()
        # latch already-satisfied milestones silently (no reward for them)
        self.milestone_mgr.update(self.gs)

        # reset planner/subgoal state for the new episode
        self.shaper.reset()
        self.advice.reset()
        self._advice_exec_pending = False
        self.active_subgoal = Subgoal(source="none")
        self._steps_since_replan = 0
        self._prev_map = self.gs.map_id()
        self._last_frontier_key = None
        if self.planner is not None:
            self._maybe_replan(force=True)

        return self._get_obs(), info

    # ------------------------------------------------------------------ #
    def _push_history(self, action: int) -> None:
        self._action_hist.appendleft(int(action))
        self._map_hist.appendleft(int(self.gs.map_id()))

    # ------------------------------------------------------------------ #
    def _build_planner_context(self) -> PlannerContext:
        x, y, m = self.gs.position()
        fr = self.milestone_mgr.frontier()
        sig = self.reward_mgr.last_signals
        return PlannerContext(
            map_id=m, x=x, y=y,
            badge_count=self.gs.badge_count(),
            party_levels=self.gs.party_levels(),
            party_hp_frac=self.gs.total_hp_fraction(),
            in_battle=self.gs.in_battle(),
            frontier_key=fr.key if fr else None,
            frontier_name=fr.name if fr else "",
            recent_maps=list(self._map_hist),
            stuck_score=float(sig.get("stuck_score", 0.0)),
            loop_score=float(sig.get("loop_score", 0.0)),
            inactivity_score=float(sig.get("inactivity_score", 0.0)),
        )

    def _on_llm_decision(self, ctx, subgoal, latency) -> None:
        """Callback fired (from the planner's background thread) whenever the local LLM
        actually produces a decision -- prints the CLI panel so you can review exactly
        when the agent 'thought' with the model vs ran on a cached/rule default."""
        self.llm_decision_count += 1
        if self.planner_verbose:
            from curriculum.llm_planner import render_llm_decision
            render_llm_decision(ctx, subgoal, latency,
                                count=self.llm_decision_count, model=self.planner_model)
        # persist to JSONL for the live monitor window / later review
        if self._llm_log_path is not None:
            try:
                rec = {
                    "n": self.llm_decision_count,
                    "t": time.time(),
                    "model": self.planner_model,
                    "latency": round(latency, 2),
                    "map_id": ctx.map_id, "x": ctx.x, "y": ctx.y,
                    "badges": ctx.badge_count,
                    "frontier": ctx.frontier_name or ctx.frontier_key or "",
                    "stuck": round(ctx.stuck_score, 2),
                    "loop": round(ctx.loop_score, 2),
                    "inactivity": round(ctx.inactivity_score, 2),
                    "skill": subgoal.skill,
                    "target_map": subgoal.target_map,
                    "goal": subgoal.text,
                    "why": subgoal.reasoning,
                }
                with open(self._llm_log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass  # logging must never break training

    def _maybe_replan(self, force: bool = False) -> None:
        """Consult the planner on a cadence, when the frontier advances, or when the
        agent is clearly stuck. Non-blocking for the Ollama (async) planner."""
        if self.planner is None:
            return
        fr = self.milestone_mgr.frontier()
        fr_key = fr.key if fr else None
        sig = self.reward_mgr.last_signals
        stuck = self.replan_on_stuck and (
            float(sig.get("inactivity_score", 0.0)) > 0.5
            or float(sig.get("loop_score", 0.0)) > 0.5)
        frontier_changed = fr_key != self._last_frontier_key
        reached = self.shaper.reached(self.active_subgoal.target_map)
        due = self._steps_since_replan >= self.replan_interval
        if not (force or frontier_changed or reached or due or stuck):
            return
        ctx = self._build_planner_context()
        try:
            self.active_subgoal = self.planner.propose(ctx)
        except Exception as e:
            print(f"[CurriculumEnv] planner.propose failed: {e}")
            self.active_subgoal = Subgoal(source="none")
        self._steps_since_replan = 0
        self._last_frontier_key = fr_key

    def _inject_subgoal_signals(self, sig) -> None:
        sg = self.active_subgoal
        sig.subgoal_active = 1.0 if (sg and sg.target_map is not None) else 0.0
        sig.subgoal_target_map = int(sg.target_map) if (sg and sg.target_map is not None) else 0
        sig.subgoal_skill_id = skill_id(sg.skill) if sg else -1
        hops = None
        if sg and sg.target_map is not None:
            hops = self.shaper.graph.hops(self.gs.map_id(), sg.target_map,
                                          cap=self.shaper.dist_cap)
        sig.subgoal_hops = (min(hops, self.shaper.dist_cap) / self.shaper.dist_cap
                            if hops is not None else 1.0)
        sig.advice_active = 1.0 if getattr(self.advice, "advice_active", False) else 0.0

    def _get_obs(self):
        obs = {}
        if self.use_structured_obs:
            sig = self.reward_mgr.episode_signals(
                self.milestone_mgr,
                list(self._action_hist),
                list(self._map_hist),
                len(self.milestones),
            )
            self._inject_subgoal_signals(sig)
            obs["structured"] = self.obs_builder.build(self.gs, sig)
        if self.use_screens or not self.use_structured_obs:
            screen = self.render()
            self.update_recent_screens(screen)
            obs["screens"] = self.recent_screens
        return obs

    # ------------------------------------------------------------------ #
    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        # advance emulator + base bookkeeping we still want for logging/visuals
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_map_progress()
        self.party_size = self.read_m(0xD163)

        # optional deterministic dialogue advancement so menus don't hard-block
        if self.helpers is not None and self.auto_advance_dialogue:
            self.helpers.advance_dialogue(self.dialogue_taps)

        # optional deterministic Cut assist: if the agent keeps bumping into something
        # while Cut is usable, try cutting (handles the post-SS-Anne tree the policy
        # otherwise can't learn the menu sequence for). Cooldown avoids menu spam.
        self._cut_assisted = False
        if self.helpers is not None and self.auto_use_cut:
            if self._cut_cooldown > 0:
                self._cut_cooldown -= 1
            elif (self.reward_mgr.anti.wall_bump_run >= self.cut_trigger_bumps
                  and self.gs.can_use_cut()):
                if self.helpers.field_use_cut(self.dialogue_taps):
                    self._cut_assisted = True
                    self._cut_cooldown = 12  # don't retry every step

        # curriculum: milestones first (so reward sees newly_completed), then reward
        newly = self.milestone_mgr.update(self.gs)
        comp = self.reward_mgr.step(self.gs, action, self.milestone_mgr)

        # high-level planner: (re)choose a subgoal, then add potential-based shaping
        # toward it. Shaping is a difference of potentials, so it densifies the
        # gradient without changing the optimal policy (cannot create a new exploit).
        cur_map = self.gs.map_id()
        self._steps_since_replan += 1
        if self.planner is not None:
            self._maybe_replan()
            shaping = self.shaper.step(self._prev_map, cur_map, self.active_subgoal)
            if shaping != 0.0:
                comp["subgoal"] = comp.get("subgoal", 0.0) + shaping
        self._prev_map = cur_map

        # advice controller: reward the policy for following the planner's advised skill
        # (any skill) and execute field-move macros it can't learn. This is how the
        # LLM's decision gets implemented in the model -- behaviour consistent with the
        # advice is reinforced every step it is advised.
        if self.advice.cfg.enabled:
            sig = self.reward_mgr.last_signals
            adv_r = self.advice.step(self.gs, self.reward_mgr.anti,
                                     self.active_subgoal, self.helpers, comp, sig)
            # attribute a one-time SUCCESS bonus when a field-move execution on the
            # previous step opened real progress this step (new map/tile). Farm-safe:
            # a macro that just opened a menu against a wall yields nothing here.
            if self._advice_exec_pending and ("new_map" in comp or "explore" in comp):
                adv_r += self.advice_success_bonus
            self._advice_exec_pending = self.advice.last_executed is not None
            if adv_r != 0.0:
                comp["advice"] = comp.get("advice", 0.0) + adv_r

        step_reward = self.reward_mgr.scaled_total(comp)
        self.episode_reward += step_reward

        # keep base stat fields meaningful for the existing TensorboardCallback
        self.total_healing_rew = self.reward_mgr.heal_total
        self.died_count = self.reward_mgr.blackout_count

        # snapshot successful milestone states for future curriculum stages
        if self.save_success_states and newly:
            self._maybe_snapshot(newly)

        # histories used by the next observation
        self._push_history(action)

        # base agent_stats (augmented) for logging
        self.append_agent_stats(action)

        # termination / truncation
        truncated = self.check_if_done()
        # efficiency: cut episodes that have stalled (no new map/event/badge/level)
        # for too long so PPO stops burning rollout on a dead state.
        if (self.max_steps_without_progress > 0
                and self.reward_mgr.anti.steps_since_progress
                > self.max_steps_without_progress):
            truncated = True
        terminated = False
        if self.end_on_target and self.milestone_mgr.reached_target():
            terminated = True
        if self.end_on_blackout and self.gs.all_fainted():
            terminated = True

        obs = self._get_obs()
        self.step_count += 1

        # Surface the milestones that *completed this step* (key, name, reward) so a
        # callback can announce them and persist progress. ``newly`` is the list of
        # keys MilestoneManager latched this step.
        new_milestones = []
        for key in newly:
            ms = self.milestone_mgr.by_key.get(key)
            if ms is None:
                continue
            new_milestones.append({
                "key": ms.key,
                "name": ms.name,
                "reward": ms.reward,
                "step": self.step_count,
                "badges": self.gs.badge_count(),
                "party_levels_sum": self.gs.party_levels_sum(),
                "map_id": self.gs.map_id(),
            })

        info = {
            "milestones_completed": self.milestone_mgr.completion_count(),
            "milestone_progress": self.milestone_mgr.progress_fraction(),
            "reached_target": self.milestone_mgr.reached_target(),
            "episode_reward": self.episode_reward,
            "reward_components": dict(comp),
            "new_milestones": new_milestones,
            "completed_milestone_keys": sorted(self.milestone_mgr.completed),
            "cut_assisted": getattr(self, "_cut_assisted", False),
            "subgoal_text": self.active_subgoal.text if self.active_subgoal else "",
            "subgoal_target_map": (self.active_subgoal.target_map
                                   if self.active_subgoal else None),
            "subgoal_source": self.active_subgoal.source if self.active_subgoal else "none",
            "subgoal_skill": self.active_subgoal.skill if self.active_subgoal else "",
            "advice_following": getattr(self.advice, "following", False),
            "advice_executed": getattr(self.advice, "last_executed", None),
            "llm_decisions": self.llm_decision_count,
            "adaptive_stats": self.reward_mgr.adaptive_stats(),
        }
        return obs, step_reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _maybe_snapshot(self, newly_completed) -> None:
        for key in newly_completed:
            ms = self.milestone_mgr.by_key.get(key)
            if ms is None or not ms.save_state:
                continue
            if self.episode_reward < self.success_state_min_reward:
                continue
            try:
                self.state_store.save(
                    key, self.pyboy, reward=self.episode_reward, step=self.step_count)
            except Exception as e:  # never let snapshotting break training
                print(f"[CurriculumEnv] snapshot failed for {key}: {e}")

    # ------------------------------------------------------------------ #
    def append_agent_stats(self, action):
        """Override base stats with curriculum-aware fields for TensorBoard.

        TensorboardCallback averages numeric fields of ``agent_stats[-1]`` across
        envs, so exposing milestone/reward counters here surfaces them in TB.
        """
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = self.gs.party_levels()
        self.agent_stats.append({
            "step": self.step_count,
            "x": x_pos,
            "y": y_pos,
            "map": map_n,
            "max_map_progress": self.max_map_progress,
            "last_action": action,
            "pcount": self.gs.party_count(),
            "levels_sum": sum(levels),
            "hp": self.gs.total_hp_fraction(),
            "coord_count": len(self.seen_coords),
            "deaths": self.reward_mgr.blackout_count,
            "badge": self.gs.badge_count(),
            "event_flags": self.gs.event_flags_sum(),
            "milestones": self.milestone_mgr.completion_count(),
            "milestone_progress": self.milestone_mgr.progress_fraction(),
            "money": self.gs.money(),
            "pokedex_owned": self.gs.pokedex_owned_count(),
            "episode_reward": self.episode_reward,
            "healr": self.total_healing_rew,
        })
