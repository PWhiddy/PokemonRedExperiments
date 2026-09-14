# Adaptive / Dynamic Agent Layers

This document covers the three layers added on top of the curriculum env to fix the
**"saturate, then exploit the reward"** failure and push the agent from *"learn to
finish the game"* toward *"think → adapt → learn"*.

Nothing here replaces the existing system — every layer is **additive and toggleable**.
With all switches off, training behaves exactly as before.

---

## Why it used to saturate then exploit

The old reward used **fixed weights**. That is fundamentally fragile in a 33-milestone,
non-stationary game:

1. `w_new_coord` is one-shot per tile → exploration reward **dries to zero** once a
   region is mapped, and the next milestone is far/sparse, so there is no gradient.
2. With exploration gone, the densest *renewable* term (grinding `w_level`) dominates
   the return and PPO collapses onto it — that is the "exploit". Classic reward hacking.
3. Anti-loop penalties (`0.05`) are ~50× weaker than the farm gradient, so they can't help.

No single static weight vector is correct across the whole game. So we made the system
adapt.

---

## Layer 1 — Adaptive reward core (fixes saturation + exploitation)

### 1a. Non-saturating intrinsic motivation — [`intrinsic.py`](intrinsic.py)
A curiosity bonus that is a function of **how often a state has been visited**, so it
stays high at the frontier, decays smoothly as a region is mastered, and *revives* when
the agent reaches genuinely new ground.

- `intrinsic_kind: count` (default) — pure-numpy pseudo-count `coef / sqrt(N(state))`.
  Cheap, per-env, perfect for many `SubprocVecEnv` workers on CPU. The state hash is
  *progress-aware* (same tile after a new badge / chunk of story flags counts as novel
  again).
- `intrinsic_kind: rnd` — Random Network Distillation (needs a GPU). Generalizes across
  similar states; opt-in.

### 1b. Homeostatic reward controller — [`reward_controller.py`](reward_controller.py)
A closed control loop around the weights. Each step it tracks every component's share of
recent reward and, when a **farmable** term (`explore / level / heal / pokedex /
intrinsic`) runs away **while real progress has stalled**, it multiplicatively **decays
that term's weight** — then relaxes it back to neutral once progress resumes.

- **Objective terms are protected**: `milestone / badge / event / new_map / subgoal` are
  *never* throttled, so the controller can only redirect the agent away from busy-work
  toward real progress — it can't suppress the goal.
- You can watch it adapt live in TensorBoard under `adaptive/adapt_mult/*`.

> Verified in the smoke test: with progress stalled, the `intrinsic` multiplier
> auto-dropped to ~0.68 while the protected terms stayed at 1.0.

---

## Layer 2 — The "think, then learn" planner — [`llm_planner.py`](llm_planner.py)

When the agent stalls (loop/stuck/inactivity spikes) a flat policy can't *reason* about
why. A planner inspects the state and proposes the next **subgoal** (a target map + a
suggested skill + a one-line rationale):

- `RuleBasedPlanner` — instant, zero-dependency lookup from the current frontier
  milestone to a curated target (the bundled Kanto route knowledge = the "walkthrough").
- `OllamaPlanner` — a **local** LLM (default `nemotron-mini`) grounded by that route
  knowledge, returning a JSON subgoal. No API key, no cloud.
- `AsyncPlanner` — wraps the LLM so the env **never blocks**: it serves the rule-based
  answer instantly and refreshes with the LLM answer on a background thread, cached by
  `(frontier milestone, current map)`.

> Measured here: env steps at **2.4 ms/step** even with the planner firing every 10
> steps; the ~10 s LLM call runs off-thread and upgrades the subgoal `rule → llm` once
> ready.

#### Seeing when the LLM "thinks" — CLI panel
With `planner_verbose: true` (default) a panel prints **only when the local LLM actually
produces a decision** (not on cached or rule-based steps), so you can review exactly when
the agent reasoned with the model:

```
+============== LLM DECISION #1 (nemotron-mini, 10.7s) ===============+
  when : Cerulean City (3) @(10,12) | badges 1/8 | objective: Beat Misty
  why  : stuck=0.30 loop=0.20 inactivity=0.70  (high => agent was stuck)
  -> skill : battle      -> target : 3 (Cerulean City)
  goal : Beat Misty at Cerulean Gym to get the Cascade Badge
  model: Need the Cascade Badge to proceed east
+====================================================================+
```

`info["llm_decisions"]` also counts how many times the LLM has been consulted. For a
clean review, run with `--num-envs 1` (or `--mode visual`) so panels from many parallel
workers don't interleave; set `planner_verbose: false` to silence it during big runs.

### Subgoal shaping — [`subgoal.py`](subgoal.py)
The subgoal becomes a **dense, safe** gradient via potential-based reward shaping
(`F = γΦ(s') − Φ(s)`), where Φ is the negative map-hop distance to the target on a graph
the env learns online from observed transitions. Because it's a potential difference, it
**provably cannot change the optimal policy** — a bad LLM suggestion can't create a new
exploit, it just (un)densifies the path. Reaching the target map pays a one-time bonus.

---

## Layer 3 — Hierarchical control / "MARL" — [`skills.py`](skills.py)

The trainable core is a **subgoal-conditioned policy**: the chosen subgoal's target map,
hop-distance and skill one-hot are fed into the observation, so one PPO policy learns
*different behavior for different situations* (the practical, sample-efficient form of
multi-policy control on this hardware).

`PlannerManager` is the explicit **manager that selects a specialist** per situation
(battle → battle specialist, etc.). Specialists are dropped into the `SkillRegistry` as
`PolicySkill` (a trained model) or `ScriptedSkill`; until one is registered, the
subgoal-conditioned PPO policy stays in control. So this composes with — rather than
replaces — the single-policy trainer.

> True multi-agent RL (MARL) is for *several agents sharing one world*; Pokémon Red is
> single-player, so the right tool is hierarchical RL with specialist sub-policies. That
> is what this layer implements.

### Making the policy OBEY the planner — [`skill_bridge.py`](skill_bridge.py)
Conditioning the policy on the advised skill lets it *see* the decision; the
`AdviceController` is what makes it *follow* the decision and thereby implements the
LLM's choices in the model. Each step the advised skill is active, behaviour consistent
with it earns a small, farm-safe **alignment reward**:

| Planner advises | Policy is rewarded when it… |
|---|---|
| `navigation` | actually moves (not wall-bumping) |
| `battle` | is in / wins the advised battle |
| `healing` | restores HP |
| `grinding` | gains a level |
| `story` | advances an event flag / enters a new map |
| `unstuck` | breaks out of a loop (moves off over-visited tiles) |
| `cut` / `surf` / `strength` | engages the advised obstacle → then a **macro** executes the unlearnable buttons |

So the LLM's decision is realized two ways at once: **learnable** skills are biased into
the policy by reward (it *learns* to obey, conditioned on the advised-skill observation),
and **unlearnable** field moves are executed by a deterministic macro once the policy
commits — with a one-time *success* bonus paid only when the macro opens real progress
(farm-safe). An `advice_active` observation bit flags "a field move is advised **and**
usable right now," giving the policy a crisp cue for the moment that matters (e.g. the
SS-Anne Cut tree). Enable with `use_advice: true` (needs a planner) or `--advice`.

---

## How to run

```bash
# Adaptive reward only (the saturation/exploitation fix) — recommended first run:
python train_full_game_curriculum.py --num-envs 8 --intrinsic count

# Add the local-LLM planner + subgoal shaping (Ollama must be running):
python train_full_game_curriculum.py --num-envs 8 --planner ollama --planner-model nemotron-mini

# Rule-based planner (no LLM, instant) — good for ablations:
python train_full_game_curriculum.py --num-envs 8 --planner rule

# Disable the homeostatic controller (to A/B its effect):
python train_full_game_curriculum.py --num-envs 8 --no-adaptive
```

All of the above also read [`configs/curriculum.yaml`](../configs/curriculum.yaml),
where every weight/threshold for these layers lives (sections: `reward.intrinsic_*`,
`reward.adaptive_*`, and the top-level `planner_*` / `subgoal_*` keys).

### TensorBoard panels to watch
- `adaptive/adapt_mult/*` — live weight multipliers (see the controller starving a farm).
- `adaptive/adapt_share/*` — each farmable term's share of recent reward.
- `adaptive/intrinsic_unique_states` — exploration breadth (should keep climbing).
- `planner/subgoal_target_map` — the map the planner is currently steering toward.
- `env_stats/milestones`, `curriculum/milestone_progress` — the real objective.

---

## Hardware note (this machine)

15.7 GB RAM, RTX 3050 Ti (4 GB), CPU-only torch. So:
- intrinsic defaults to **count-based** (no torch), not RND.
- the planner uses **`nemotron-mini` (4B, ~10 s warm)** locally — *Nemotron 3 Ultra*
  (253B, ~140 GB) **cannot run here** and was not installed; `nemotron-mini` is the
  runnable member of that family. Swap `planner_model` for any Ollama model you pull.
