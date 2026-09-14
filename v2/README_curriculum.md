# Curriculum / Hierarchical RL Extension (v2)

A practical hierarchical-RL / curriculum-learning system layered **on top of** the
existing `v2/` Pokémon Red environment. It splits the full game into staged
milestones, adds a structured RAM/event observation, a configurable & logged reward
system with anti-loop/anti-farming penalties, saved emulator success-states for
curriculum warm-starts, scripted menu/dialogue helpers, and an option/skill scaffold
for future hierarchical control — while keeping the original V2 scripts working
unchanged.

Nothing here includes or downloads a ROM. You must legally provide
`PokemonRed.gb` at the **repository root** (one level above `v2/`).

---

## 0. Quick start — train past SS Anne / toward the full game

Everything runs from inside `v2/`. The recommended path to get **past the SS Anne
checkpoint** (and keep going toward the Champion) is the staged curriculum: it
warm-starts each milestone from the saved success-states of the previous one, so the
agent doesn't have to re-discover the whole early game every episode.

```bash
cd PokemonRedExperiments/v2

# Whole-game shared policy (simplest; good baseline, auto-resumes by default):
python train_full_game_curriculum.py --mode headless --num-envs 16

# Staged curriculum up to and beyond SS Anne (Vermilion + HM01 Cut):
python train_full_game_curriculum.py --staged --num-envs 16 \
    --milestones start_game,get_starter,first_rival_battle,deliver_parcel,get_pokedex,\
reach_viridian_forest,exit_viridian_forest,reach_pewter,beat_brock,reach_mt_moon,\
exit_mt_moon,reach_cerulean,beat_misty,help_bill,reach_vermilion,get_cut,beat_lt_surge

# Full game, all 33 milestones, staged:
python train_full_game_curriculum.py --staged --num-envs 16

# Watch a single agent in a window:
python train_full_game_curriculum.py --mode visual --speed 3
```

Tune `--num-envs` to your CPU core count (more envs = faster learning). Reaching SS
Anne / Vermilion typically needs tens of millions of timesteps; the Champion needs
far more. Leave it running — it checkpoints and resumes automatically (below).

### Significant-checkpoint display

While training, every significant checkpoint prints a **live banner** the first time
the fleet reaches it, tiered by importance:

```
========================================================================
>>>   MAJOR CHECKPOINT REACHED:  BEAT BROCK (BOULDER BADGE)
     first reach at training step 1,284,096 (env 7, episode step 5031)
     badges=1  party_levels_sum=34  progress=9/33
========================================================================
```

- `*** LEGENDARY` — Elite Four, Champion
- `>>> MAJOR`     — gym badges, major dungeons (Rocket Hideout, Silph Co., ...)
- `[+] KEY`       — key items / Pokédex / **SS Ticket**
- ` ->  step`     — reaching a new town/route

Periodic summaries print the furthest milestone, badge/level bests, and the next few
objectives. Two durable files are written into the session dir:

- `progress.json`        — live dashboard: furthest milestone, % complete, best
  badges/levels, per-milestone first-reach step and reach counts.
- `milestones_log.jsonl` — append-only history of **every** checkpoint reach
  (timestep, env, badges, levels) — survives restarts.

(Markers are ASCII on purpose so they never crash the Windows console.)

### Save & resume — never lose progress

Training **auto-resumes by default** (`--resume auto`). If a run stops (crash, power
loss, Ctrl-C) just re-run the *same command* and it continues:

- It finds the newest `poke_<stage>_<steps>_steps.zip` in the session dir and loads
  it, preserving the global timestep counter (checkpoint numbering stays continuous).
- Checkpoints are written every `--save-freq` timesteps (default 200k).
- In `--staged` mode, finished stages are recorded in `stage_progress.json` and
  **skipped** on the next run; the next stage warm-starts from the last completed
  stage's model. An interrupted stage resumes from *its own* latest checkpoint.
- Successful emulator snapshots per milestone are saved under `curriculum_states/`
  for warm-starting later stages.

Force a fresh start with `--resume ""`, or resume an explicit file with
`--resume runs_curriculum/poke_fullgame_4000000_steps`.

The legacy `baseline_fast_v2.py` also auto-resumes now: it picks up the newest
`poke_*_steps.zip` in `runs/` and continues with a continuous step counter.

---

## 0b. Getting *past* the baseline wall (SS Anne) — robustness & efficiency

The stock baseline reliably stalls around the SS Anne because (1) every episode
restarts in Pallet Town, so almost none of the training experience lands near the
frontier, and (2) progress there is gated behind precise menu/field-move sequences
that exploration rewards don't shape. These features attack both directly:

**Go-Explore warm-restart (`start_state_prob`, default 0.5).**
Even without `--staged`, each episode now has a configurable chance to restart from a
saved *frontier* success-state (the furthest milestones that have snapshots), instead
of always from `init.state`. As the agent reaches deeper milestones, more of its
rollout is spent *at the wall* rather than re-walking the early game — the single
biggest lever for crossing a far-from-start checkpoint. It is safe from a cold start
(falls back to `init.state` until any snapshots exist) and reward accounting is
re-anchored on the warm state so pre-earned progress isn't re-paid. `frontier_window`
controls how many of the furthest states are sampled from.

**Stalled-episode truncation (`max_steps_without_progress`, default 4000).**
Episodes that go this many steps with no new map/event/badge/level are truncated, so
PPO stops burning rollout on a dead state. Paired with Go-Explore restart, the freed
samples get re-seeded near the frontier.

**Deterministic Cut assist (`auto_use_cut`, default OFF — experimental).**
The classic hard wall just past the SS Anne is *using HM01 Cut on the blocking tree*:
a multi-step field-move menu sequence PPO rarely discovers. With `auto_use_cut: true`,
after the agent bumps a wall `cut_trigger_bumps` times in a row while Cut is actually
usable (a party member knows Cut **and** the Cascade Badge is owned — checked from
RAM), the env executes a deterministic Cut macro (`ScriptedHelpers.field_use_cut`,
which reads which party slot knows Cut so the cursor navigation is correct). It is a
safe no-op when Cut isn't usable. **Marked experimental:** the party field-move submenu
cursor offset is ROM/party-dependent — validate it (e.g. in `--mode visual`) before
relying on it. This is the lever that turns "probably crosses, with risk" into "yes".

**Training stability.** The PPO trainer now uses a linearly-decayed learning rate and
clip range (`--lr`, plus `--ent-coef` to dial exploration), `vf_coef=0.5`,
`max_grad_norm=0.5` — steadier behaviour over the long runs mid-game progress needs.

Recommended command to push past SS Anne (Go-Explore on via the config defaults):

```bash
# whole-game run; Go-Explore + stalled-truncation come from configs/curriculum.yaml
python train_full_game_curriculum.py --mode headless --num-envs 16

# or the focused, fastest route: staged through the SS-Anne region, then enable the
# Cut assist for the tree (edit configs/curriculum.yaml: auto_use_cut: true)
python train_full_game_curriculum.py --staged --num-envs 16 \
    --milestones help_bill,reach_vermilion,get_cut,beat_lt_surge
```

New RAM accessors backing this: `GameState.party_knows_move`, `mon_index_with_move`,
`can_use_cut` ([ram_map.py](curriculum/ram_map.py)).

---

## 1. Why this exists

Pokémon Red is a long-horizon, sparse-reward game with brutal credit assignment. A
single monolithic agent struggles to reach the credits. This extension follows the
report's recommended structure:

- **Structured observation** (RAM/events first, pixels optional) → sample-efficient.
- **Milestone curriculum** → the 25-hour quest becomes ~33 small, detectable goals.
- **Dense, shaped, configurable rewards** with explicit anti-reward-hacking penalties.
- **Saved success-states** so later stages warm-start from where earlier stages ended.
- **Skill/option scaffolding** so you can grow from one PPO policy to true HRL later.

## 2. Backwards compatibility

This extension is **purely additive**. It does not modify `red_gym_env_v2.py`,
`baseline_fast_v2.py`, or any other existing file. `CurriculumRedGymEnv` *subclasses*
`RedGymEnv`. Your existing training (`python baseline_fast_v2.py`) is unaffected.

## 3. Layout

```
v2/
  curriculum/
    ram_map.py          # GameState: single source of truth for reading RAM
    structured_obs.py   # flat normalized observation vector + gym space
    milestones.py       # the ~33-milestone curriculum graph + MilestoneManager
    rewards.py          # configurable, logged reward system + anti-loop tracker
    scripted_helpers.py # deterministic dialogue/heal/shop/use macros
    skills.py           # option-policy / skill scaffold + high-level controller stub
    state_store.py      # save/load successful emulator snapshots per milestone
    config.py           # CurriculumConfig dataclasses + YAML/JSON loader
  curriculum_env.py     # CurriculumRedGymEnv (subclass of RedGymEnv) — the glue
  configs/curriculum.yaml
  train_full_game_curriculum.py   # staged PPO training driver (headless/visual)
  run_curriculum_interactive.py   # watch/evaluate (random or trained policy)
  requirements-curriculum.txt     # adds PyYAML (for YAML configs)
```

## 4. Setup

```bash
cd v2
pip install -r requirements.txt
pip install -r requirements-curriculum.txt   # PyYAML, for YAML configs
# place your legally-obtained ROM at the repo root:
#   PokemonRedExperiments/PokemonRed.gb
```

Linux/macOS are the priority; the code is OS-agnostic Python and also runs on Windows.

## 5. Running

### Headless training (default, fast)
No emulator window, `SubprocVecEnv`, many parallel envs, unlimited speed:

```bash
python train_full_game_curriculum.py --mode headless --num-envs 16 --speed 0
```

This trains a single shared PPO policy on the **whole-game** curriculum reward (every
milestone grants its one-time bonus). It's the simplest working entry point.

### Visual mode (watch one agent)
```bash
python run_curriculum_interactive.py --mode visual --speed 3
# or evaluate a trained model:
python run_curriculum_interactive.py --mode visual --model runs_curriculum/poke_fullgame_best
```

### Explicit staged curriculum
Walk milestones one stage at a time, warm-starting each stage from the previous
milestone's saved success states, saving the best model per milestone:

```bash
python train_full_game_curriculum.py --staged --num-envs 16 --steps-per-stage 2000000
# subset of stages:
python train_full_game_curriculum.py --staged --milestones get_starter,reach_pewter,beat_brock
```

### Train one milestone
```bash
python train_full_game_curriculum.py --target-milestone beat_brock --num-envs 8
```

All flags: see `python train_full_game_curriculum.py --help`.

## 6. Observation space

`CurriculumRedGymEnv` exposes a `Dict` observation:

- `structured` *(primary)* — a normalized `float32` vector (~162 dims) built by
  `StructuredObservationBuilder` from `GameState` + per-episode signals: map id,
  prev map, x/y, badges (+count), event-flag sum, party species/levels/HP/status,
  party size, total HP, battle state, enemy species/level/HP, money, poké-ball count,
  Cut/Surf/Strength ownership, HM/TM counts, key items (parcel, SS ticket, town map,
  bike), Pokédex owned/has-dex, heal/blackout/poké-center signals, stuck/loop/
  inactivity/wall-bump scores, recent action & map history, and milestone bits.
- `screens` *(optional auxiliary)* — the base 72×80×3 grayscale frame-stack. Disable
  with `use_screens: false` to train RAM-only (faster).

The builder is schema-driven and asserts its output length matches the declared
space, so the two can never silently drift.

## 7. Milestones

`curriculum/milestones.py` defines an ordered list (Start Game → Get Starter → … →
Beat Champion). Each milestone has:

- a **detector** `(GameState, ctx) -> bool` using event flags (`events.json`),
  badge state, and/or map ids (`map_data.json`),
- a one-time **reward**, a **timeout**, **allowed start states** (previous milestone's
  snapshots), a **next** pointer, and a **save_state** flag.

`MilestoneManager` latches completion (once done, stays done), maintains a visited-map
context, and reports the active "frontier" for staging.

> **Detector accuracy note.** Most detectors use precise named event flags or badge
> bits. A few dungeon-completion milestones (`complete_rocket_hideout`,
> `complete_silph_co`) use "reached the deepest known floor" as a robust proxy and are
> commented as such. Refine these with exact event flags if you want stricter gating —
> the system is built so swapping a detector is a one-line change.

## 8. Reward system & anti-reward-hacking

`curriculum/rewards.py` (`RewardConfig` + `RewardManager`) computes **incremental**
per-step rewards and keeps cumulative totals for logging. Positive: milestones
(one-time), new maps, new coordinates, story events (max-based), badges (max-based),
level growth (diminishing returns), Pokédex, healing. Penalties: blackouts, standing
on over-visited tiles (stuck), wall bumps, menu spam, region loops, inactivity, and
repeated healing without progress (PC-farming).

Key anti-farming properties:
- Milestone / badge / event / level / Pokédex rewards are **one-time or max-based** —
  they cannot be farmed by revisiting.
- Warm-started episodes call `RewardManager.baseline(gs)` so already-satisfied progress
  isn't re-rewarded on step 1.
- The `AntiLoopTracker` produces the stuck/loop/inactivity signals that feed both the
  penalties and the observation.

Every reward component is surfaced via `info["reward_components"]` and the env's
`agent_stats`, so the existing `TensorboardCallback` and `MilestoneCallback` log them.

## 9. Curriculum state store

`curriculum/state_store.py` saves PyBoy `.state` snapshots (never ROMs) under
`curriculum_states/<milestone_key>/` whenever a milestone with `save_state=True`
completes. Reward is encoded in the filename, so selection (`random`/`latest`/`best`)
is a lock-free directory scan — safe across `SubprocVecEnv` workers. Later stages
warm-start from these via `allowed_start_states`.

## 10. Scripted helpers & skills (hybrid RL)

`scripted_helpers.py` provides deterministic button macros (advance dialogue, heal at a
counter, confirm a purchase, use/teach an item) so menus never hard-block exploration.
Enable with `use_scripted_helpers: true` / `auto_advance_dialogue: true`. The RL agent
still makes the strategic/exploration decisions.

`skills.py` is forward-looking scaffolding: a `Skill` option interface,
`ScriptedSkill`/`PolicySkill` implementations, a `SkillRegistry`, and a rule-based
`HighLevelController` stub. You can start with one PPO policy today and grow into true
hierarchical control (or an LLM-guided manager) without reworking the env.

## 11. Configuration

Everything is configurable via `configs/curriculum.yaml` (or a `.json` equivalent),
loaded by `CurriculumConfig`. CLI flags override the file. See the YAML comments for
each field. To disable a reward component, set its weight to `0`.

## 12. Suggested workflow toward full-game completion

1. Start with whole-game shared-policy headless training to get an agent making
   early-game progress and **collecting success snapshots** automatically.
2. Switch to `--staged` to harden later milestones using those snapshots as warm
   starts; save the best model per milestone.
3. Tighten reward weights / detectors as you observe reward-hacking (the expected
   iterative loop: train → observe exploit → patch reward → repeat).
4. When ready, add concrete skills in `skills.py` and a learned/LLM manager for HRL.
