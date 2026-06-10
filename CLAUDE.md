# Robotics MDP Abstract Strategies & Affordabilities

> **For Claude Code sessions** — canonical decisions in this section override everything below.

## What this project is

MSc thesis by Louis Golding, supervised by Dr. Khen Elimelech.

We're extending abstract strategies and affordances (from Khen's published research on deterministic planning) into **non-deterministic / MDP environments**. We build on the *concepts* from the papers — we do NOT have access to their code. Everything here is our own implementation.

**Core research question:** assuming we already have strategies, how do we integrate them into MDP planning so it becomes cheaper (fewer replans / lower total cost)?

## Canonical decisions (if anything contradicts these, these win)

**Environment.** Static, non-deterministic navigation with repeatable structure. Rooms connected by doors. Each door has a known open-probability. **One attempt per door** — no retries (retries would make the env dynamic). No keys. No temporally-extended goals. Goal = reach a target state with confidence ≥ α.

**Approach.** Online MDP planning. We implement our own planner (MCTS-style, off-the-shelf is fine, need not be SOTA). We integrate strategy/affordance concepts from the papers into it. Plan → act → observe → replan. Strategies replace random expansion with experience-informed expansion.

**Scope.** Strategies are assumed given (no grounding/reconstruction for now). Stay at task-level abstraction. Lifelong library is the vision but start with a fixed set.

**Contribution.** (1) Qualitative: strategies under uncertainty — applying a strategy no longer guarantees reaching its end state. (2) Quantitative: extend the affordance vector with a **reliability score** (not just usefulness).

**Evaluation.** Three-way comparison: MDP alone vs. MDP + macro-actions vs. MDP + abstract strategies. Success = fewer replans + lower total cost. Macro-actions (sequences of *actions*) are the middle comparison; strategies (sequences of *states*) are ours — with states we still know where we're going.

## Key concepts (from the papers)

- **Abstract strategy** = a road map (sequence of *states* in abstract space) + an abstraction key (projection/reconstruction functions). More general than macro-actions.
- **Affordance** = predictive numeric vector estimating the benefit of using a strategy from the current state. Scores: start affordance (effort to reach strategy's first state), strategy affordance (effort to traverse the road map), task affordance (remaining effort to goal). We add: reliability affordance.
- **ACSD** = abstraction-critical state detection. How strategies are extracted from a single execution trace.
- **Two senses of "success probability"** — don't conflate: (1) probability a plan succeeds in the MDP (from transition probabilities), (2) probability a strategy can be grounded into actions (refine; if can't, abort and backtrack).

## Full project log

The detailed meeting notes, background, and changelog live in the Claude.ai project knowledge base (`PROJECT_LOG.md`). This file is the compact version for Claude Code sessions.

## Development notes

<!-- Append implementation decisions, algorithm choices, experiment parameters here as we go -->

---

## Project Overview

Research codebase for a Master's thesis investigating whether **abstract
strategies** and **affordances** reduce planning cost in a stochastic
navigation MDP.  The core hypothesis: by reasoning at a coarser level of
abstraction (macro-actions, strategies) and by using affordance scores to
select applicable strategies, the planner should need fewer decisions, fewer
replans, and less wall-clock time than a primitive-action baseline.

**What affordances are here.**  An affordance is a *predictive score* of how
beneficial a given strategy is from the current state — it defines a
strategy's basin of attraction and is used to rank and select which strategies
are applicable.  Our contribution adds a **reliability dimension** to this
score: not just "how useful is this strategy?" but "how reliably can it be
grounded and executed under transition uncertainty?"  Affordances do *not*
filter primitive actions before search; they operate at the strategy-selection
level, above the MCTS planner.

**What the three evaluation conditions are.**  All three use the *same* MCTS
inner planner and the *same* online outer loop.  Only the set of available
moves differs:

| Condition | Available moves | What changes |
|-----------|----------------|--------------|
| MDP alone | Primitive door actions | Baseline — MCTS over raw actions |
| + Macro-actions | Primitive + macro-actions (options) | Coarser action space |
| + Strategies + Affordances | Macro-actions, guided by affordance-ranked strategies | Strategy selection biases MCTS expansion |

---

## Problem Formulation

**Stochastic Shortest-Path MDP** (Bertsekas & Tsitsiklis, 1991):

| Component   | Definition |
|-------------|------------|
| States S    | Rooms — graph nodes (row, col) |
| Actions A   | Attempt a door to an adjacent room |
| Transition  | Door (s→s') opens with prob p ∈ (0,1]; on failure the door is **permanently closed** for this episode |
| Cost        | 1 per action (unit step cost) |
| Goal        | Reach any node in G ⊆ S |

The permanently-closing door is the key stochastic element: the agent learns
about transition availability only by attempting it.

---

## Solver: Online Plan-Act-Observe-Replan

We do **NOT** compute a full offline policy.  Instead we use an **online
replanning loop** (Yoon, Fern & Givan, ICAPS 2007):

```
while not at goal:
    plan ← MCTSPlanner.plan(current_state, env)   # inner planner = MCTS/UCT
    if plan is None: give up
    execute plan[1]                                 # one step only
    observe outcome (success / locked)
    if door locked: replan                          # loop back, plan from new state
```

Advantages over offline value iteration for this setting:
- The available-action set changes as doors lock → full-policy computation
  would need to be repeated; the online loop handles this naturally.
- Scales to large maps without enumerating all states.
- Clean separation: the outer loop never changes; only the available moves
  passed to the inner planner differ between experimental conditions.

**Reference**: Yoon, S., Fern, A. & Givan, R. (2007). *FF-Replan: A baseline
for probabilistic planning.* ICAPS 2007.

---

## Architecture: Swappable Inner Planner

The outer loop (`OnlineReplanningAgent`) is decoupled from how plans are
produced via the `InnerPlanner` abstract interface:

```
InnerPlanner  (abstract)
├── ReliablePathPlanner   ← scaffolding / validation oracle ONLY (not an eval condition)
└── MCTSPlanner           ← the planner used in ALL three experimental conditions
```

**All three experimental conditions** share the **same MCTSPlanner and the
same outer loop**.  Only the set of available moves passed to the planner
differs.

### Scaffolding oracle: ReliablePathPlanner

Dijkstra on edge weights `−log(p)` (Dijkstra, 1959).  Finds the
most-reliable (maximum cumulative success probability) path exactly.

```
weight(u→v) = −log(p_uv)
most-reliable path = Dijkstra shortest path
cumulative prob    = exp(−total_weight) = ∏ p_i
```

**Role**: validation oracle only.  Used in unit tests to verify that
`MCTSPlanner` converges to the same path on small maps where the exact
optimum is known.  It is **not** one of the three evaluation conditions and
does not appear in thesis results.

**Reference**: Dijkstra, E. W. (1959). A note on two problems in connexion
with graphs. *Numerische Mathematik*, 1(1), 269–271.

### Evaluation inner planner: MCTSPlanner / UCT

Monte Carlo Tree Search with UCB1 (UCT; Kocsis & Szepesvári, ECML 2006).
From the current state, UCT builds a partial lookahead tree via repeated
rollouts, balancing exploration vs. exploitation.

- **Condition 1 (MDP alone):** MCTS over primitive door actions.
- **Condition 2 (+macro-actions):** MCTS over primitive + macro-action moves.
- **Condition 3 (+strategies + affordances):** MCTS expansion biased by
  affordance-ranked strategies; the affordance score (utility × reliability)
  determines which strategies are offered as candidate moves.

**Reference**: Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo
planning. *ECML 2006*, LNCS vol 4212.

### Affordances (Phase 5)

An affordance is a **predictive score of how beneficial a stored strategy is
from the current state**.  We maintain a library of strategies (road maps —
sequences of states that worked before).  At each decision point the affordance
module scores every candidate strategy and selects the best one(s) to bias
MCTS tree expansion toward.  This defines each strategy's *basin of attraction*:
the set of states from which that strategy is a good choice.

The affordance is represented as a small vector (lower = better, 0 ideal):

| Component | Meaning |
|-----------|---------|
| `start_aff` | Effort to reach the strategy's entry state from the current state |
| `strategy_aff` | Effort to traverse / refine the strategy's road map |
| `task_aff` | Remaining effort to the goal after the strategy completes |
| `reliability_aff` | **Our contribution** — how likely the strategy can be grounded into real moves given that doors may be locked |

Affordances do **not** filter or prune primitive door-actions.  They operate
entirely at the strategy-selection level, above the MCTS planner.  The MCTS
planner still decides how to execute the selected strategy via its normal
tree search over available moves.

---

## Development Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 1 | GridWorld env + online loop + ReliablePathPlanner oracle | ✅ Done |
| 2 | MCTSPlanner (UCT) + primitive actions — "MDP alone" baseline; validate against Dijkstra oracle | ✅ Done |
| 2b | Evaluation harness (`tests/evaluate.py`) — fixed benchmark maps, side-by-side comparison table | ✅ Done |
| 3 | Macro-actions / options — second evaluation condition | ⬜ Next |
| 4 | Abstract strategies — third evaluation condition | ⬜ |
| 5 | Affordance module — rank/select applicable strategies; add reliability score | ⬜ |
| 6 | Final evaluation run across all conditions | ⬜ |

---

## Repo Structure

```
Robotics_MDP_abstract_strategies_affordabilities/
├── CLAUDE.md                    # This file — project guide & design notes
├── README.md
├── requirements.txt             # networkx, matplotlib, numpy
│
├── mdp_nav/                     # Core library
│   ├── __init__.py              # Public API
│   ├── environment.py           # GridWorld, DoorProbabilitySpec (Phase 1)
│   │                            #   DoorProbabilitySpec: min_prob + per_door overrides
│   │                            #   per-door probs drawn from Uniform[min_prob, 0.99]
│   ├── planner.py               # InnerPlanner interface
│   │                            #   ReliablePathPlanner  (oracle, Phase 1)
│   │                            #   OnlineReplanningAgent (outer loop, all phases)
│   │                            #     tracks: reached_goal, actions_taken,
│   │                            #             replans, planning_time
│   │                            #   MCTSPlanner / UCT    (Phase 2 — all eval conditions)
│   ├── macro_actions.py         # Macro-actions / options (Phase 3)
│   ├── strategies.py            # Abstract strategy policies (Phase 4)
│   └── affordances.py           # Affordance scoring: rank/select strategies, incl. reliability (Phase 5)
│
└── tests/
    ├── __init__.py
    ├── test_baseline.py         # Phase 1: ReliablePathPlanner smoke tests
    ├── test_mcts.py             # Phase 2: MCTSPlanner validation + episode experiments
    └── evaluate.py              # PRIMARY EVALUATION — three-way comparison table
                                 #   CONDITIONS registry: add Phase 3/4/5 conditions here
                                 #   BENCHMARK_MAPS: 4 fixed maps, seeded, reproducible
                                 #   Metrics: success rate, avg replans, avg actions, planning time
```

---

## Key Design Principles

1. **Online only** — the outer loop replans from the current state after each
   action.  No full offline policy is computed.

2. **Separation of concerns** — `GridWorld` owns stochastic transitions;
   `OnlineReplanningAgent` owns the outer loop; `InnerPlanner` owns search.

3. **MCTS is the single evaluation planner** — `ReliablePathPlanner` is an
   oracle for testing only.  All three conditions use `MCTSPlanner`; only
   the available move set differs.

4. **Affordances operate above the planner** — they rank strategies, not
   primitive actions.  The reliability dimension is the novel contribution.

5. **Reproducibility** — every `GridWorld` and agent call accepts a `seed`.

---

## Key References

- Bertsekas, D. P. & Tsitsiklis, J. N. (1991). An analysis of stochastic
  shortest path problems. *Mathematics of Operations Research*, 16(3).
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
  *Numerische Mathematik*, 1(1), 269–271.
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo planning.
  *ECML 2006*, LNCS vol 4212.
- Koenig, S. & Likhachev, M. (2002). D* Lite. *AAAI 2002*.
- Puterman, M. L. (1994). *Markov Decision Processes*. Wiley.
- Yoon, S., Fern, A. & Givan, R. (2007). FF-Replan: A baseline for
  probabilistic planning. *ICAPS 2007*.

---

## Running

```bash
pip install -r requirements.txt

# Smoke tests (fast)
python tests/test_baseline.py   # Phase 1: ReliablePathPlanner oracle
python tests/test_mcts.py       # Phase 2: MCTSPlanner validation

# Primary evaluation — three-way comparison table (~2 min)
python tests/evaluate.py
```

## Evaluation harness (`tests/evaluate.py`)

This is the primary tool for the thesis results.  It runs all active
conditions on a fixed set of 4 benchmark maps and prints a comparison table:

```
Map : 5×5  easy   (min_prob=0.75)   (100 episodes each)
──────────────────────────────────────────────────────────────────
Condition                    Success   Replans   Actions   Plan(s)
──────────────────────────────────────────────────────────────────
MDP alone (primitive)          88.0%      2.42     17.44     0.294
+ Macro-actions               TBD        TBD       TBD       TBD     ← Phase 3
+ Strategies                  TBD        TBD       TBD       TBD     ← Phase 4+5
──────────────────────────────────────────────────────────────────
```

**To add a new condition** (Phases 3/4/5): implement a factory function
`(env: GridWorld) -> OnlineReplanningAgent` and append it to the
`CONDITIONS` list in `evaluate.py`.  Nothing else needs to change.

**Benchmark maps** are fixed seeds so every condition sees the same doors
and the same per-door probabilities.  Door probabilities are drawn from
`Uniform[min_prob, 0.99]` — heterogeneous and realistic, not uniform.

**Metrics:**
- `Success` — % of episodes where the agent reached the goal
- `Replans` — avg number of times a door failure forced replanning
- `Actions` — avg total door attempts (including failed ones)
- `Plan(s)` — avg total wall-clock time spent inside `plan()` per episode
