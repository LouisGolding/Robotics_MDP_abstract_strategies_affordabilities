# Robotics MDP Abstract Strategies & Affordabilities

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

### Affordances (Condition 3)

An affordance associates a state with a *ranked list of applicable
strategies* scored on two dimensions:

1. **Utility** — how beneficial is this strategy from the current state
   (its basin of attraction; follows Khen's deterministic formulation).
2. **Reliability** — how likely is the strategy to be successfully grounded
   and executed under stochastic transitions (our novel contribution).

The combined affordance score determines which strategies are passed to the
MCTS planner as candidate macro-moves.  This is **not** action filtering;
it is strategy-level selection operating above the planner.

---

## Development Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 1 | GridWorld env + online loop + ReliablePathPlanner oracle | ✅ Done |
| 2 | MCTSPlanner (UCT) + primitive actions — "MDP alone" baseline; validate against Dijkstra oracle | ⬜ Next |
| 3 | Macro-actions / options — second evaluation condition | ⬜ |
| 4 | Abstract strategies + affordance scores (utility × reliability) — third condition | ⬜ |
| 5 | Evaluation: three-way comparison across all conditions | ⬜ |

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
│   ├── planner.py               # InnerPlanner interface
│   │                            #   ReliablePathPlanner  (oracle, Phase 1)
│   │                            #   OnlineReplanningAgent (outer loop, all phases)
│   │                            #   MCTSPlanner          (Phase 2 — all eval conditions)
│   ├── macro_actions.py         # Macro-actions / options (Phase 3)
│   ├── strategies.py            # Abstract strategy policies (Phase 4)
│   └── affordances.py           # Affordance scoring: utility × reliability (Phase 4)
│
└── tests/
    ├── __init__.py
    └── test_baseline.py         # Phase 1: oracle validation experiments
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
python tests/test_baseline.py
```
