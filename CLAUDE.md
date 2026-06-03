# Robotics MDP Abstract Strategies & Affordabilities

## Project Overview

Research codebase for a Master's thesis investigating whether **abstract
strategies** and **affordances** reduce planning cost in a stochastic
navigation MDP.  The core hypothesis: by reasoning at a coarser level of
abstraction (macro-actions, strategies) and by filtering implausible actions
before search (affordances), the planner should need fewer decisions, fewer
replans, and less wall-clock time than a primitive-action baseline.

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
    plan ← InnerPlanner.plan(current_state, env)
    if plan is None: give up
    execute plan[1]                    # one step only
    observe outcome (success / locked)
    if door locked: replan             # loop back, plan from new state
```

Advantages over offline value iteration here:
- The available-action set changes as doors lock → full-policy computation
  would need to be repeated; the online loop handles this naturally.
- Scales to large maps without enumerating all states.
- Clean separation: the outer loop never changes; only the inner planner
  is swapped between experimental conditions.

**Reference**: Yoon, S., Fern, A. & Givan, R. (2007). *FF-Replan: A baseline
for probabilistic planning.* ICAPS 2007.

---

## Architecture: Swappable Inner Planner

The outer loop (`OnlineReplanningAgent`) is decoupled from how plans are
produced via the `InnerPlanner` abstract interface:

```
InnerPlanner  (abstract)
├── ReliablePathPlanner   ← Phase 1  (scaffolding baseline)
└── MCTSPlanner           ← Phase 2+ (target algorithm)
```

**All three experimental conditions** (primitives / +macro-actions /
+strategies) share the **same outer loop and inner planner interface**.
Only the set of available moves passed to the inner planner differs.

### Current inner planner: ReliablePathPlanner (scaffolding)

Dijkstra on edge weights `−log(p)` (Dijkstra, 1959).  The shortest path
under these weights maximises cumulative success probability.

```
weight(u→v) = −log(p_uv)
most-reliable path = Dijkstra shortest path
cumulative prob    = exp(−total_weight) = ∏ p_i
```

Returns the path only if cumulative prob ≥ α (confidence threshold).
This is **scaffolding** — a correct, simple baseline that validates the
online loop end-to-end on small maps.  It is not the thesis contribution.

**Reference**: Dijkstra, E. W. (1959). A note on two problems in connexion
with graphs. *Numerische Mathematik*, 1(1), 269–271.

### Target inner planner: MCTS/UCT (Phase 2+)

Monte Carlo Tree Search with UCB1 (Kocsis & Szepesvári, ECML 2006).  From
the current state, UCT builds a partial lookahead tree via rollouts, trading
off exploration vs. exploitation.  In later phases, abstract strategies will
bias tree expansion, reducing the number of rollouts required.

**Reference**: Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo
planning. *ECML 2006*, LNCS vol 4212.

---

## Development Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 1 | GridWorld env + online loop + ReliablePathPlanner (scaffolding) | ✅ Done |
| 2 | Macro-actions / options + MCTSPlanner inner planner | ⬜ Next |
| 3 | Abstract strategies (bias MCTS expansion) | ⬜ |
| 4 | Affordance module (filter actions before planning) | ⬜ |
| 5 | Evaluation: compare all conditions | ⬜ |

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
│   │                            #   ReliablePathPlanner  (Phase 1 scaffolding)
│   │                            #   OnlineReplanningAgent (outer loop, all phases)
│   │                            #   MCTSPlanner          (Phase 2, TODO)
│   ├── macro_actions.py         # Macro-actions / options (Phase 2)
│   ├── strategies.py            # Abstract strategy policies (Phase 3)
│   └── affordances.py           # Affordance filtering (Phase 4)
│
└── tests/
    ├── __init__.py
    └── test_baseline.py         # Phase 1: 5 experiments, prints metrics
```

---

## Key Design Principles

1. **Online only** — the outer loop replans from the current state after each
   action.  No full offline policy is computed.

2. **Separation of concerns** — `GridWorld` owns stochastic transitions;
   `OnlineReplanningAgent` owns the outer loop; `InnerPlanner` owns search.

3. **Swappable inner planner** — `ReliablePathPlanner` → `MCTSPlanner` is a
   one-line swap; the outer loop, environment, and metrics are unchanged.

4. **Shared loop across conditions** — primitive / macro-action / strategy
   conditions all use `OnlineReplanningAgent`; only the available moves differ.

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
