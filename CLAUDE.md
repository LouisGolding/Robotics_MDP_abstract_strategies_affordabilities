# Robotics MDP Abstract Strategies & Affordabilities

## Project Overview

Research codebase implementing an MDP-based navigation framework that uses
**abstract strategies** and **affordances** to improve planning efficiency.
The key idea: instead of planning purely over primitive actions, the agent
reasons about macro-level *strategies* (sequences of actions that achieve
sub-goals) and uses *affordances* (perceptual signals about what actions are
available/viable) to prune the search space.

---

## Research Motivation

Standard MDP planners replan from scratch on every failure. This is expensive
when the environment has many stochastic doors and long paths. The hypothesis is
that:

1. **Macro-actions / options** allow the planner to reason at a coarser
   granularity, reducing the branching factor.
2. **Abstract strategies** (goal-conditioned policies over macro-actions)
   further compress the plan representation.
3. **Affordances** allow the agent to filter out actions that are implausible
   *before* running the search, cutting planning time.

---

## Development Roadmap

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Environment + baseline primitive-action MDP planner | ✅ Done |
| 2 | Macro-actions / options (hallway traversal, room sweep) | ⬜ Todo |
| 3 | Abstract strategies (goal-conditioned strategy policies) | ⬜ Todo |
| 4 | Affordance module (filter unavailable/implausible actions) | ⬜ Todo |
| 5 | Evaluation & comparison across conditions | ⬜ Todo |

---

## Repo Structure

```
Robotics_MDP_abstract_strategies_affordabilities/
├── CLAUDE.md                    # This file
├── README.md
├── requirements.txt             # networkx, matplotlib, numpy
│
├── mdp_nav/                     # Core library
│   ├── __init__.py
│   ├── environment.py           # GridWorld: graph-based grid-world (Phase 1)
│   ├── planner.py               # MDPPlanner: baseline online replanning (Phase 1)
│   ├── macro_actions.py         # Macro-actions / options (Phase 2)
│   ├── strategies.py            # Abstract strategy policies (Phase 3)
│   └── affordances.py           # Affordance filtering module (Phase 4)
│
└── tests/
    ├── __init__.py
    └── test_baseline.py         # Phase 1: 5 experiments, prints metrics
```

---

## Phase 1 Design Details

### Environment (`mdp_nav/environment.py`)

- **`GridWorld`** — graph-based grid-world backed by `networkx.Graph`.
  - Rooms = nodes `(row, col)`; doors = edges with attribute `prob` ∈ (0,1].
  - On `step(target)`: samples Bernoulli(prob); on failure the door is
    permanently closed (`failed_doors` set) for this episode.
  - `reset()` clears `failed_doors` and returns agent to `start`.
  - `available_doors(node)` returns `[(neighbour, prob), ...]` for open doors.
  - `render_text()` — ASCII grid. `render_matplotlib()` — networkx drawing.
  - `cluster_size > 1` tiles the world with modular `cs×cs` sub-grids plus
    bridge edges, enabling repeatable substructures.

- **`DoorProbabilitySpec`** — priority-based prob assignment:
  1. `per_door` dict (canonical edge key)
  2. `default` fallback

### Planner (`mdp_nav/planner.py`)

- **`MDPPlanner`** — online replanning, primitive actions only.
  - Enumerates all simple paths to any goal (via `nx.all_simple_paths`).
  - Scores paths by cumulative product of edge probs (ignoring failed doors).
  - Picks highest-prob path with prob ≥ α; executes first action; replans on failure.
  - Tracks `actions_taken`, `replans`, `reached_goal` per episode.

### Test Script (`tests/test_baseline.py`)

Five experiments:
1. 5×5 uniform p=0.8
2. 5×5 uniform p=0.5, alpha=0.3
3. 5×5 with bottleneck corridor (per-door overrides)
4. 6×6 clustered (3×3 tiles)
5. 8×8 uniform p=0.85, alpha=0.6

Each prints: success rate, avg replans, avg actions.

---

## Running

```bash
pip install -r requirements.txt
python tests/test_baseline.py
```

---

## Key Design Principles

- **Separation of concerns**: environment owns stochastic transitions;
  planner owns search strategy. Later phases will add strategy/affordance layers
  *without* modifying the environment.
- **Reproducibility**: every `GridWorld` and `MDPPlanner` call accepts a `seed`.
- **Extensibility**: `MDPPlanner._best_path` is the natural extension point for
  macro-action planning in Phase 2.
