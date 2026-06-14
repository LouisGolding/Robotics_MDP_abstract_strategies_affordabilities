"""
Three-way evaluation harness.

Runs all implemented planning conditions on a fixed set of benchmark maps
and prints a side-by-side comparison table.  This is the primary tool for
assessing the thesis hypothesis:

  MDP alone  <  MDP + macro-actions  <  MDP + strategies + affordances

in terms of success rate, replans, actions taken, and planning time.

Conditions
----------
1. MDP alone         — MCTSPlanner over primitive door actions (Phase 2)
2. + Macro-actions   — MCTSPlanner with macro-actions added (Phase 3)
3. + Strategies      — MCTSPlanner biased by affordance-ranked strategies (P4+5)

Each condition uses the SAME MCTSPlanner and the SAME OnlineReplanningAgent
outer loop.  Only the set of available moves differs — keeping the comparison
clean and the contribution isolated.

Benchmark maps
--------------
Four fixed maps covering different difficulty profiles.  All use seeded,
random per-door probabilities (Uniform[min_prob, 0.99]) for realism.
The same seed → same map across all conditions, so differences in outcomes
are attributable to the planning condition, not map randomness.

Usage
-----
    python tests/evaluate.py
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import (
    GridWorld,
    DoorProbabilitySpec,
    make_mcts_agent,
    make_macro_agent,
    TraceRecorder,
)
from mdp_nav.planner import OnlineReplanningAgent


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    label: str
    successes: int = 0
    total_replans: float = 0.0
    total_actions: float = 0.0
    total_decisions: float = 0.0
    total_planning_time: float = 0.0
    n_episodes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_episodes * 100 if self.n_episodes else 0.0

    @property
    def avg_replans(self) -> float:
        return self.total_replans / self.n_episodes if self.n_episodes else 0.0

    @property
    def avg_actions(self) -> float:
        return self.total_actions / self.n_episodes if self.n_episodes else 0.0

    @property
    def avg_decisions(self) -> float:
        return self.total_decisions / self.n_episodes if self.n_episodes else 0.0

    @property
    def avg_planning_time(self) -> float:
        return self.total_planning_time / self.n_episodes if self.n_episodes else 0.0


# ---------------------------------------------------------------------------
# Benchmark map definitions
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMap:
    name: str
    rows: int
    cols: int
    min_prob: float
    seed: int
    cluster_size: int = 1
    per_door: Dict = field(default_factory=dict)

    def make_env(self) -> GridWorld:
        return GridWorld(
            rows=self.rows,
            cols=self.cols,
            spec=DoorProbabilitySpec(min_prob=self.min_prob, per_door=self.per_door),
            seed=self.seed,
            cluster_size=self.cluster_size,
        )


BENCHMARK_MAPS: List[BenchmarkMap] = [
    BenchmarkMap(
        name="5×5  easy   (min_prob=0.75)",
        rows=5, cols=5, min_prob=0.75, seed=1,
    ),
    BenchmarkMap(
        name="5×5  medium (min_prob=0.50)",
        rows=5, cols=5, min_prob=0.50, seed=2,
    ),
    BenchmarkMap(
        name="5×5  bottleneck",
        rows=5, cols=5, min_prob=0.60, seed=3,
        per_door={((0, 2), (1, 2)): 0.15, ((1, 2), (2, 2)): 0.15},
    ),
    BenchmarkMap(
        name="6×6  clustered (min_prob=0.50)",
        rows=6, cols=6, min_prob=0.50, seed=4, cluster_size=3,
    ),
]

N_EPISODES     = 100   # episodes per condition per map
MCTS_ROLLOUTS  = 300   # UCT iterations per plan() call  — single source of truth
ROLLOUT_DEPTH  = 50    # max steps per simulation rollout — single source of truth

PERSIST_TRACES = True                          # write traces to SQLite when True
TRACES_DB      = os.path.join(               # path to the SQLite database
    os.path.dirname(__file__), "..", "results", "trials.db"
)


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

# Each condition is a factory: (env) -> OnlineReplanningAgent
# Phases 3+ will register additional conditions here.

def _condition_mcts_primitive(env: GridWorld) -> OnlineReplanningAgent:
    """Condition 1: MCTSPlanner with primitive door actions only."""
    return make_mcts_agent(env, n_rollouts=MCTS_ROLLOUTS,
                           rollout_depth=ROLLOUT_DEPTH, seed=0)


def _condition_mcts_macro(env: GridWorld) -> OnlineReplanningAgent:
    """Condition 2: MCTSPlanner with auto-generated macro-actions (Phase 3)."""
    return make_macro_agent(env, n_rollouts=MCTS_ROLLOUTS,
                            rollout_depth=ROLLOUT_DEPTH, seed=0)


CONDITIONS: List[tuple] = [
    ("MDP alone (primitive)", _condition_mcts_primitive),
    ("+ Macro-actions",       _condition_mcts_macro),         # Phase 3
    # ("+ Strategies",         _condition_mcts_strategies),  # Phase 4+5
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_condition(
    label: str,
    agent_factory: Callable[[GridWorld], OnlineReplanningAgent],
    benchmark: BenchmarkMap,
    n_episodes: int,
    recorder: Optional["TraceRecorder"] = None,
) -> ConditionResult:
    env = benchmark.make_env()
    agent = agent_factory(env)
    result = ConditionResult(label=label, n_episodes=n_episodes)

    run_id: Optional[int] = None
    if recorder is not None:
        params = {"n_rollouts": MCTS_ROLLOUTS, "rollout_depth": ROLLOUT_DEPTH}
        run_id = recorder.begin_run(label, type(agent.inner_planner).__name__, params, env)

    for ep in range(n_episodes):
        env.reset(seed=ep)
        step_cb = recorder.make_step_callback() if recorder is not None else None
        ep_result = agent.run_episode(step_callback=step_cb)
        if ep_result["reached_goal"]:
            result.successes += 1
        result.total_replans += ep_result["replans"]
        result.total_actions += ep_result["actions_taken"]
        result.total_decisions += ep_result["decisions"]
        result.total_planning_time += ep_result["planning_time"]
        if recorder is not None:
            recorder.end_episode(run_id, ep, ep_result, env)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(benchmark: BenchmarkMap, results: List[ConditionResult]) -> None:
    col_w = 26
    num_w = 10

    divider = "─" * (col_w + num_w * 5)
    header = (f"{'Condition':<{col_w}}"
              f"{'Success':>{num_w}}"
              f"{'Replans':>{num_w}}"
              f"{'Actions':>{num_w}}"
              f"{'Decisions':>{num_w}}"
              f"{'Plan(s)':>{num_w}}")

    print(f"\n{'═' * len(divider)}")
    print(f"Map : {benchmark.name}   ({N_EPISODES} episodes each)")
    print(divider)
    print(header)
    print(divider)

    for r in results:
        print(f"{r.label:<{col_w}}"
              f"{r.success_rate:>{num_w-1}.1f}%"
              f"{r.avg_replans:>{num_w}.2f}"
              f"{r.avg_actions:>{num_w}.2f}"
              f"{r.avg_decisions:>{num_w}.2f}"
              f"{r.avg_planning_time:>{num_w}.3f}")

    print(divider)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\nThree-way evaluation harness")
    print(f"Conditions active : {len(CONDITIONS)}")
    print(f"Benchmark maps    : {len(BENCHMARK_MAPS)}")
    print(f"Episodes per cell : {N_EPISODES}")
    print(f"MCTS rollouts     : {MCTS_ROLLOUTS}")
    if PERSIST_TRACES:
        print(f"Trace DB          : {TRACES_DB}")

    recorder: Optional[TraceRecorder] = (
        TraceRecorder(TRACES_DB) if PERSIST_TRACES else None
    )

    t_start = time.perf_counter()

    try:
        for benchmark in BENCHMARK_MAPS:
            results = []
            for label, factory in CONDITIONS:
                r = run_condition(label, factory, benchmark, N_EPISODES,
                                  recorder=recorder)
                results.append(r)
            print_table(benchmark, results)
    finally:
        if recorder is not None:
            recorder.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
