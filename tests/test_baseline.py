"""
Phase 1 test script — online replanning baseline (ReliablePathPlanner).

Problem : Stochastic Shortest-Path MDP; doors fail permanently on first
          unsuccessful attempt.
Solver  : OnlineReplanningAgent — plan-act-observe-replan loop.
Planner : ReliablePathPlanner — Dijkstra on -log(p) weights (scaffolding).

Five experiments varying door probabilities, grid structure, and alpha.

Note on alpha selection
-----------------------
Alpha is the minimum acceptable path probability.  For a k-hop shortest path
with per-door probability p, the achievable ceiling is p^k.  Alpha must sit
below that ceiling or the agent will always give up immediately.

  5x5 grid  (8-hop min path):  p=0.80 → ceil=0.168
                                p=0.90 → ceil=0.430
                                p=0.95 → ceil=0.663
  6x6 grid (10-hop min path):  p=0.75 → ceil=0.056
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import GridWorld, DoorProbabilitySpec, make_baseline_agent


def run_experiment(
    label: str,
    env: GridWorld,
    alpha: float,
    n_episodes: int = 50,
    verbose_first: bool = False,
) -> None:
    agent = make_baseline_agent(env, alpha=alpha)
    successes = total_replans = total_actions = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        result = agent.run_episode(verbose=(ep == 0 and verbose_first))
        if result["reached_goal"]:
            successes += 1
        total_replans += result["replans"]
        total_actions += result["actions_taken"]

    print(f"\n{'='*52}")
    print(f"Experiment : {label}")
    print(f"  Inner planner : ReliablePathPlanner (Dijkstra on -log p)")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Success rate  : {successes / n_episodes * 100:.1f}%")
    print(f"  Avg replans   : {total_replans / n_episodes:.2f}")
    print(f"  Avg actions   : {total_actions / n_episodes:.2f}")
    print(f"{'='*52}")
    print("\nFinal episode state:")
    print(env.render_text())


def main():
    # Exp 1 — moderate doors
    run_experiment(
        "5x5  p=0.80  alpha=0.10",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.80),
                  alpha=0.10, seed=42),
        alpha=0.10,
    )

    # Exp 2 — high door probability → fewer replans
    run_experiment(
        "5x5  p=0.90  alpha=0.30",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.90),
                  alpha=0.30, seed=42),
        alpha=0.30,
    )

    # Exp 3 — bottleneck: two low-prob doors force detour
    run_experiment(
        "5x5  bottleneck  alpha=0.25",
        GridWorld(rows=5, cols=5, alpha=0.25, seed=42,
                  spec=DoorProbabilitySpec(
                      default=0.90,
                      per_door={((0, 2), (1, 2)): 0.20,
                                ((1, 2), (2, 2)): 0.20})),
        alpha=0.25,
    )

    # Exp 4 — 6x6 clustered (3x3 tiles)
    run_experiment(
        "6x6  clustered (3x3 tiles)  alpha=0.04",
        GridWorld(rows=6, cols=6, spec=DoorProbabilitySpec(default=0.75),
                  alpha=0.04, seed=42, cluster_size=3),
        alpha=0.04,
    )

    # Exp 5 — near-optimal: very high p, demanding alpha
    run_experiment(
        "5x5  p=0.95  alpha=0.50",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.95),
                  alpha=0.50, seed=42),
        alpha=0.50,
    )


if __name__ == "__main__":
    main()
