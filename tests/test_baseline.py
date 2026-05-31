"""
Phase 1 test script: baseline MDP planner on a 5x5 grid-world.

Runs N episodes and prints:
  - Success rate
  - Average replans per episode
  - Average actions taken per episode

Also demonstrates:
  - Per-door probability overrides
  - Clustered environment variant
  - Text visualisation of the final episode state
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import GridWorld, DoorProbabilitySpec, MDPPlanner


def run_experiment(
    label: str,
    env: GridWorld,
    n_episodes: int = 50,
    verbose_episode: int = 0,
    alpha: float = 0.5,
) -> None:
    planner = MDPPlanner(env, alpha=alpha)

    successes, total_replans, total_actions = 0, 0, 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        verbose = ep == verbose_episode
        if verbose:
            print(f"\n--- Episode {ep} (verbose) ---")
        result = planner.run_episode(verbose=verbose)
        if result["reached_goal"]:
            successes += 1
        total_replans += result["replans"]
        total_actions += result["actions_taken"]

    print(f"\n{'='*50}")
    print(f"Experiment: {label}")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Success rate  : {successes/n_episodes*100:.1f}%")
    print(f"  Avg replans   : {total_replans/n_episodes:.2f}")
    print(f"  Avg actions   : {total_actions/n_episodes:.2f}")
    print(f"{'='*50}")

    # show text render of last episode state
    print("\nFinal episode state (text render):")
    print(env.render_text())


def main():
    # Note on alpha selection:
    # Alpha is the *minimum acceptable path probability*. For a 5x5 grid the
    # shortest path from (0,0) to (4,4) is 8 hops, so with p=0.8 per door the
    # best achievable probability is 0.8^8 ≈ 0.168.  Alpha must be below that
    # ceiling; we print the ceiling so the numbers are interpretable.

    def path_ceiling(p, hops):
        return p ** hops

    print("Min-path probability ceilings (p^hops):")
    print(f"  5x5  (8 hops): p=0.80 → {path_ceiling(0.80, 8):.3f}")
    print(f"  5x5  (8 hops): p=0.90 → {path_ceiling(0.90, 8):.3f}")
    print(f"  6x6 (10 hops): p=0.75 → {path_ceiling(0.75, 10):.3f}")
    print(f"  5x5  (8 hops): p=0.95 → {path_ceiling(0.95, 8):.3f}")
    print()

    # ------------------------------------------------------------------
    # Experiment 1: uniform 5x5, moderate door probability
    # ceiling ≈ 0.168 → use alpha=0.10
    # ------------------------------------------------------------------
    spec1 = DoorProbabilitySpec(default=0.8)
    env1 = GridWorld(rows=5, cols=5, spec=spec1, alpha=0.10, seed=42)
    run_experiment("5x5 uniform p=0.80, alpha=0.10", env1, n_episodes=50, alpha=0.10)

    # ------------------------------------------------------------------
    # Experiment 2: uniform 5x5, high door probability → fewer replans
    # ceiling ≈ 0.430 → use alpha=0.30
    # ------------------------------------------------------------------
    spec2 = DoorProbabilitySpec(default=0.9)
    env2 = GridWorld(rows=5, cols=5, spec=spec2, alpha=0.30, seed=42)
    run_experiment("5x5 uniform p=0.90, alpha=0.30", env2, n_episodes=50, alpha=0.30)

    # ------------------------------------------------------------------
    # Experiment 3: per-door overrides — bottleneck corridor
    # Two risky doors in the middle; agent must detour around them.
    # ceiling for detour path (also 8 hops, but all p=0.9) ≈ 0.43
    # ------------------------------------------------------------------
    spec3 = DoorProbabilitySpec(
        default=0.9,
        per_door={
            ((0, 2), (1, 2)): 0.2,  # bottleneck door 1
            ((1, 2), (2, 2)): 0.2,  # bottleneck door 2
        },
    )
    env3 = GridWorld(rows=5, cols=5, spec=spec3, alpha=0.25, seed=42)
    run_experiment("5x5 bottleneck corridor, alpha=0.25", env3, n_episodes=50, alpha=0.25)

    # ------------------------------------------------------------------
    # Experiment 4: 6x6 with 3x3 cluster structure
    # 10-hop min path, p=0.75 → ceiling ≈ 0.056 → alpha=0.04
    # ------------------------------------------------------------------
    spec4 = DoorProbabilitySpec(default=0.75)
    env4 = GridWorld(
        rows=6, cols=6, spec=spec4, alpha=0.04, seed=42, cluster_size=3
    )
    run_experiment("6x6 clustered (3x3 tiles), alpha=0.04", env4, n_episodes=50, alpha=0.04)

    # ------------------------------------------------------------------
    # Experiment 5: 5x5, very high door probability → near-optimal baseline
    # ceiling ≈ 0.663 → alpha=0.50 (demanding but achievable)
    # ------------------------------------------------------------------
    spec5 = DoorProbabilitySpec(default=0.95)
    env5 = GridWorld(rows=5, cols=5, spec=spec5, alpha=0.50, seed=42)
    run_experiment("5x5 uniform p=0.95, alpha=0.50", env5, n_episodes=50, alpha=0.50)


if __name__ == "__main__":
    main()
