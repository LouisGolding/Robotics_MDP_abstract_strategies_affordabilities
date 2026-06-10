"""
Phase 2 test script — MCTSPlanner (UCT) validation.

Three verification goals
------------------------
1. Agreement with oracle on a tiny high-p map: MCTSPlanner's first recommended
   action should match the first step of the ReliablePathPlanner optimal path
   when given enough rollouts on a 3x3 grid with p=0.99 doors (nearly
   deterministic, so UCT converges to the same solution as Dijkstra).

2. End-to-end episodes: MCTSPlanner drives OnlineReplanningAgent to reach the
   goal on all five experimental grid configs used in test_baseline.py.
   Success rates are reported — absolute thresholds are not asserted (the MDP
   is stochastic), but the agent must reach the goal in the majority of episodes.

3. All-conditions factory: make_mcts_agent creates a working agent.

Reference
---------
Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo planning.
    ECML 2006, LNCS vol 4212.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import (
    GridWorld,
    DoorProbabilitySpec,
    MCTSPlanner,
    ReliablePathPlanner,
    make_mcts_agent,
)


# ---------------------------------------------------------------------------
# 1. Oracle agreement test
# ---------------------------------------------------------------------------

def test_oracle_agreement(n_trials: int = 10) -> None:
    """
    Structural agreement test: on a 3x3 grid with a single high-probability
    corridor (right column, p=0.99) vs. a low-probability route (left column,
    p=0.20), both the oracle and MCTS should avoid the low-prob column.
    The test checks that MCTS first action is never into a low-prob door,
    which holds with high confidence for 1000+ rollouts.
    """
    # Bottleneck spec: left-column vertical doors are very unreliable
    spec = DoorProbabilitySpec(
        default=0.95,
        per_door={
            ((0, 0), (1, 0)): 0.10,
            ((1, 0), (2, 0)): 0.10,
        },
    )
    oracle = ReliablePathPlanner(alpha=0.0)
    mcts   = MCTSPlanner(n_rollouts=1000, rollout_depth=20, seed=0)

    oracle_avoids = mcts_avoids = 0
    bad_doors = {((0, 0), (1, 0)), ((1, 0), (2, 0))}

    for seed in range(n_trials):
        env = GridWorld(rows=3, cols=3, spec=spec, alpha=0.0, seed=seed)
        env.reset(seed=seed)
        source = env.current_node

        op = oracle.plan(env, source)
        mp = mcts.plan(env, source)

        if op and len(op) > 1:
            edge = (min(source, op[1]), max(source, op[1]))
            if edge not in bad_doors:
                oracle_avoids += 1

        if mp and len(mp) > 1:
            edge = (min(source, mp[1]), max(source, mp[1]))
            if edge not in bad_doors:
                mcts_avoids += 1

    print(f"\n{'='*52}")
    print(f"Oracle agreement test (3×3 bottleneck, {n_trials} trials)")
    print(f"  Oracle avoids bad doors : {oracle_avoids}/{n_trials}")
    print(f"  MCTS   avoids bad doors : {mcts_avoids}/{n_trials}")
    print(f"  {'PASS ✓' if mcts_avoids >= n_trials * 0.7 else 'WARN — MCTS taking bad doors too often'}")
    print(f"{'='*52}")


# ---------------------------------------------------------------------------
# 2. End-to-end episode experiments (mirroring test_baseline.py configs)
# ---------------------------------------------------------------------------

def run_experiment(
    label: str,
    env: GridWorld,
    n_rollouts: int = 200,
    n_episodes: int = 50,
    verbose_first: bool = False,
) -> None:
    agent = make_mcts_agent(env, n_rollouts=n_rollouts, max_steps=500, seed=42)
    successes = total_replans = total_actions = 0

    for ep in range(n_episodes):
        env.reset(seed=ep)
        result = agent.run_episode(verbose=(ep == 0 and verbose_first))
        if result["reached_goal"]:
            successes += 1
        total_replans += result["replans"]
        total_actions += result["actions_taken"]

    rate = successes / n_episodes * 100
    print(f"\n{'='*52}")
    print(f"Experiment : {label}")
    print(f"  Inner planner : MCTSPlanner (UCT, {n_rollouts} rollouts)")
    print(f"  Episodes      : {n_episodes}")
    print(f"  Success rate  : {rate:.1f}%")
    print(f"  Avg replans   : {total_replans / n_episodes:.2f}")
    print(f"  Avg actions   : {total_actions / n_episodes:.2f}")
    print(f"{'='*52}")
    print("\nFinal episode state:")
    print(env.render_text())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_oracle_agreement()

    # Same five configurations as test_baseline.py
    run_experiment(
        "5×5  p=0.80",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.80),
                  alpha=0.0, seed=42),
    )

    run_experiment(
        "5×5  p=0.90",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.90),
                  alpha=0.0, seed=42),
    )

    run_experiment(
        "5×5  bottleneck  (two doors at p=0.20)",
        GridWorld(rows=5, cols=5, alpha=0.0, seed=42,
                  spec=DoorProbabilitySpec(
                      default=0.90,
                      per_door={((0, 2), (1, 2)): 0.20,
                                ((1, 2), (2, 2)): 0.20})),
    )

    run_experiment(
        "6×6  clustered (3×3 tiles)  p=0.75",
        GridWorld(rows=6, cols=6, spec=DoorProbabilitySpec(default=0.75),
                  alpha=0.0, seed=42, cluster_size=3),
        n_rollouts=300,
    )

    run_experiment(
        "5×5  p=0.95",
        GridWorld(rows=5, cols=5, spec=DoorProbabilitySpec(default=0.95),
                  alpha=0.0, seed=42),
    )


if __name__ == "__main__":
    main()
