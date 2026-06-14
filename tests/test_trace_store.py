"""
Round-trip tests for TraceRecorder (mdp_nav/trace_store.py).

Three assertions:
  (a) len(steps) == actions_taken reported by run_episode
  (b) count of failed steps == replans reported by run_episode
  (c) stored state sequence matches the actually visited sequence
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import GridWorld, DoorProbabilitySpec, make_mcts_agent, TraceRecorder
from mdp_nav.planner import MCTSPlanner


def _small_env(seed: int = 42) -> GridWorld:
    return GridWorld(
        rows=3, cols=3,
        spec=DoorProbabilitySpec(min_prob=0.6),
        seed=seed,
    )


def test_round_trip():
    env = _small_env()
    agent = make_mcts_agent(env, n_rollouts=50, rollout_depth=20, seed=0)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        recorder = TraceRecorder(db_path)
        run_id = recorder.begin_run(
            condition_label="test",
            planner_type=type(agent.inner_planner).__name__,
            params={"n_rollouts": 50, "rollout_depth": 20},
            env=env,
        )

        # Track visited sequence independently for assertion (c).
        visited: list[tuple[int, int]] = []
        env.reset(seed=7)
        visited.append(env.current_node)

        def tracking_cb(info):
            visited.append(info["current"])

        step_cb = recorder.make_step_callback()

        # Chain both callbacks.
        def combined_cb(info):
            tracking_cb(info)
            step_cb(info)

        env.reset(seed=7)
        visited.clear()
        visited.append(env.current_node)
        result = agent.run_episode(step_callback=combined_cb)
        recorder.end_episode(run_id, 7, result, env)
        recorder.close()

        # Query the DB directly.
        conn = sqlite3.connect(db_path)
        steps = conn.execute(
            "SELECT success, resulting_state FROM steps WHERE episode_id=1 ORDER BY step_index"
        ).fetchall()
        conn.close()

        # (a) step count matches actions_taken
        assert len(steps) == result["actions_taken"], (
            f"step rows={len(steps)} != actions_taken={result['actions_taken']}"
        )

        # (b) failed steps == replans
        failed = sum(1 for s, _ in steps if s == 0)
        assert failed == result["replans"], (
            f"failed_steps={failed} != replans={result['replans']}"
        )

        # (c) resulting_state sequence matches visited (skip initial state)
        stored_states = [rs for _, rs in steps]
        expected_states = [f"{n[0]},{n[1]}" for n in visited[1:]]
        assert stored_states == expected_states, (
            f"state sequence mismatch:\nstored  ={stored_states}\nexpected={expected_states}"
        )

        print("All assertions passed.")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_round_trip()
    print("test_trace_store.py: OK")
