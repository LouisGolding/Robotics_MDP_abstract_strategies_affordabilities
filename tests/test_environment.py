"""
Environment semantics tests — the door model is STATIC.

Canonical rule (CLAUDE.md / PROJECT_LOG.md):
  Each door's state is decided exactly once, on the first attempt.
    · open  → stays open forever (effective probability 1, no re-roll)
    · closed → stays closed forever (permanently locked, no retry)

These tests guard against regressions where a traversed-open door is
re-sampled on later traversals (which would make the environment dynamic).

Run:
    python tests/test_environment.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mdp_nav import GridWorld, DoorProbabilitySpec


def test_opened_door_stays_open():
    """A door that has opened must never fail on any later traversal."""
    env = GridWorld(rows=2, cols=2,
                    spec=DoorProbabilitySpec(min_prob=0.5), seed=1)
    opened_one = False
    for trial in range(50):
        env.reset(seed=trial)
        a = env.current_node
        nb = (0, 1) if env.graph.has_edge((0, 0), (0, 1)) else (1, 0)
        _, success, _ = env.step(nb)
        if not success:
            continue
        opened_one = True
        assert env.is_open(a, nb), "opened door must be marked open"
        # Bounce across the same door 200 times — must never fail.
        for _ in range(200):
            back = a if env.current_node == nb else nb
            _, succ, _ = env.step(back)
            assert succ, "BUG: an opened door re-closed (environment not static)"
        break
    assert opened_one, "test never managed to open a door"
    print("· opened doors stay open (prob 1) across 200 re-traversals")


def test_closed_door_stays_closed():
    """A failed door is permanently locked — re-attempting it raises."""
    env = GridWorld(rows=2, cols=2,
                    spec=DoorProbabilitySpec(min_prob=0.5,
                                             per_door={((0, 0), (0, 1)): 0.0}),
                    seed=1)
    env.reset(seed=0)
    # door (0,0)-(0,1) has prob 0 → first attempt fails and locks it
    if env.graph.has_edge((0, 0), (0, 1)):
        _, success, _ = env.step((0, 1))
        assert not success, "prob-0 door should fail"
        assert env.is_failed((0, 0), (0, 1)), "failed door must be marked failed"
        raised = False
        try:
            env.step((0, 1))   # retry the locked door
        except ValueError:
            raised = True
        assert raised, "re-attempting a locked door must raise"
        print("· closed doors stay closed (retry raises)")
    else:
        print("· (no (0,0)-(0,1) edge on this map; skipped closed-door check)")


def test_opened_and_failed_sets_disjoint():
    """Over a full episode, no door is ever both opened and failed."""
    env = GridWorld(rows=5, cols=5,
                    spec=DoorProbabilitySpec(min_prob=0.5), seed=3)
    for ep in range(20):
        env.reset(seed=ep)
        for _ in range(40):
            avail = env.available_doors()
            if not avail or env.current_node in env.goals:
                break
            env.step(avail[0][0])
        assert env.opened_doors.isdisjoint(env.failed_doors), \
            "a door cannot be both open and failed"
    print("· opened and failed door sets stay disjoint across episodes")


if __name__ == "__main__":
    test_opened_door_stays_open()
    test_closed_door_stays_closed()
    test_opened_and_failed_sets_disjoint()
    print("\ntest_environment.py: ALL STATIC-DOOR TESTS PASSED")
