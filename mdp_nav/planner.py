"""
Baseline online MDP planner — primitive actions only.

Algorithm (replanning loop):
  1. From current state, enumerate all simple paths to any goal.
  2. Score each path by its cumulative success probability
     (product of edge open-probabilities for doors not yet failed).
  3. Pick the best-scoring path whose probability >= alpha.
     If none exists, declare failure.
  4. Execute the first action on the chosen path.
  5. If the door failed, replan (go to step 1).
  6. If we reach a goal, declare success.

Metrics tracked per episode:
  - actions_taken  : total door attempts
  - replans        : number of times step 1 was re-entered after a failure
  - reached_goal   : bool
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .environment import GridWorld


class MDPPlanner:
    """
    Online replanning MDP planner over primitive door actions.

    Parameters
    ----------
    env   : GridWorld instance (shared; planner does NOT own it)
    alpha : confidence threshold (overrides env.alpha if provided)
    max_steps : hard cap on total actions per episode to prevent loops
    """

    def __init__(
        self,
        env: GridWorld,
        alpha: Optional[float] = None,
        max_steps: int = 500,
        detour_slack: int = 2,
    ):
        self.env = env
        self.alpha = alpha if alpha is not None else env.alpha
        self.max_steps = max_steps
        # paths longer than shortest_path + detour_slack are skipped;
        # keeps all_simple_paths tractable on large grids
        self._detour_slack = detour_slack

        # per-episode counters, reset each run
        self.actions_taken: int = 0
        self.replans: int = 0
        self.reached_goal: bool = False

    # ------------------------------------------------------------------
    # Path search
    # ------------------------------------------------------------------

    def _best_path(
        self,
        source: Tuple[int, int],
    ) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        """
        Find the highest-probability simple path from source to any goal
        that uses only available (non-failed) doors.

        Returns (path, cumulative_prob) or None if no valid path exists.
        """
        G = self.env.graph
        available = {
            frozenset({u, v})
            for u, v in G.edges()
            if not self.env.is_failed(u, v)
        }

        best_path: Optional[List] = None
        best_prob: float = -1.0

        for goal in self.env.goals:
            if goal == source:
                return ([source], 1.0)
            # Limit search to paths no longer than shortest_path + slack.
            # This keeps the search tractable on large grids.
            try:
                shortest = nx.shortest_path_length(G, source, goal)
            except nx.NetworkXNoPath:
                continue
            cutoff = shortest + self._detour_slack
            for path in nx.all_simple_paths(G, source, goal, cutoff=cutoff):
                # Check all edges on path are available
                prob = 1.0
                valid = True
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    key = frozenset({u, v})
                    if key not in available:
                        valid = False
                        break
                    prob *= G[u][v]["prob"]
                if valid and prob > best_prob:
                    best_prob = prob
                    best_path = path

        if best_path is None or best_prob < self.alpha:
            return None
        return (best_path, best_prob)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------

    def run_episode(self, verbose: bool = False) -> Dict:
        """
        Run a single episode from env.current_node to a goal.
        Assumes env.reset() has already been called.

        Returns dict with keys: reached_goal, actions_taken, replans.
        """
        self.actions_taken = 0
        self.replans = 0
        self.reached_goal = False

        while self.actions_taken < self.max_steps:
            current = self.env.current_node

            if current in self.env.goals:
                self.reached_goal = True
                break

            result = self._best_path(current)
            if result is None:
                if verbose:
                    print(f"  [planner] No viable path from {current} (alpha={self.alpha}). Giving up.")
                break

            path, prob = result
            next_node = path[1]  # first action

            if verbose:
                print(f"  [planner] At {current}, plan: {path} (p={prob:.3f}), attempting door→{next_node}")

            _, success, done = self.env.step(next_node)
            self.actions_taken += 1

            if verbose:
                status = "OK" if success else "FAILED"
                print(f"  [planner] Door {current}↔{next_node}: {status}. Now at {self.env.current_node}")

            if done:
                self.reached_goal = True
                break

            if not success:
                self.replans += 1

        return {
            "reached_goal": self.reached_goal,
            "actions_taken": self.actions_taken,
            "replans": self.replans,
        }
