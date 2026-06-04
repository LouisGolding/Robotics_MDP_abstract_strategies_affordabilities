"""
Macro-actions / options for the stochastic navigation MDP.

Phase 3 — second evaluation condition.

A macro-action (option; Sutton, Precup & Singh, 1999) is a temporally
extended action that bundles a sequence of primitive door-attempts into a
single reusable unit.  Examples: "traverse the left corridor", "cross the
central hub".

Each macro-action is defined by:
  - initiation set  : states from which it may be started
  - policy          : how to execute it (sequence of primitive door attempts)
  - termination     : condition under which it ends (reached sub-goal or stuck)

In the MCTS planner (Phase 3+), macro-actions appear alongside primitive
actions as candidate moves in the tree.  The planner may select a macro-action
at any node; the outer loop then executes it step by step, replanning if a
door fails mid-execution.

Reference
---------
Sutton, R. S., Precup, D. & Singh, S. (1999). Between MDPs and semi-MDPs:
    A framework for temporal abstraction in reinforcement learning.
    Artificial Intelligence, 112(1-2), 181-211.
"""

from __future__ import annotations
from typing import List, Optional, Set, Tuple

from .environment import GridWorld


class MacroAction:
    """
    A single macro-action (option).

    Parameters
    ----------
    name            : human-readable identifier
    initiation_set  : nodes from which this macro-action may be started
    waypoints       : ordered list of intermediate sub-goal nodes
    terminal_node   : the node this macro-action aims to reach
    """

    def __init__(
        self,
        name: str,
        initiation_set: Set[Tuple[int, int]],
        waypoints: List[Tuple[int, int]],
        terminal_node: Tuple[int, int],
    ):
        self.name = name
        self.initiation_set = initiation_set
        self.waypoints = waypoints
        self.terminal_node = terminal_node

    def is_applicable(self, state: Tuple[int, int]) -> bool:
        return state in self.initiation_set

    def __repr__(self) -> str:
        return f"MacroAction({self.name!r}, terminal={self.terminal_node})"


class MacroActionLibrary:
    """
    Collection of macro-actions available to the planner.

    TODO (Phase 3): populate from GridWorld structure — e.g. auto-generate
    corridor-traversal and room-crossing macro-actions from the graph topology.
    """

    def __init__(self) -> None:
        self._actions: List[MacroAction] = []

    def add(self, macro: MacroAction) -> None:
        self._actions.append(macro)

    def applicable(self, state: Tuple[int, int]) -> List[MacroAction]:
        """Return all macro-actions whose initiation set contains state."""
        return [m for m in self._actions if m.is_applicable(state)]

    def __len__(self) -> int:
        return len(self._actions)
