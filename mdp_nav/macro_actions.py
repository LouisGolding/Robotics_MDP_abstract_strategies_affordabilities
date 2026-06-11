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

    Concretely, a macro-action is a *committed road* through the graph: an
    ordered list of rooms the agent intends to traverse from a single entry
    state.  In the MCTS tree it appears as one candidate move whose outcome is
    evaluated by simulating each hop's door under its open-probability; in the
    online loop the planner returns the macro's full node sequence and the
    agent commits to it, replanning only if a door along the way fails.

    Parameters
    ----------
    name            : human-readable identifier
    initiation_set  : nodes from which this macro-action may be started
                      (for auto-generated corridor macros this is the single
                      entry node)
    waypoints       : ordered list of rooms AFTER the entry state, ending at
                      terminal_node.  waypoints[0] is the first hop.
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

    @property
    def first_hop(self) -> Tuple[int, int]:
        """The first room entered when the macro is executed."""
        return self.waypoints[0]

    def __len__(self) -> int:
        return len(self.waypoints)

    def __repr__(self) -> str:
        return (f"MacroAction({self.name!r}, "
                f"{len(self.waypoints)} hops → {self.terminal_node})")


class MacroActionLibrary:
    """Collection of macro-actions available to the planner."""

    def __init__(self) -> None:
        self._actions: List[MacroAction] = []

    def add(self, macro: MacroAction) -> None:
        self._actions.append(macro)

    def applicable(self, state: Tuple[int, int]) -> List[MacroAction]:
        """Return all macro-actions whose initiation set contains state."""
        return [m for m in self._actions if m.is_applicable(state)]

    def all(self) -> List[MacroAction]:
        return list(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions)


# ---------------------------------------------------------------------------
# Auto-generation from graph topology (Phase 3)
# ---------------------------------------------------------------------------

# The four grid directions, as (d_row, d_col) deltas.
_DIRECTIONS = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0),
}


def _straight_run(
    graph,
    start: Tuple[int, int],
    delta: Tuple[int, int],
    max_length: int,
) -> List[Tuple[int, int]]:
    """
    Follow a straight line from `start` in direction `delta`, collecting the
    rooms reached while consecutive doors exist in the base graph.

    Returns the list of rooms AFTER start (the waypoints), up to max_length
    hops.  Crucially this walks the *topology only* — door probabilities and
    failed doors are irrelevant here; reliability is evaluated later by MCTS.
    """
    dr, dc = delta
    waypoints: List[Tuple[int, int]] = []
    cur = start
    for _ in range(max_length):
        nxt = (cur[0] + dr, cur[1] + dc)
        if not graph.has_edge(cur, nxt):
            break
        waypoints.append(nxt)
        cur = nxt
    return waypoints


def generate_macro_library(
    env: "GridWorld",
    max_length: Optional[int] = None,
    min_length: int = 2,
) -> MacroActionLibrary:
    """
    Auto-generate a library of corridor-traversal / room-crossing macros from
    a GridWorld's graph topology.

    For every room and every one of the four grid directions we build the
    maximal straight run of rooms reachable along that line (capped at
    `max_length` hops).  A run of at least `min_length` hops becomes a
    macro-action whose entry state is the starting room.  On clustered maps
    these straight runs naturally cross the bridge edges between clusters, so
    they double as room-crossing macros.

    Why straight runs?  They are the cheapest interpretable temporal
    abstraction — "head East down this corridor" — and they give the MCTS
    planner a coarse move that, when committed to, collapses several
    plan-act-replan cycles into one decision.

    Parameters
    ----------
    env        : the GridWorld whose topology defines the macros
    max_length : cap on hops per macro (default: span of the grid)
    min_length : shortest run (in hops) that qualifies as a macro (default 2 —
                 a 1-hop run is just a primitive door)

    Returns
    -------
    MacroActionLibrary populated with the generated macros.
    """
    graph = env.graph
    if max_length is None:
        max_length = max(env.rows, env.cols)

    library = MacroActionLibrary()
    for node in graph.nodes():
        for dname, delta in _DIRECTIONS.items():
            run = _straight_run(graph, node, delta, max_length)
            if len(run) < min_length:
                continue
            terminal = run[-1]
            library.add(MacroAction(
                name=f"{node}-{dname}-{terminal}",
                initiation_set={node},
                waypoints=run,
                terminal_node=terminal,
            ))
    return library
