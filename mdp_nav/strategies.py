"""
Abstract strategy policies for the stochastic navigation MDP.

Phase 4 — third evaluation condition.

A strategy is a *road map*: an ordered sequence of abstract states (nodes)
that describes a high-level path through the environment that has worked
before.  Strategies are stored in a library and reused across episodes.

Strategies are NOT primitive action sequences — they live at the level of
graph nodes (rooms), not individual door attempts.  The MCTS planner uses a
strategy as a structured prior over which parts of the space to explore,
replacing undirected random rollouts with experience-informed expansion.

Relationship to affordances (Phase 5)
--------------------------------------
Affordances SCORE strategies — they answer "which strategy in the library is
most applicable right now?"  Strategies are the objects being scored.
The two modules are deliberately separate: strategies.py owns the road-map
representation and the library; affordances.py owns the scoring/selection
logic that picks which strategy to activate.

Reference
---------
Khen Elimelech's deterministic formulation of abstract strategies and
affordances (supervisor's prior work — deterministic setting).  This module
extends that concept to the stochastic (MDP) setting.
"""

from __future__ import annotations
from typing import List, Optional, Tuple


class Strategy:
    """
    A stored road map — a sequence of abstract states (rooms) that
    constitutes a high-level plan fragment.

    Parameters
    ----------
    name      : human-readable identifier
    waypoints : ordered sequence of (row, col) nodes from entry to exit
    """

    def __init__(self, name: str, waypoints: List[Tuple[int, int]]) -> None:
        self.name = name
        self.waypoints = waypoints

    @property
    def entry(self) -> Tuple[int, int]:
        """First node in the road map."""
        return self.waypoints[0]

    @property
    def exit(self) -> Tuple[int, int]:
        """Last node in the road map."""
        return self.waypoints[-1]

    def __len__(self) -> int:
        return len(self.waypoints)

    def __repr__(self) -> str:
        return f"Strategy({self.name!r}, {self.waypoints})"


class StrategyLibrary:
    """
    Collection of stored strategies available to the planner.

    TODO (Phase 4): populate from successful episode traces — extract
    recurring sub-paths and store them as reusable strategies.
    """

    def __init__(self) -> None:
        self._strategies: List[Strategy] = []

    def add(self, strategy: Strategy) -> None:
        self._strategies.append(strategy)

    def all(self) -> List[Strategy]:
        return list(self._strategies)

    def __len__(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        return f"StrategyLibrary({len(self)} strategies)"
