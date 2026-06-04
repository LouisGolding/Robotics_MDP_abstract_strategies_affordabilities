"""
Affordance scoring and strategy selection for the stochastic navigation MDP.

Phase 5 — activates the third evaluation condition.

WHAT AFFORDANCES ARE IN THIS PROJECT
--------------------------------------
An affordance is a *predictive score* of how beneficial a stored strategy is
from the current state.  Given the agent's current position and a library of
strategies, the affordance module scores each candidate strategy and returns
a ranked list — the top-ranked strategy (or strategies) then bias MCTS tree
expansion in the planner.

Affordances define each strategy's *basin of attraction*: the set of states
from which that strategy is a good choice to activate.

WHAT AFFORDANCES ARE NOT
-------------------------
Affordances do NOT filter or prune primitive door-actions before search.
The MCTS planner retains full access to all available primitive moves; the
affordance module only influences WHICH STRATEGY the planner is nudged toward
at the macro level.

THE AFFORDANCE VECTOR
----------------------
Each strategy is scored as a 4-component vector (lower = better, 0 = ideal):

  start_aff       : effort to reach the strategy's entry state from the
                    current state (how far away is the strategy's starting
                    point?)
  strategy_aff    : effort to traverse / refine the strategy's road map
                    (how costly is the strategy itself?)
  task_aff        : remaining effort to the goal after the strategy completes
                    (how much work is left after the strategy exits?)
  reliability_aff : OUR NOVEL CONTRIBUTION — how likely is the strategy to be
                    successfully grounded into real primitive moves given that
                    some doors may be locked?  Low = strategy is robust under
                    transition uncertainty; high = strategy is fragile.

The combined score drives strategy selection.  Strategies that score poorly
on reliability_aff are down-ranked even if their utility components are good —
this is the quantitative contribution of the thesis.

INTERFACE (to be implemented in Phase 5)
-----------------------------------------
  AffordanceScorer.score(state, goal, strategy, env) -> AffordanceVector
  AffordanceSelector.select(state, goal, library, env) -> List[Strategy]

Reference
---------
Khen Elimelech's deterministic affordance formulation (supervisor's prior
work).  This module extends utility-only affordances with reliability_aff
to handle stochastic transition uncertainty.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .strategies import Strategy, StrategyLibrary


@dataclass
class AffordanceVector:
    """
    4-component affordance score for one strategy from one state.
    All components are non-negative; lower is better (0 = ideal).
    """
    start_aff:       float   # effort: current state → strategy entry
    strategy_aff:    float   # effort: traverse the strategy road map
    task_aff:        float   # effort: strategy exit → goal
    reliability_aff: float   # fragility under stochastic transitions (novel)

    @property
    def total(self) -> float:
        return self.start_aff + self.strategy_aff + self.task_aff + self.reliability_aff

    def __repr__(self) -> str:
        return (f"AffordanceVector(start={self.start_aff:.3f}, "
                f"strategy={self.strategy_aff:.3f}, "
                f"task={self.task_aff:.3f}, "
                f"reliability={self.reliability_aff:.3f}, "
                f"total={self.total:.3f})")


class AffordanceScorer:
    """
    Scores a single strategy from a given state.

    TODO (Phase 5): implement each component.
      - start_aff      : shortest-path distance from state to strategy.entry
      - strategy_aff   : sum of -log(p) weights along strategy waypoints
      - task_aff       : shortest-path distance from strategy.exit to goal
      - reliability_aff: probability that all doors along the strategy road map
                         are still available (product of open-probs for
                         untried doors; 0 cost for already-confirmed-open doors)
    """

    def score(
        self,
        state: Tuple[int, int],
        goal: Tuple[int, int],
        strategy: Strategy,
        env,                    # GridWorld — read-only
    ) -> AffordanceVector:
        raise NotImplementedError("AffordanceScorer.score — implement in Phase 5")


class AffordanceSelector:
    """
    Ranks all strategies in the library by affordance score and returns
    the top-k candidates to bias MCTS expansion toward.

    TODO (Phase 5): implement selection logic.
    """

    def __init__(self, scorer: AffordanceScorer, top_k: int = 3) -> None:
        self.scorer = scorer
        self.top_k  = top_k

    def select(
        self,
        state: Tuple[int, int],
        goal: Tuple[int, int],
        library: StrategyLibrary,
        env,                    # GridWorld — read-only
    ) -> List[Strategy]:
        """
        Score every strategy in the library and return the top_k by
        total affordance (lowest score = most applicable).
        """
        raise NotImplementedError("AffordanceSelector.select — implement in Phase 5")
