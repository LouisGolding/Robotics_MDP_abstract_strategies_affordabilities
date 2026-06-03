"""
Online plan-act-observe-replan loop for stochastic navigation.

═══════════════════════════════════════════════════════════════════════════════
PROBLEM FORMULATION
═══════════════════════════════════════════════════════════════════════════════
We model navigation as a Stochastic Shortest-Path MDP (Bertsekas & Tsitsiklis,
1991) over a graph-based grid-world:

  • States   S : rooms (graph nodes)
  • Actions  A : attempt a door to an adjacent room
  • Transition: door (s→s') succeeds with prob p ∈ (0,1]; on failure the door
                is permanently closed for this episode (no retry)
  • Cost      : each action costs 1 (unit step cost)
  • Goal      : reach any node in the goal set G ⊆ S

The key constraint is that failed doors cannot be retried — the agent receives
partial observability about transition availability as the episode unfolds.

═══════════════════════════════════════════════════════════════════════════════
SOLVER: ONLINE PLAN-ACT-OBSERVE-REPLAN
═══════════════════════════════════════════════════════════════════════════════
We deliberately do NOT compute a full offline policy (value iteration over all
states). Instead we use an ONLINE loop (Koenig & Likhachev, 2002):

    while not at goal:
        plan  ← InnerPlanner.plan(current_state, env)   # plan from HERE
        if plan is None: give up (no viable route found)
        execute plan[1]                                   # one step only
        observe outcome (success / door now locked)
        if door locked: replan                            # back to top

This is the standard "determinize-and-replan" paradigm (Yoon, Fern & Givan,
ICAPS 2007) generalised to support swappable inner planners.

Advantages over offline value iteration for this setting:
  - Handles permanently-closing doors naturally (plan is always over current
    available graph)
  - Scales to large maps where full-policy computation is expensive
  - Clean extension point: swap inner planner for MCTS/UCT without touching
    the outer loop (see MCTS note below)

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: INNER PLANNER INTERFACE
═══════════════════════════════════════════════════════════════════════════════
The outer loop (OnlineReplanningAgent) is decoupled from HOW a plan is
produced. Any class implementing InnerPlanner can be plugged in:

    InnerPlanner (ABC)
    ├── ReliablePathPlanner   ← CURRENT (scaffolding baseline)
    └── MCTSPlanner           ← NEXT (target inner planner, Phase 2+)

This means all three experimental conditions (primitive / +macro-actions /
+strategies) share the SAME outer loop — only the set of available moves
passed to the inner planner differs.

═══════════════════════════════════════════════════════════════════════════════
CURRENT INNER PLANNER: ReliablePathPlanner (scaffolding)
═══════════════════════════════════════════════════════════════════════════════
Uses Dijkstra's algorithm (Dijkstra, 1959) on edge weights −log(p) to find
the most-reliable (maximum cumulative success probability) path to any goal,
restricted to doors not yet failed.  Returns the path only if its cumulative
probability ≥ α (confidence threshold).

  weight(u→v) = −log(p_uv)
  most-reliable path = shortest path under these weights
  cumulative prob    = exp(−total_weight) = ∏ p_i

This is a clean, well-understood baseline but is NOT the target algorithm.
It is "scaffolding" in the sense of Yoon et al. (2007): a simple, correct
baseline that makes the online loop work end-to-end on small maps.

═══════════════════════════════════════════════════════════════════════════════
TARGET INNER PLANNER: MCTS/UCT (planned, Phase 2+)
═══════════════════════════════════════════════════════════════════════════════
The target is Monte Carlo Tree Search with UCB1 (UCT; Kocsis & Szepesvári,
2006).  From the current state, UCT builds a partial search tree via repeated
rollouts, balancing exploration vs. exploitation.  In later phases, abstract
strategies will bias the tree expansion (fewer rollouts needed).

References
----------
Bertsekas, D. P. & Tsitsiklis, J. N. (1991). An analysis of stochastic
    shortest path problems. Mathematics of Operations Research, 16(3).

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
    Numerische Mathematik, 1(1), 269–271.

Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo planning.
    ECML 2006. Lecture Notes in Computer Science, vol 4212.

Koenig, S. & Likhachev, M. (2002). D* Lite. AAAI 2002.

Puterman, M. L. (1994). Markov Decision Processes. Wiley.

Yoon, S., Fern, A. & Givan, R. (2007). FF-Replan: A baseline for
    probabilistic planning. ICAPS 2007.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .environment import GridWorld


# ═══════════════════════════════════════════════════════════════════════════
# Inner Planner Interface
# ═══════════════════════════════════════════════════════════════════════════

class InnerPlanner(ABC):
    """
    Abstract interface for the planning component of the online loop.

    Contract
    --------
    Given the current environment state (accessible via `env`) and the agent's
    current position `source`, return an ordered list of nodes from `source`
    toward a goal, or None if no acceptable plan exists.

    Only the FIRST element of the returned plan is ever executed by the outer
    loop — the planner is called again after each step.  The plan should
    therefore reflect the world as it is RIGHT NOW (failed doors excluded).

    The inner planner must NOT modify env state — it is read-only.
    """

    @abstractmethod
    def plan(
        self,
        env: GridWorld,
        source: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Produce a plan from `source` toward any goal in env.goals.

        Parameters
        ----------
        env    : current environment (read-only; failed_doors reflects history)
        source : current agent position

        Returns
        -------
        List of nodes [source, n1, n2, ..., goal] using only available doors,
        or None if no acceptable plan can be found.
        """


# ═══════════════════════════════════════════════════════════════════════════
# ReliablePathPlanner — scaffolding baseline
# ═══════════════════════════════════════════════════════════════════════════

class ReliablePathPlanner(InnerPlanner):
    """
    Most-reliable-path inner planner (scaffolding baseline).

    Finds the path to any goal that maximises cumulative success probability,
    implemented as Dijkstra on edge weights −log(p).  Returns the path only if
    its cumulative probability ≥ alpha; otherwise returns None.

    Complexity: O((V + E) log V) per call via networkx Dijkstra.

    This is SCAFFOLDING — a clean, correct baseline to validate the online
    loop end-to-end on small maps.  Replace with MCTSPlanner for the main
    experimental results.

    Parameters
    ----------
    alpha        : minimum acceptable path probability (confidence threshold)
    detour_slack : unused (kept for API compatibility); Dijkstra naturally
                   finds the globally optimal path without a hop limit
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def plan(
        self,
        env: GridWorld,
        source: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return the most-reliable path from source to the nearest goal,
        or None if no path meets the alpha threshold.
        """
        if source in env.goals:
            return [source]

        # Build a view of the graph with only available doors
        G = env.graph
        available_graph = nx.Graph()
        available_graph.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            if not env.is_failed(u, v):
                w = -math.log(data["prob"])   # weight = -log(p); shorter = more reliable
                available_graph.add_edge(u, v, weight=w, prob=data["prob"])

        best_path: Optional[List] = None
        best_prob: float = -1.0

        for goal in env.goals:
            if not nx.has_path(available_graph, source, goal):
                continue
            try:
                path = nx.dijkstra_path(available_graph, source, goal, weight="weight")
            except nx.NetworkXNoPath:
                continue
            # compute cumulative probability
            prob = 1.0
            for i in range(len(path) - 1):
                prob *= available_graph[path[i]][path[i + 1]]["prob"]
            if prob > best_prob:
                best_prob = prob
                best_path = path

        if best_path is None or best_prob < self.alpha:
            return None
        return best_path


# ═══════════════════════════════════════════════════════════════════════════
# Online Replanning Agent — the outer loop
# ═══════════════════════════════════════════════════════════════════════════

class OnlineReplanningAgent:
    """
    Online plan-act-observe-replan loop.

    This is the OUTER loop and is completely independent of how plans are
    produced.  Any InnerPlanner implementation can be passed in.

    Loop (per episode)
    ------------------
    1. Call inner_planner.plan(env, current_state) → plan
    2. If plan is None → no viable route, give up
    3. Execute plan[1] (the very next step)
    4. Observe outcome: success (move) or failure (door locked)
    5. If failure → increment replan counter, go to 1
    6. If at goal → success

    Metrics tracked per episode
    ---------------------------
    reached_goal  : bool
    actions_taken : int  — total door attempts (including failed ones)
    replans       : int  — number of times a door failure triggered replanning

    Parameters
    ----------
    inner_planner : any InnerPlanner implementation
    env           : GridWorld (shared; agent does NOT own it)
    alpha         : confidence threshold forwarded to inner_planner if needed
    max_steps     : hard cap to prevent infinite loops
    """

    def __init__(
        self,
        inner_planner: InnerPlanner,
        env: GridWorld,
        alpha: Optional[float] = None,
        max_steps: int = 500,
    ):
        self.inner_planner = inner_planner
        self.env = env
        self.alpha = alpha if alpha is not None else env.alpha
        self.max_steps = max_steps

        # episode state — reset by run_episode()
        self.reached_goal: bool = False
        self.actions_taken: int = 0
        self.replans: int = 0

    def run_episode(self, verbose: bool = False) -> Dict:
        """
        Execute one episode.  Caller must call env.reset() beforehand.

        Returns
        -------
        dict with keys: reached_goal (bool), actions_taken (int), replans (int)
        """
        self.reached_goal = False
        self.actions_taken = 0
        self.replans = 0

        while self.actions_taken < self.max_steps:
            current = self.env.current_node

            if current in self.env.goals:
                self.reached_goal = True
                break

            # ── 1. Plan from current state ───────────────────────────────
            plan = self.inner_planner.plan(self.env, current)

            if plan is None:
                if verbose:
                    print(
                        f"  [agent] No viable plan from {current} "
                        f"(alpha={self.alpha:.2f}). Giving up."
                    )
                break

            next_node = plan[1]  # execute only the first step

            if verbose:
                print(
                    f"  [agent] At {current} → plan {plan} "
                    f"| attempting door → {next_node}"
                )

            # ── 2. Act ───────────────────────────────────────────────────
            _, success, done = self.env.step(next_node)
            self.actions_taken += 1

            # ── 3. Observe ───────────────────────────────────────────────
            if verbose:
                status = "OK" if success else "FAILED (door locked)"
                print(f"  [agent] {status}. Now at {self.env.current_node}")

            if done:
                self.reached_goal = True
                break

            # ── 4. Replan if door failed (loop back to top) ──────────────
            if not success:
                self.replans += 1

        return {
            "reached_goal": self.reached_goal,
            "actions_taken": self.actions_taken,
            "replans": self.replans,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience alias
# ═══════════════════════════════════════════════════════════════════════════

def make_baseline_agent(
    env: GridWorld,
    alpha: Optional[float] = None,
    max_steps: int = 500,
) -> OnlineReplanningAgent:
    """
    Construct an OnlineReplanningAgent using the ReliablePathPlanner.
    Convenience function for test scripts and quick experiments.
    """
    a = alpha if alpha is not None else env.alpha
    return OnlineReplanningAgent(
        inner_planner=ReliablePathPlanner(alpha=a),
        env=env,
        alpha=a,
        max_steps=max_steps,
    )
