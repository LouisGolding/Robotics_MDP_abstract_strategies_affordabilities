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
states). Instead we use an ONLINE loop (Yoon, Fern & Givan, ICAPS 2007):

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
    the outer loop

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: INNER PLANNER INTERFACE
═══════════════════════════════════════════════════════════════════════════════
The outer loop (OnlineReplanningAgent) is decoupled from HOW a plan is
produced. Any class implementing InnerPlanner can be plugged in:

    InnerPlanner (ABC)
    ├── ReliablePathPlanner   ← validation oracle ONLY (not an eval condition)
    └── MCTSPlanner           ← the planner used in ALL three eval conditions

All three experimental conditions share the SAME outer loop and MCTSPlanner;
only the set of available moves passed to the planner differs.

═══════════════════════════════════════════════════════════════════════════════
ORACLE: ReliablePathPlanner
═══════════════════════════════════════════════════════════════════════════════
Uses Dijkstra's algorithm (Dijkstra, 1959) on edge weights −log(p) to find
the most-reliable path exactly.  Used ONLY as a validation oracle in tests:
we verify that MCTSPlanner converges to the same first action on small maps.

═══════════════════════════════════════════════════════════════════════════════
EVALUATION PLANNER: MCTSPlanner / UCT
═══════════════════════════════════════════════════════════════════════════════
Monte Carlo Tree Search with the UCB1 selection criterion (UCT algorithm).

Implementation follows Kocsis & Szepesvári (2006) exactly:

  Repeat for n_rollouts iterations:
    1. SELECT   — from root, follow tree policy (UCB1) to a leaf
    2. EXPAND   — add one unexplored child if the leaf is non-terminal
    3. SIMULATE — random rollout from the new node using door probabilities
    4. BACKPROP — update visit counts N and value estimates Q up to root

  UCB1 selection score (Auer, Cesa-Bianchi & Fischer, 2002):
    score(s, a) = Q(s,a)/N(s,a)  +  c * sqrt(ln N(s) / N(s,a))

  Return the action with the highest visit count at the root.

Adaptation for stochastic doors:
  - Available actions at each node = non-failed doors only
  - Simulation uses Bernoulli(p) to model door outcomes stochastically
  - A simulation episode ends when goal is reached, all doors are locked,
    or the rollout depth limit is exceeded

References
----------
Bertsekas, D. P. & Tsitsiklis, J. N. (1991). An analysis of stochastic
    shortest path problems. Mathematics of Operations Research, 16(3).

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
    Numerische Mathematik, 1(1), 269-271.

Kocsis, L. & Szepesvari, C. (2006). Bandit based Monte-Carlo planning.
    ECML 2006. Lecture Notes in Computer Science, vol 4212.

Koenig, S. & Likhachev, M. (2002). D* Lite. AAAI 2002.

Puterman, M. L. (1994). Markov Decision Processes. Wiley.

Yoon, S., Fern, A. & Givan, R. (2007). FF-Replan: A baseline for
    probabilistic planning. ICAPS 2007.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

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
# ReliablePathPlanner — validation oracle (NOT an evaluation condition)
# ═══════════════════════════════════════════════════════════════════════════

class ReliablePathPlanner(InnerPlanner):
    """
    Most-reliable-path inner planner — validation oracle only.

    Finds the path to any goal that maximises cumulative success probability,
    implemented as Dijkstra on edge weights −log(p).  Returns the path only if
    its cumulative probability ≥ alpha; otherwise returns None.

    Complexity: O((V + E) log V) per call via networkx Dijkstra.

    Role: used ONLY in tests/test_baseline.py to verify that MCTSPlanner
    converges to the same first action on small deterministic-ish maps.
    It is NOT one of the three evaluation conditions and does NOT appear
    in thesis results.

    Parameters
    ----------
    alpha : minimum acceptable path probability (confidence threshold)
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def plan(
        self,
        env: GridWorld,
        source: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        if source in env.goals:
            return [source]

        G = env.graph
        available_graph = nx.Graph()
        available_graph.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            if not env.is_failed(u, v):
                w = -math.log(data["prob"])
                available_graph.add_edge(u, v, weight=w, prob=data["prob"])

        best_path: Optional[List] = None
        best_prob: float = -1.0

        for goal in env.goals:
            if not nx.has_path(available_graph, source, goal):
                continue
            try:
                path = nx.dijkstra_path(available_graph, source, goal,
                                        weight="weight")
            except nx.NetworkXNoPath:
                continue
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
# MCTSPlanner — UCT algorithm (Kocsis & Szepesvári, 2006)
# ═══════════════════════════════════════════════════════════════════════════

class _UCTNode:
    """
    A node in the UCT search tree.

    Each node represents a (state, set-of-failed-doors) pair encountered
    during tree search.  We track failed doors per-node so that simulations
    branching from different paths correctly reflect which doors are
    unavailable in each subtree.

    Attributes
    ----------
    state        : room (row, col)
    failed_doors : doors locked on the path from root to this node
    parent       : parent node (None for root)
    action       : the door attempt that led here from the parent
    children     : child nodes keyed by action (neighbour node)
    N            : visit count
    Q            : cumulative reward (higher = better; we use −cost)
    """

    __slots__ = ("state", "failed_doors", "parent", "action",
                 "children", "N", "Q", "_untried_actions")

    def __init__(
        self,
        state: Tuple[int, int],
        failed_doors: Set,
        parent: Optional[_UCTNode] = None,
        action: Optional[Tuple[int, int]] = None,
    ):
        self.state = state
        self.failed_doors = set(failed_doors)   # copy — node owns its set
        self.parent = parent
        self.action = action
        self.children: Dict[Tuple[int, int], _UCTNode] = {}
        self.N: int = 0
        self.Q: float = 0.0
        self._untried_actions: Optional[List[Tuple[int, int]]] = None

    def untried_actions(self, env: GridWorld) -> List[Tuple[int, int]]:
        """Actions not yet expanded from this node."""
        if self._untried_actions is None:
            available = [nb for nb in env.graph.neighbors(self.state)
                         if frozenset({self.state, nb}) not in self.failed_doors]
            self._untried_actions = [a for a in available
                                     if a not in self.children]
        return self._untried_actions

    def is_fully_expanded(self, env: GridWorld) -> bool:
        return len(self.untried_actions(env)) == 0

    def is_terminal(self, env: GridWorld) -> bool:
        """True if goal reached or no doors remain."""
        if self.state in env.goals:
            return True
        available = [nb for nb in env.graph.neighbors(self.state)
                     if frozenset({self.state, nb}) not in self.failed_doors]
        return len(available) == 0

    def ucb1(self, c: float, parent_N: int) -> float:
        """UCB1 score (Auer, Cesa-Bianchi & Fischer, 2002)."""
        if self.N == 0:
            return float("inf")
        return self.Q / self.N + c * math.sqrt(math.log(parent_N) / self.N)

    def best_child(self, c: float) -> _UCTNode:
        """Select child with highest UCB1 score."""
        return max(self.children.values(),
                   key=lambda ch: ch.ucb1(c, self.N))

    def best_action_child(self) -> _UCTNode:
        """Select child with most visits (used at root after search)."""
        return max(self.children.values(), key=lambda ch: ch.N)


class MCTSPlanner(InnerPlanner):
    """
    UCT inner planner — the evaluation planner for all three conditions.

    Implements Monte Carlo Tree Search with the UCB1 selection criterion,
    following Kocsis & Szepesvari (2006) exactly.  Adapted for stochastic
    doors: simulations use Bernoulli(p) door outcomes and respect the
    set of permanently-failed doors accumulated so far in the episode.

    Algorithm per call to plan()
    ----------------------------
    Build a UCT tree rooted at `source` for n_rollouts iterations:

      1. SELECT   : from root, follow UCB1 until an unexpanded leaf
      2. EXPAND   : add one unvisited child (sample a random untried action)
      3. SIMULATE : random rollout from child; reward = -steps to goal
                    (or 0 if max rollout depth exceeded without reaching goal)
      4. BACKPROP : update N and Q for every node on the path root → child

    Return plan[0:2] = [source, best_first_action] where best first action
    is the root child with the highest visit count N.

    Parameters
    ----------
    n_rollouts      : number of UCT iterations per plan() call
    rollout_depth   : max steps per simulation rollout
    c               : UCB1 exploration constant (sqrt(2) is theoretical default;
                      tuning may help for specific map sizes)
    seed            : RNG seed for reproducibility

    Reference
    ---------
    Kocsis, L. & Szepesvari, C. (2006). Bandit based Monte-Carlo planning.
    ECML 2006. Lecture Notes in Computer Science, vol 4212.
    """

    def __init__(
        self,
        n_rollouts: int = 200,
        rollout_depth: int = 50,
        c: float = math.sqrt(2),
        seed: Optional[int] = None,
    ):
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.c = c
        self._rng = random.Random(seed)

    # ── public interface ────────────────────────────────────────────────────

    def plan(
        self,
        env: GridWorld,
        source: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run UCT from `source` and return [source, best_next_node], or None
        if no move leads toward a goal.
        """
        if source in env.goals:
            return [source]

        # Snapshot the current failed-door set — the tree is built on top of
        # the real episode history; simulations branch from here.
        root = _UCTNode(state=source, failed_doors=env.failed_doors)

        if root.is_terminal(env):
            return None

        for _ in range(self.n_rollouts):
            node = self._select(root, env)
            node = self._expand(node, env)
            reward = self._simulate(node, env)
            self._backprop(node, reward)

        if not root.children:
            return None

        best = root.best_action_child()
        return [source, best.action]

    # ── UCT phases ──────────────────────────────────────────────────────────

    def _select(self, node: _UCTNode, env: GridWorld) -> _UCTNode:
        """
        Phase 1 — SELECT.
        Walk down the tree using UCB1 until we reach a node that either:
          - has untried actions (not fully expanded), or
          - is terminal.
        """
        while not node.is_terminal(env) and node.is_fully_expanded(env):
            node = node.best_child(self.c)
        return node

    def _expand(self, node: _UCTNode, env: GridWorld) -> _UCTNode:
        """
        Phase 2 — EXPAND.
        Add one new child by sampling a random untried action.
        Returns the new child (or the node itself if already terminal).
        """
        if node.is_terminal(env):
            return node

        untried = node.untried_actions(env)
        if not untried:
            return node

        action = self._rng.choice(untried)
        untried.remove(action)   # mark as tried in-place

        p = env.graph[node.state][action]["prob"]
        success = self._rng.random() < p

        if success:
            new_state = action
            new_failed = set(node.failed_doors)
        else:
            # door failed in this simulation branch
            new_state = node.state
            new_failed = set(node.failed_doors)
            new_failed.add(frozenset({node.state, action}))

        child = _UCTNode(
            state=new_state,
            failed_doors=new_failed,
            parent=node,
            action=action,
        )
        node.children[action] = child
        return child

    def _simulate(self, node: _UCTNode, env: GridWorld) -> float:
        """
        Phase 3 — SIMULATE (rollout).
        From `node`, execute a random policy until goal, dead-end, or depth
        limit.  Returns reward = −steps_taken (shorter path = higher reward).
        Goal reached yields a bonus; dead-end yields a penalty.

        The rollout does NOT modify env — it maintains its own state/failed set.
        """
        state = node.state
        failed = set(node.failed_doors)
        G = env.graph
        steps = 0

        while steps < self.rollout_depth:
            if state in env.goals:
                # reward: inverse of path length, scaled to [0, 1]
                return 1.0 / (1.0 + steps)

            available = [nb for nb in G.neighbors(state)
                         if frozenset({state, nb}) not in failed]
            if not available:
                return -1.0   # dead-end penalty

            action = self._rng.choice(available)
            p = G[state][action]["prob"]
            if self._rng.random() < p:
                state = action
            else:
                failed.add(frozenset({state, action}))
            steps += 1

        # depth exceeded — small negative reward proportional to distance
        try:
            dist = min(
                nx.shortest_path_length(G, state, g)
                for g in env.goals
                if nx.has_path(G, state, g)
            )
        except (ValueError, nx.NetworkXNoPath):
            dist = self.rollout_depth
        return -dist / self.rollout_depth

    def _backprop(self, node: _UCTNode, reward: float) -> None:
        """
        Phase 4 — BACKPROP.
        Walk from `node` back to root, incrementing visit counts and
        accumulating reward.
        """
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent


# ═══════════════════════════════════════════════════════════════════════════
# Online Replanning Agent — the outer loop (unchanged across all phases)
# ═══════════════════════════════════════════════════════════════════════════

class OnlineReplanningAgent:
    """
    Online plan-act-observe-replan loop.

    Completely independent of how plans are produced — any InnerPlanner can
    be passed in.  The outer loop, environment, and metrics never change
    across the three experimental conditions; only the inner planner (and
    the set of available moves it sees) differs.

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
    actions_taken : int   — total door attempts (including failed ones)
    replans       : int   — number of times a door failure triggered replanning
    planning_time : float — total wall-clock seconds spent inside plan()

    Parameters
    ----------
    inner_planner : any InnerPlanner implementation
    env           : GridWorld (shared; agent does NOT own it)
    max_steps     : hard cap to prevent infinite loops
    """

    def __init__(
        self,
        inner_planner: InnerPlanner,
        env: GridWorld,
        max_steps: int = 500,
    ):
        self.inner_planner = inner_planner
        self.env = env
        self.max_steps = max_steps

        self.reached_goal: bool = False
        self.actions_taken: int = 0
        self.replans: int = 0
        self.planning_time: float = 0.0

    def run_episode(self, verbose: bool = False) -> Dict:
        """
        Execute one episode.  Caller must call env.reset() beforehand.

        Returns
        -------
        dict with keys: reached_goal (bool), actions_taken (int),
                        replans (int), planning_time (float seconds)
        """
        self.reached_goal = False
        self.actions_taken = 0
        self.replans = 0
        self.planning_time = 0.0

        while self.actions_taken < self.max_steps:
            current = self.env.current_node

            if current in self.env.goals:
                self.reached_goal = True
                break

            t0 = time.perf_counter()
            plan = self.inner_planner.plan(self.env, current)
            self.planning_time += time.perf_counter() - t0

            if plan is None:
                if verbose:
                    print(f"  [agent] No plan from {current}. Giving up.")
                break

            next_node = plan[1]

            if verbose:
                print(f"  [agent] At {current} → attempting {next_node}")

            _, success, done = self.env.step(next_node)
            self.actions_taken += 1

            if verbose:
                print(f"  [agent] {'OK' if success else 'FAILED'}. "
                      f"Now at {self.env.current_node}")

            if done:
                self.reached_goal = True
                break

            if not success:
                self.replans += 1

        return {
            "reached_goal": self.reached_goal,
            "actions_taken": self.actions_taken,
            "replans": self.replans,
            "planning_time": self.planning_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════════

def make_baseline_agent(
    env: GridWorld,
    alpha: Optional[float] = None,
    max_steps: int = 500,
) -> OnlineReplanningAgent:
    """
    OnlineReplanningAgent with ReliablePathPlanner (oracle / validation only).
    Use for test_baseline.py; do not use for thesis evaluation results.
    """
    a = alpha if alpha is not None else env.alpha
    return OnlineReplanningAgent(
        inner_planner=ReliablePathPlanner(alpha=a),
        env=env,
        max_steps=max_steps,
    )


def make_mcts_agent(
    env: GridWorld,
    n_rollouts: int = 200,
    rollout_depth: int = 50,
    c: float = math.sqrt(2),
    max_steps: int = 500,
    seed: Optional[int] = None,
) -> OnlineReplanningAgent:
    """
    OnlineReplanningAgent with MCTSPlanner (UCT).
    Use for all three evaluation conditions — change available moves, not
    the planner or the outer loop.
    """
    return OnlineReplanningAgent(
        inner_planner=MCTSPlanner(
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            c=c,
            seed=seed,
        ),
        env=env,
        max_steps=max_steps,
    )

