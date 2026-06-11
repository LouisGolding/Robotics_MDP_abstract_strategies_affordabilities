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
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from .environment import GridWorld
from .macro_actions import MacroAction, MacroActionLibrary


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

#  An action in the tree is EITHER a primitive door (a neighbour node tuple)
#  OR a MacroAction object.  `_action_key` maps either to a hashable token so
#  children can be keyed uniformly.

def _action_key(action) -> object:
    """Hashable key identifying an action (primitive node or macro)."""
    if isinstance(action, MacroAction):
        return ("macro", action.name)
    return action   # primitive: the neighbour node tuple


class _UCTNode:
    """
    A node in the UCT search tree.

    Each node represents a (state, set-of-failed-doors) pair encountered
    during tree search.  We track failed doors per-node so that simulations
    branching from different paths correctly reflect which doors are
    unavailable in each subtree.

    Actions leaving a node may be primitive door attempts (neighbour nodes)
    or macro-actions; both are stored uniformly, keyed by `_action_key`.
    Enumeration of available actions and terminality live on the planner
    (they depend on the macro library), so this node stays a pure data holder.

    Attributes
    ----------
    state        : room (row, col)
    failed_doors : doors locked on the path from root to this node
    parent       : parent node (None for root)
    action       : the primitive node OR MacroAction that led here from parent
    children     : child nodes keyed by _action_key(action)
    N            : visit count
    Q            : cumulative reward (higher = better; we use −cost)
    untried      : actions not yet expanded (lazily filled by the planner)
    """

    __slots__ = ("state", "failed_doors", "parent", "action",
                 "children", "N", "Q", "untried")

    def __init__(
        self,
        state: Tuple[int, int],
        failed_doors: Set,
        parent: Optional[_UCTNode] = None,
        action=None,
    ):
        self.state = state
        self.failed_doors = set(failed_doors)   # copy — node owns its set
        self.parent = parent
        self.action = action
        self.children: Dict[object, _UCTNode] = {}
        self.N: int = 0
        self.Q: float = 0.0
        self.untried: Optional[List] = None     # filled lazily by planner

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
        macro_library: Optional[MacroActionLibrary] = None,
    ):
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.c = c
        self._rng = random.Random(seed)
        # When provided, macro-actions are offered alongside primitive doors as
        # candidate moves in the tree.  None ⇒ primitive-only (Phase 2 baseline,
        # behaviour identical to before).
        self.macro_library = macro_library

    # ── action enumeration (primitive + macro) ───────────────────────────────

    def _available_actions(self, state, failed) -> List:
        """
        Candidate moves from `state` given the doors locked in `failed`:
        every non-failed primitive door, plus every applicable macro whose
        first hop is still available.  Primitives are neighbour-node tuples;
        macros are MacroAction objects.
        """
        actions: List = []
        # The graph here is the env graph captured via closure in plan(); we
        # read it through self._graph for the duration of one plan() call.
        G = self._graph
        for nb in G.neighbors(state):
            if frozenset({state, nb}) not in failed:
                actions.append(nb)
        if self.macro_library is not None:
            for macro in self.macro_library.applicable(state):
                first = macro.first_hop
                if (G.has_edge(state, first)
                        and frozenset({state, first}) not in failed):
                    actions.append(macro)
        return actions

    def _is_terminal(self, node: _UCTNode, env: GridWorld) -> bool:
        """True if goal reached or no primitive door remains."""
        if node.state in env.goals:
            return True
        # A macro's first hop is itself a primitive door, so "no primitive door
        # available" already implies "no macro available" → terminal.
        for nb in self._graph.neighbors(node.state):
            if frozenset({node.state, nb}) not in node.failed_doors:
                return False
        return True

    def _ensure_untried(self, node: _UCTNode, env: GridWorld) -> List:
        if node.untried is None:
            acts = self._available_actions(node.state, node.failed_doors)
            node.untried = [a for a in acts
                            if _action_key(a) not in node.children]
        return node.untried

    def _apply_action(self, state, failed, action):
        """
        Simulate taking `action` from `state` under door uncertainty.

        Returns (new_state, new_failed).  A primitive succeeds (move) or fails
        (door locked, stay put).  A macro walks its waypoints hop by hop, each
        a Bernoulli(p) door attempt, stopping at the first hop that fails —
        exactly how the online loop will execute a committed macro.
        """
        G = self._graph
        if isinstance(action, MacroAction):
            new_failed = set(failed)
            cur = state
            for nxt in action.waypoints:
                door = frozenset({cur, nxt})
                if door in new_failed or not G.has_edge(cur, nxt):
                    break
                if self._rng.random() < G[cur][nxt]["prob"]:
                    cur = nxt          # door opened — advance
                else:
                    new_failed.add(door)   # door locked — macro stops here
                    break
            return cur, new_failed
        # primitive door
        if self._rng.random() < G[state][action]["prob"]:
            return action, set(failed)
        new_failed = set(failed)
        new_failed.add(frozenset({state, action}))
        return state, new_failed

    @staticmethod
    def _plan_from_action(source, action) -> List[Tuple[int, int]]:
        """
        Turn the chosen root action into a plan for the outer loop.

        Primitive → a single committed step [source, next].
        Macro     → the macro's full committed road [source, w0, w1, ..., term].
        The outer loop executes the road until a door fails, then replans.
        """
        if isinstance(action, MacroAction):
            return [source] + list(action.waypoints)
        return [source, action]

    # ── public interface ────────────────────────────────────────────────────

    def plan(
        self,
        env: GridWorld,
        source: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run UCT from `source` and return the committed plan, or None if no move
        leads toward a goal.  The plan is [source, next] for a primitive best
        move, or [source, ...road] when the best move is a macro-action.
        """
        if source in env.goals:
            return [source]

        # Cache the graph for the duration of this call (read-only access).
        self._graph = env.graph

        # Snapshot the current failed-door set — the tree is built on top of
        # the real episode history; simulations branch from here.
        root = _UCTNode(state=source, failed_doors=env.failed_doors)

        if self._is_terminal(root, env):
            return None

        for _ in range(self.n_rollouts):
            node = self._select(root, env)
            node = self._expand(node, env)
            reward = self._simulate(node, env)
            self._backprop(node, reward)

        if not root.children:
            return None

        best = root.best_action_child()
        return self._plan_from_action(source, best.action)

    # ── UCT phases ──────────────────────────────────────────────────────────

    def _select(self, node: _UCTNode, env: GridWorld) -> _UCTNode:
        """
        Phase 1 — SELECT.
        Walk down the tree using UCB1 until we reach a node that either:
          - has untried actions (not fully expanded), or
          - is terminal.
        """
        while (not self._is_terminal(node, env)
               and len(self._ensure_untried(node, env)) == 0):
            node = node.best_child(self.c)
        return node

    def _expand(self, node: _UCTNode, env: GridWorld) -> _UCTNode:
        """
        Phase 2 — EXPAND.
        Add one new child by sampling a random untried action (primitive door
        or macro-action).  Returns the new child (or the node itself if already
        terminal).
        """
        if self._is_terminal(node, env):
            return node

        untried = self._ensure_untried(node, env)
        if not untried:
            return node

        action = self._rng.choice(untried)
        untried.remove(action)   # mark as tried in-place

        new_state, new_failed = self._apply_action(
            node.state, node.failed_doors, action)

        child = _UCTNode(
            state=new_state,
            failed_doors=new_failed,
            parent=node,
            action=action,
        )
        node.children[_action_key(action)] = child
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
    3. Execute the plan step by step until a door fails, the goal is reached,
       or the plan is exhausted
    4. On a door failure → increment replan counter and re-plan from here
    5. On plan exhaustion (no failure) → re-plan from the new position
    6. If at goal → success

    Committed multi-step plans are how macro-actions earn their keep: a
    primitive plan is just [source, next] (executed as one step, identical to
    the Phase 2 baseline), whereas a macro plan is a whole committed road
    [source, w0, w1, ...] that the agent follows without re-planning between
    successful hops — collapsing several plan-act-replan cycles into one
    decision.

    Metrics tracked per episode
    ---------------------------
    reached_goal  : bool
    actions_taken : int   — total door attempts (including failed ones)
    replans       : int   — number of times a door failure triggered replanning
    decisions     : int   — number of plan() calls (planning decisions made)
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
        self.decisions: int = 0
        self.planning_time: float = 0.0

    def run_episode(
        self,
        verbose: bool = False,
        step_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """
        Execute one episode.  Caller must call env.reset() beforehand.

        Parameters
        ----------
        verbose       : print each step to stdout
        step_callback : optional hook called after every door attempt with a
                        dict describing the step (used by the live dashboard).
                        Keys: source, target, success, done, current,
                        failed_doors (list of [u, v]), plan, action_kind,
                        actions_taken, replans, decisions.

        Returns
        -------
        dict with keys: reached_goal (bool), actions_taken (int),
                        replans (int), decisions (int),
                        planning_time (float seconds)
        """
        self.reached_goal = False
        self.actions_taken = 0
        self.replans = 0
        self.decisions = 0
        self.planning_time = 0.0

        while self.actions_taken < self.max_steps:
            current = self.env.current_node

            if current in self.env.goals:
                self.reached_goal = True
                break

            t0 = time.perf_counter()
            plan = self.inner_planner.plan(self.env, current)
            self.planning_time += time.perf_counter() - t0
            self.decisions += 1

            if plan is None or len(plan) < 2:
                if verbose:
                    print(f"  [agent] No plan from {current}. Giving up.")
                break

            # Whether this decision committed to a macro-road (>1 hop) or a
            # single primitive step — useful context for the dashboard.
            action_kind = "macro" if len(plan) > 2 else "primitive"

            # Execute the committed plan hop by hop until a door fails, the
            # goal is reached, or the road is exhausted.
            for target in plan[1:]:
                pos = self.env.current_node
                if self.env.is_failed(pos, target) or \
                        not self.env.graph.has_edge(pos, target):
                    # Road no longer walkable from here → re-plan.
                    break

                if verbose:
                    print(f"  [agent] At {pos} → attempting {target}")

                _, success, done = self.env.step(target)
                self.actions_taken += 1

                if verbose:
                    print(f"  [agent] {'OK' if success else 'FAILED'}. "
                          f"Now at {self.env.current_node}")

                if not success:
                    self.replans += 1

                if step_callback is not None:
                    step_callback({
                        "source": pos,
                        "target": target,
                        "success": success,
                        "done": done,
                        "current": self.env.current_node,
                        "failed_doors": [sorted(d) for d in
                                         self.env.failed_doors],
                        "plan": plan,
                        "action_kind": action_kind,
                        "actions_taken": self.actions_taken,
                        "replans": self.replans,
                        "decisions": self.decisions,
                    })

                if done:
                    self.reached_goal = True
                    break

                if not success:
                    # Committed road is broken at this door — stop and re-plan.
                    break

            if self.reached_goal:
                break

        return {
            "reached_goal": self.reached_goal,
            "actions_taken": self.actions_taken,
            "replans": self.replans,
            "decisions": self.decisions,
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
    macro_library: Optional[MacroActionLibrary] = None,
) -> OnlineReplanningAgent:
    """
    OnlineReplanningAgent with MCTSPlanner (UCT).
    Use for all three evaluation conditions — change available moves, not
    the planner or the outer loop.  Pass a macro_library to enable the
    macro-action condition (Phase 3); leave it None for the primitive baseline.
    """
    return OnlineReplanningAgent(
        inner_planner=MCTSPlanner(
            n_rollouts=n_rollouts,
            rollout_depth=rollout_depth,
            c=c,
            seed=seed,
            macro_library=macro_library,
        ),
        env=env,
        max_steps=max_steps,
    )


def make_macro_agent(
    env: GridWorld,
    n_rollouts: int = 200,
    rollout_depth: int = 50,
    c: float = math.sqrt(2),
    max_steps: int = 500,
    seed: Optional[int] = None,
    max_macro_length: Optional[int] = None,
) -> OnlineReplanningAgent:
    """
    Phase 3 condition: MCTSPlanner with auto-generated corridor / room-crossing
    macro-actions offered alongside primitive doors.  The macro library is
    built from the env's graph topology (see generate_macro_library).
    """
    from .macro_actions import generate_macro_library
    library = generate_macro_library(env, max_length=max_macro_length)
    return make_mcts_agent(
        env,
        n_rollouts=n_rollouts,
        rollout_depth=rollout_depth,
        c=c,
        max_steps=max_steps,
        seed=seed,
        macro_library=library,
    )

