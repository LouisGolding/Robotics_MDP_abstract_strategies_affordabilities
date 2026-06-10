"""
Graph-based grid-world environment.

Rooms are nodes identified by (row, col) tuples.
Doors are undirected edges. Each door has an open-probability p:
  - On first attempt: succeeds with prob p, fails with prob 1-p.
  - On failure the door is permanently closed for this episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import networkx as nx


# ---------------------------------------------------------------------------
# Door probability specification helpers
# ---------------------------------------------------------------------------

@dataclass
class DoorProbabilitySpec:
    """
    Flexible spec for assigning door open-probabilities.

    Each door gets a probability drawn uniformly from [min_prob, 0.99],
    seeded from the GridWorld seed for reproducibility.  per_door entries
    override the random assignment for specific doors (e.g. forced bottlenecks).

    Parameters
    ----------
    min_prob : lower bound for random door probabilities (default 0.5).
               Below 0.5 a door is more likely to fail than succeed on first
               attempt, making it effectively a wall in a one-attempt world.
    per_door : exact probability overrides for specific doors
    """
    min_prob: float = 0.5
    per_door: Dict[Tuple, float] = field(default_factory=dict)

    # resolved random assignments — populated by GridWorld._build_graph
    _resolved: Dict[Tuple, float] = field(default_factory=dict, repr=False)

    def resolve(self, edges, rng: random.Random) -> None:
        """Assign random probabilities to every edge not in per_door."""
        for u, v in edges:
            key = _canonical(u, v)
            if key not in self.per_door:
                self._resolved[key] = rng.uniform(self.min_prob, 0.99)

    def get(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        key = _canonical(u, v)
        if key in self.per_door:
            return self.per_door[key]
        if key in self._resolved:
            return self._resolved[key]
        return self.min_prob


def _canonical(u, v):
    return (u, v) if u <= v else (v, u)


# ---------------------------------------------------------------------------
# Modular room-cluster generators
# ---------------------------------------------------------------------------

def _add_grid_cluster(G: nx.Graph, rows: int, cols: int,
                      row_offset: int, col_offset: int,
                      spec: DoorProbabilitySpec) -> None:
    """Add a rows×cols sub-grid of rooms to G with given offsets."""
    for r in range(rows):
        for c in range(cols):
            node = (r + row_offset, c + col_offset)
            G.add_node(node)
            if c > 0:
                nb = (r + row_offset, c - 1 + col_offset)
                p = spec.get(node, nb)
                G.add_edge(node, nb, prob=p, label=f"{node}-{nb}")
            if r > 0:
                nb = (r - 1 + row_offset, c + col_offset)
                p = spec.get(node, nb)
                G.add_edge(node, nb, prob=p, label=f"{node}-{nb}")


# ---------------------------------------------------------------------------
# Main GridWorld class
# ---------------------------------------------------------------------------

class GridWorld:
    """
    Grid-world MDP environment.

    Parameters
    ----------
    rows, cols      : grid dimensions
    spec            : DoorProbabilitySpec for edge probabilities
    start           : (row, col) starting node
    goals           : set of (row, col) goal nodes
    alpha           : confidence threshold — planner must find path with
                      cumulative success probability ≥ alpha
    seed            : random seed for reproducibility
    cluster_size    : if > 1, builds world as (rows//cs × cols//cs) tiled
                      clusters of size cs×cs with bridge edges between them
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        spec: Optional[DoorProbabilitySpec] = None,
        start: Tuple[int, int] = (0, 0),
        goals: Optional[Set[Tuple[int, int]]] = None,
        alpha: float = 0.5,
        seed: Optional[int] = None,
        cluster_size: int = 1,
    ):
        self.rows = rows
        self.cols = cols
        self.spec = spec or DoorProbabilitySpec()
        self.start = start
        self.goals = goals or {(rows - 1, cols - 1)}
        self.alpha = alpha
        self.seed = seed
        self.cluster_size = cluster_size

        self._base_graph: nx.Graph = nx.Graph()
        self._build_graph()

        # episode state — reset() initialises these
        self._rng = random.Random(seed)
        self.current_node: Tuple[int, int] = start
        self.failed_doors: Set[FrozenSet] = set()  # frozenset({u, v})

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> None:
        # Collect all edges first so spec.resolve() can assign probs in one pass
        cs = self.cluster_size
        edge_set: List[Tuple] = []
        if cs <= 1:
            for r in range(self.rows):
                for c in range(self.cols):
                    node = (r, c)
                    if c > 0:
                        edge_set.append(_canonical(node, (r, c - 1)))
                    if r > 0:
                        edge_set.append(_canonical(node, (r - 1, c)))
        else:
            cluster_rows = self.rows // cs
            cluster_cols = self.cols // cs
            for cr in range(cluster_rows):
                for cc in range(cluster_cols):
                    for r in range(cs):
                        for c in range(cs):
                            node = (cr * cs + r, cc * cs + c)
                            if c > 0:
                                edge_set.append(_canonical(node, (cr * cs + r, cc * cs + c - 1)))
                            if r > 0:
                                edge_set.append(_canonical(node, (cr * cs + r - 1, cc * cs + c)))
            for cr in range(cluster_rows):
                for cc in range(cluster_cols - 1):
                    for r in range(cs):
                        u = (cr * cs + r, cc * cs + cs - 1)
                        v = (cr * cs + r, (cc + 1) * cs)
                        edge_set.append(_canonical(u, v))
            for cr in range(cluster_rows - 1):
                for cc in range(cluster_cols):
                    for c in range(cs):
                        u = (cr * cs + cs - 1, cc * cs + c)
                        v = ((cr + 1) * cs, cc * cs + c)
                        edge_set.append(_canonical(u, v))

        # Resolve random probabilities using a dedicated seeded RNG
        prob_rng = random.Random(self.seed)
        self.spec.resolve(edge_set, prob_rng)

        if cs <= 1:
            _add_grid_cluster(self._base_graph,
                              self.rows, self.cols, 0, 0, self.spec)
        else:
            # Build tiled clusters; add bridge edges between adjacent clusters
            cluster_rows = self.rows // cs
            cluster_cols = self.cols // cs
            for cr in range(cluster_rows):
                for cc in range(cluster_cols):
                    _add_grid_cluster(
                        self._base_graph, cs, cs,
                        cr * cs, cc * cs, self.spec
                    )
            # Horizontal bridges: right edge of cluster → left edge of next
            for cr in range(cluster_rows):
                for cc in range(cluster_cols - 1):
                    for r in range(cs):
                        u = (cr * cs + r, cc * cs + cs - 1)
                        v = (cr * cs + r, (cc + 1) * cs)
                        p = self.spec.get(u, v)
                        self._base_graph.add_edge(u, v, prob=p)
            # Vertical bridges
            for cr in range(cluster_rows - 1):
                for cc in range(cluster_cols):
                    for c in range(cs):
                        u = (cr * cs + cs - 1, cc * cs + c)
                        v = ((cr + 1) * cs, cc * cs + c)
                        p = self.spec.get(u, v)
                        self._base_graph.add_edge(u, v, prob=p)

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[int, int]:
        """Start a new episode. Returns starting node."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.current_node = self.start
        self.failed_doors = set()
        return self.current_node

    def available_doors(
        self, node: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Returns list of (neighbour, prob) for doors not yet failed
        from `node` (defaults to current_node).
        """
        node = node if node is not None else self.current_node
        result = []
        for nb in self._base_graph.neighbors(node):
            key = frozenset({node, nb})
            if key not in self.failed_doors:
                p = self._base_graph[node][nb]["prob"]
                result.append((nb, p))
        return result

    def step(
        self, target: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], bool, bool]:
        """
        Agent attempts to move through the door to `target`.

        Returns
        -------
        new_state : current node after attempt
        success   : True if the door opened
        done      : True if now at a goal node
        """
        if not self._base_graph.has_edge(self.current_node, target):
            raise ValueError(
                f"No door between {self.current_node} and {target}"
            )
        key = frozenset({self.current_node, target})
        if key in self.failed_doors:
            raise ValueError(
                f"Door {self.current_node}↔{target} already permanently closed"
            )

        p = self._base_graph[self.current_node][target]["prob"]
        success = self._rng.random() < p
        if not success:
            self.failed_doors.add(key)
        else:
            self.current_node = target

        done = self.current_node in self.goals
        return self.current_node, success, done

    def edge_prob(
        self, u: Tuple[int, int], v: Tuple[int, int]
    ) -> Optional[float]:
        """Return door probability, or None if no such edge."""
        if self._base_graph.has_edge(u, v):
            return self._base_graph[u][v]["prob"]
        return None

    def is_failed(self, u: Tuple[int, int], v: Tuple[int, int]) -> bool:
        return frozenset({u, v}) in self.failed_doors

    @property
    def graph(self) -> nx.Graph:
        return self._base_graph

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def render_text(self) -> str:
        """ASCII grid showing agent position, goals, and failed doors."""
        lines = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                node = (r, c)
                if node == self.current_node:
                    cell = " A "
                elif node in self.goals:
                    cell = " G "
                else:
                    cell = " . "
                row_cells.append(cell)
                # horizontal door to the right
                if c < self.cols - 1:
                    right = (r, c + 1)
                    if self._base_graph.has_edge(node, right):
                        if self.is_failed(node, right):
                            row_cells.append("X")
                        else:
                            p = self._base_graph[node][right]["prob"]
                            row_cells.append("|" if p >= 0.5 else "?")
                    else:
                        row_cells.append(" ")
            lines.append("".join(row_cells))

            # vertical doors
            if r < self.rows - 1:
                vrow = []
                for c in range(self.cols):
                    node = (r, c)
                    below = (r + 1, c)
                    if self._base_graph.has_edge(node, below):
                        if self.is_failed(node, below):
                            vrow.append(" X ")
                        else:
                            p = self._base_graph[node][below]["prob"]
                            vrow.append(" - " if p >= 0.5 else " ~ ")
                    else:
                        vrow.append("   ")
                    if c < self.cols - 1:
                        vrow.append(" ")
                lines.append("".join(vrow))
        return "\n".join(lines)

    def render_matplotlib(
        self, ax=None, title: str = "GridWorld"
    ):
        """Draw the graph on a matplotlib axes."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if ax is None:
            _, ax = plt.subplots(figsize=(self.cols * 1.2, self.rows * 1.2))

        pos = {node: (node[1], -node[0]) for node in self._base_graph.nodes()}

        node_colors = []
        for node in self._base_graph.nodes():
            if node == self.current_node:
                node_colors.append("tab:blue")
            elif node in self.goals:
                node_colors.append("tab:green")
            else:
                node_colors.append("lightgrey")

        edge_colors, edge_styles, edge_widths = [], [], []
        for u, v, data in self._base_graph.edges(data=True):
            if self.is_failed(u, v):
                edge_colors.append("red")
                edge_styles.append("dashed")
                edge_widths.append(1.0)
            else:
                p = data["prob"]
                edge_colors.append("black" if p >= 0.5 else "orange")
                edge_styles.append("solid")
                edge_widths.append(1.5)

        nx.draw_networkx(
            self._base_graph, pos=pos, ax=ax,
            node_color=node_colors, edge_color=edge_colors,
            style=edge_styles, width=edge_widths,
            node_size=400, font_size=7,
        )

        legend = [
            mpatches.Patch(color="tab:blue", label="Agent"),
            mpatches.Patch(color="tab:green", label="Goal"),
            mpatches.Patch(color="red", label="Failed door"),
            mpatches.Patch(color="orange", label="Risky door (p<0.5)"),
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=7)
        ax.set_title(title)
        return ax
