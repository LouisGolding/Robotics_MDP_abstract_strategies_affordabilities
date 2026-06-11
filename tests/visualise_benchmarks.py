"""
Visualise the 4 benchmark maps used in evaluate.py.

Produces two figures:
  1. benchmark_maps.png      — all 4 maps showing door probabilities (colour-coded)
  2. benchmark_episodes.png  — same 4 maps after one sample episode each,
                               showing the agent's path and any failed doors

Run from repo root:
    python tests/visualise_benchmarks.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from mdp_nav import GridWorld, DoorProbabilitySpec, make_mcts_agent
from tests.evaluate import BENCHMARK_MAPS


# ---------------------------------------------------------------------------
# Shared drawing helper
# ---------------------------------------------------------------------------

def draw_map(env: GridWorld, ax, title: str, path=None) -> None:
    """
    Draw a GridWorld on ax.
    - Edges coloured by probability: green (high) → red (low), grey if failed
    - path: list of (row,col) nodes visited — drawn as a highlighted trail
    """
    G = env.graph
    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    # Node colours
    node_colors = []
    for node in G.nodes():
        if node == env.current_node:
            node_colors.append("#2196F3")   # blue — agent
        elif node in env.goals:
            node_colors.append("#4CAF50")   # green — goal
        elif path and node in path:
            node_colors.append("#FFF176")   # yellow — visited
        else:
            node_colors.append("#EEEEEE")   # light grey

    # Edge colours by probability (green=high, red=low, grey=failed)
    cmap = plt.cm.RdYlGn
    edge_colors, edge_widths, edge_styles = [], [], []
    for u, v, data in G.edges(data=True):
        if env.is_failed(u, v):
            edge_colors.append("#BDBDBD")
            edge_widths.append(1.5)
            edge_styles.append("dashed")
        else:
            p = data["prob"]
            edge_colors.append(cmap(p))
            edge_widths.append(2.5)
            edge_styles.append("solid")

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colors,
                           node_size=500, linewidths=0.8, edgecolors="#888")

    # Draw edges one by one to support per-edge style
    for (u, v, data), ec, ew, es in zip(
            G.edges(data=True), edge_colors, edge_widths, edge_styles):
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=[(u, v)],
                               edge_color=[ec], width=ew, style=es, alpha=0.9)

    # Edge probability labels
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if not env.is_failed(u, v):
            edge_labels[(u, v)] = f"{data['prob']:.2f}"
    nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels=edge_labels,
                                 font_size=5.5, label_pos=0.3,
                                 bbox=dict(boxstyle="round,pad=0.1",
                                           fc="white", alpha=0.6, ec="none"))

    # Node labels
    node_labels = {}
    for node in G.nodes():
        if node == env.current_node:
            node_labels[node] = "A"
        elif node in env.goals:
            node_labels[node] = "G"
        else:
            node_labels[node] = f"{node[0]},{node[1]}"
    nx.draw_networkx_labels(G, pos=pos, ax=ax, labels=node_labels, font_size=7)

    # Highlight path
    if path and len(path) > 1:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=path_edges,
                               edge_color="#2196F3", width=4, alpha=0.5)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.axis("off")


def probability_colorbar(fig, axes_row):
    """Add a shared probability colourbar for a row of axes."""
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_row, shrink=0.7, pad=0.02)
    cbar.set_label("Door probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Figure 1 — benchmark maps (no episodes run, fresh state)
# ---------------------------------------------------------------------------

def figure_maps() -> None:
    n = len(BENCHMARK_MAPS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    fig.suptitle("Benchmark Maps — Door Probabilities\n"
                 "(green = reliable, red = risky; each map has unique random probs)",
                 fontsize=11, y=1.01)

    for ax, bm in zip(axes, BENCHMARK_MAPS):
        env = bm.make_env()
        env.reset(seed=0)
        draw_map(env, ax, bm.name)

    probability_colorbar(fig, axes)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_maps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — sample episodes (agent path + failed doors visible)
# ---------------------------------------------------------------------------

def figure_episodes() -> None:
    n = len(BENCHMARK_MAPS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    fig.suptitle("Sample Episodes — Agent Path & Failed Doors\n"
                 "(blue = agent final pos, yellow = visited, grey dashed = locked door)",
                 fontsize=11, y=1.01)

    for ax, bm in zip(axes, BENCHMARK_MAPS):
        env = bm.make_env()
        env.reset(seed=7)          # fixed seed for reproducibility

        agent = make_mcts_agent(env, n_rollouts=300, seed=0)

        # Track path manually
        path = [env.current_node]
        for _ in range(500):
            if env.current_node in env.goals:
                break
            plan = agent.inner_planner.plan(env, env.current_node)
            if plan is None:
                break
            _, _, done = env.step(plan[1])
            path.append(env.current_node)
            if done:
                break

        outcome = "REACHED GOAL" if env.current_node in env.goals else "STUCK"
        title = f"{bm.name}\n{outcome} — {len(path)-1} actions"
        draw_map(env, ax, title, path=path)

    probability_colorbar(fig, axes)

    legend_elements = [
        mpatches.Patch(color="#2196F3", label="Agent (final)"),
        mpatches.Patch(color="#4CAF50", label="Goal"),
        mpatches.Patch(color="#FFF176", label="Visited room"),
        mpatches.Patch(color="#BDBDBD", label="Locked door"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "benchmark_episodes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — evaluation results bar chart (Phase 2 baseline)
# ---------------------------------------------------------------------------

def figure_results() -> None:
    """
    Bar chart of the Phase 2 baseline results from evaluate.py.
    Update the numbers here after each evaluate.py run.
    """
    maps = [
        "5×5 easy\n(min=0.75)",
        "5×5 medium\n(min=0.50)",
        "5×5 bottleneck",
        "6×6 clustered\n(min=0.50)",
    ]
    # Phase 2 baseline numbers from evaluate.py (100 episodes each)
    success  = [88.0, 77.0, 79.0, 45.0]
    replans  = [2.42, 3.61, 2.74, 5.37]
    actions  = [17.44, 19.15, 14.51, 22.73]

    x = np.arange(len(maps))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Phase 2 Baseline — MDP alone (MCTSPlanner, 100 episodes per map)",
                 fontsize=11)

    for ax, values, ylabel, color, ylim in [
        (axes[0], success,  "Success rate (%)",   "#42A5F5", (0, 105)),
        (axes[1], replans,  "Avg replans",         "#EF5350", (0, None)),
        (axes[2], actions,  "Avg actions taken",   "#66BB6A", (0, None)),
    ]:
        bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(maps, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim[1]:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "baseline_results.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — condition comparison (Phase 2 primitive vs Phase 3 macro)
# ---------------------------------------------------------------------------

def figure_comparison() -> None:
    """
    Grouped bar chart comparing MDP-alone (primitive) against + Macro-actions
    across the four benchmark maps.  Numbers are from a 100-episode-per-cell
    run of evaluate.py; update them here after re-running the harness.

    The headline of Phase 3: macro-actions slash planning *decisions* and
    wall-clock planning time (committing to a corridor collapses several
    plan-act-replan cycles into one), at a modest cost in success rate from
    committing blindly to long roads — the gap that strategies + reliability
    affordances (Phase 4/5) are meant to close.
    """
    maps = [
        "5×5 easy\n(min=0.75)",
        "5×5 medium\n(min=0.50)",
        "5×5 bottleneck",
        "6×6 clustered\n(min=0.50)",
    ]
    # (primitive, macro) per map — from evaluate.py, 100 episodes each
    success   = ([88.0, 77.0, 79.0, 45.0], [83.0, 68.0, 73.0, 40.0])
    decisions = ([17.56, 19.38, 14.72, 23.28], [7.91, 11.76, 9.87, 16.66])
    actions   = ([17.44, 19.15, 14.51, 22.73], [15.62, 18.72, 16.54, 25.17])
    plan_time = ([0.176, 0.164, 0.120, 0.228], [0.039, 0.065, 0.060, 0.112])

    x = np.arange(len(maps))
    width = 0.38

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Phase 2 (primitive)  vs  Phase 3 (+ macro-actions)"
                 " — 100 episodes per map",
                 fontsize=12)

    panels = [
        (axes[0], success,   "Success rate (%)",        (0, 105)),
        (axes[1], decisions, "Avg planning decisions",  (0, None)),
        (axes[2], actions,   "Avg actions taken",       (0, None)),
        (axes[3], plan_time, "Avg planning time (s)",   (0, None)),
    ]

    for ax, (prim, macro), ylabel, ylim in panels:
        b1 = ax.bar(x - width / 2, prim,  width, label="MDP alone",
                    color="#42A5F5", alpha=0.9, edgecolor="white")
        b2 = ax.bar(x + width / 2, macro, width, label="+ Macro-actions",
                    color="#AB47BC", alpha=0.9, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(maps, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim[1]:
            ax.set_ylim(*ylim)
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "results_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating benchmark map visualisations...")
    figure_maps()
    print("Running sample episodes...")
    figure_episodes()
    print("Generating baseline results chart...")
    figure_results()
    print("Generating condition-comparison chart...")
    figure_comparison()
    print("\nDone. PNG files written to tests/")
