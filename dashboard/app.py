"""
Live web dashboard for the stochastic-navigation MDP.

Animates an online plan-act-observe-replan episode step by step in the
browser: the agent moves room to room, door probabilities are colour-coded,
failed doors are highlighted, the committed plan (a macro-action road, when
one is chosen) is drawn ahead of the agent, and a results table accumulates
metrics across runs.

It reuses the *exact* same machinery as the evaluation harness — the same
GridWorld, the same OnlineReplanningAgent outer loop, the same MCTSPlanner —
so what you watch is what evaluate.py measures.  Animation is driven by the
step_callback hook on run_episode; no planner code is duplicated here.

Run from the repo root:

    pip install -r requirements.txt
    python dashboard/app.py

then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template
from flask_socketio import SocketIO

from mdp_nav import GridWorld, make_mcts_agent, make_macro_agent
from tests.evaluate import BENCHMARK_MAPS


app = Flask(__name__)
app.config["SECRET_KEY"] = "mdp-nav-dashboard"
# threading async mode keeps the install light (no eventlet/gevent needed).
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


# ---------------------------------------------------------------------------
# Conditions available in the dashboard (mirror evaluate.py's CONDITIONS)
# ---------------------------------------------------------------------------

CONDITIONS = {
    "primitive": ("MDP alone (primitive)", make_mcts_agent),
    "macro":     ("+ Macro-actions",       make_macro_agent),
}


# ---------------------------------------------------------------------------
# Serialisation helpers — tuples → "r,c" strings for the browser
# ---------------------------------------------------------------------------

def _nid(node) -> str:
    return f"{node[0]},{node[1]}"


def serialise_map(env: GridWorld, name: str) -> dict:
    """Describe the current graph for the frontend to render."""
    G = env.graph
    nodes = [{"id": _nid(n), "r": n[0], "c": n[1]} for n in G.nodes()]
    edges = [
        {"u": _nid(u), "v": _nid(v), "prob": round(d["prob"], 3)}
        for u, v, d in G.edges(data=True)
    ]
    return {
        "name": name,
        "rows": env.rows,
        "cols": env.cols,
        "nodes": nodes,
        "edges": edges,
        "start": _nid(env.start),
        "goals": [_nid(g) for g in env.goals],
    }


# ---------------------------------------------------------------------------
# Episode driver — runs in a SocketIO background task, emits per step
# ---------------------------------------------------------------------------

def run_episode_task(map_index: int, condition_key: str,
                     seed: int, delay: float) -> None:
    benchmark = BENCHMARK_MAPS[map_index]
    label, factory = CONDITIONS[condition_key]

    env = benchmark.make_env()
    env.reset(seed=seed)
    agent = factory(env, n_rollouts=200, seed=0)

    socketio.emit("map", serialise_map(env, benchmark.name))
    socketio.emit("episode_start", {
        "map": benchmark.name,
        "condition": label,
        "seed": seed,
        "start": _nid(env.start),
        "goals": [_nid(g) for g in env.goals],
    })

    def on_step(info: dict) -> None:
        socketio.emit("step", {
            "source": _nid(info["source"]),
            "target": _nid(info["target"]),
            "success": info["success"],
            "done": info["done"],
            "current": _nid(info["current"]),
            "failed_doors": [[_nid(u), _nid(v)]
                             for u, v in info["failed_doors"]],
            "plan": [_nid(n) for n in info["plan"]],
            "action_kind": info["action_kind"],
            "actions_taken": info["actions_taken"],
            "replans": info["replans"],
            "decisions": info["decisions"],
        })
        # Yield to the server so the event is flushed, and pace the animation.
        socketio.sleep(delay)

    result = agent.run_episode(step_callback=on_step)

    socketio.emit("episode_done", {
        "map": benchmark.name,
        "condition": label,
        "reached_goal": result["reached_goal"],
        "actions_taken": result["actions_taken"],
        "replans": result["replans"],
        "decisions": result["decisions"],
        "planning_time": round(result["planning_time"], 4),
    })


# ---------------------------------------------------------------------------
# Routes & socket events
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    maps = [{"index": i, "name": m.name}
            for i, m in enumerate(BENCHMARK_MAPS)]
    conditions = [{"key": k, "label": v[0]} for k, v in CONDITIONS.items()]
    return render_template("index.html", maps=maps, conditions=conditions)


@socketio.on("preview")
def handle_preview(data):
    """Render a map without running an episode (for the map dropdown)."""
    idx = int(data.get("map_index", 0))
    benchmark = BENCHMARK_MAPS[idx]
    env = benchmark.make_env()
    env.reset(seed=0)
    socketio.emit("map", serialise_map(env, benchmark.name))


@socketio.on("run")
def handle_run(data):
    map_index = int(data.get("map_index", 0))
    condition = data.get("condition", "primitive")
    seed = int(data.get("seed", 0))
    delay = float(data.get("delay", 0.4))
    if condition not in CONDITIONS:
        condition = "primitive"
    socketio.start_background_task(
        run_episode_task, map_index, condition, seed, delay)


if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:5000")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
