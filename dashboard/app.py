"""
Live web dashboard for the stochastic-navigation MDP.

Three views, all driven from the persisted trace store (results/trials.db):

  1. Results — the aggregated comparison table (success / replans / actions /
     decisions / planning time) per benchmark map, exactly mirroring
     tests/evaluate.py's terminal output.
  2. Maps    — every benchmark maze rendered side by side; click one to focus.
  3. Focus   — one maze enlarged, with a scrollable list of every recorded run
     (episode) on that maze.  Click a run and its stored trajectory is replayed
     step by step in the maze: the agent moves room to room, doors light up
     green (opened) or red→grey (locked), and the committed road is drawn ahead.

Replays come straight from the `steps` table written by mdp_nav.trace_store —
no planner is re-run, so what you watch is the exact trajectory evaluate.py
recorded.  A live "run a fresh episode" mode is also kept (reusing the same
GridWorld / OnlineReplanningAgent / MCTSPlanner as evaluate.py).

Run from the repo root:

    pip install -r requirements.txt
    python tests/evaluate.py        # generate results/trials.db first
    python dashboard/app.py         # → http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template
from flask_socketio import SocketIO

from mdp_nav import GridWorld, make_mcts_agent, make_macro_agent
from tests.evaluate import BENCHMARK_MAPS, MCTS_ROLLOUTS, ROLLOUT_DEPTH


app = Flask(__name__)
app.config["SECRET_KEY"] = "mdp-nav-dashboard"
# threading async mode keeps the install light (no eventlet/gevent needed).
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trials.db")


# ---------------------------------------------------------------------------
# Conditions available for the live "run a fresh episode" mode
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


def serialise_map(env: GridWorld, name: str, index: int) -> dict:
    """Describe a benchmark graph for the frontend to render."""
    G = env.graph
    nodes = [{"id": _nid(n), "r": n[0], "c": n[1]} for n in G.nodes()]
    edges = [
        {"u": _nid(u), "v": _nid(v), "prob": round(d["prob"], 3)}
        for u, v, d in G.edges(data=True)
    ]
    return {
        "index": index,
        "name": name,
        "rows": env.rows,
        "cols": env.cols,
        "nodes": nodes,
        "edges": edges,
        "start": _nid(env.start),
        "goals": [_nid(g) for g in env.goals],
    }


def serialise_all_maps() -> list:
    """All benchmark maps, freshly built (seeded → deterministic door probs)."""
    maps = []
    for i, bm in enumerate(BENCHMARK_MAPS):
        env = bm.make_env()
        env.reset(seed=0)
        maps.append(serialise_map(env, bm.name, i))
    return maps


# ---------------------------------------------------------------------------
# Trace-DB access
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _db_exists() -> bool:
    return os.path.exists(DB_PATH)


def _benchmark_signature(bm) -> tuple:
    """A benchmark map is uniquely identified by its shape + difficulty."""
    return (bm.rows, bm.cols, round(bm.min_prob, 4), bm.cluster_size)


def _run_signature(row) -> tuple:
    return (row["rows"], row["cols"], round(row["min_prob"], 4), row["cluster_size"])


def _runs_by_benchmark() -> dict:
    """
    Map each benchmark index → list of (run_id, condition_label) in the DB.

    Runs are matched to benchmarks by configuration signature
    (rows, cols, min_prob, cluster_size), so the dashboard works with any
    existing trials.db without needing a schema change.
    """
    out = {i: [] for i in range(len(BENCHMARK_MAPS))}
    if not _db_exists():
        return out
    sig_to_index = {_benchmark_signature(bm): i for i, bm in enumerate(BENCHMARK_MAPS)}
    conn = _connect()
    try:
        for row in conn.execute(
            "SELECT run_id, condition_label, rows, cols, min_prob, cluster_size FROM runs"
        ):
            idx = sig_to_index.get(_run_signature(row))
            if idx is not None:
                out[idx].append((row["run_id"], row["condition_label"]))
    finally:
        conn.close()
    return out


def get_results() -> list:
    """
    Aggregate the episodes table into the per-map comparison table that
    evaluate.py prints — one block per benchmark map, one row per condition.
    """
    by_bm = _runs_by_benchmark()
    results = []
    if not _db_exists():
        return results
    conn = _connect()
    try:
        for i, bm in enumerate(BENCHMARK_MAPS):
            conditions = []
            by_cond: dict = {}
            for run_id, label in by_bm[i]:
                by_cond.setdefault(label, []).append(run_id)
            for label, run_ids in by_cond.items():
                placeholders = ",".join("?" * len(run_ids))
                agg = conn.execute(
                    f"SELECT COUNT(*) n, "
                    f"AVG(reached_goal) succ, AVG(replans) rep, "
                    f"AVG(actions_taken) act, AVG(decisions) dec, "
                    f"AVG(planning_time) plan "
                    f"FROM episodes WHERE run_id IN ({placeholders})",
                    run_ids,
                ).fetchone()
                if agg["n"]:
                    conditions.append({
                        "label": label,
                        "n": agg["n"],
                        "success": round(agg["succ"] * 100, 1),
                        "replans": round(agg["rep"], 2),
                        "actions": round(agg["act"], 2),
                        "decisions": round(agg["dec"], 2),
                        "plan": round(agg["plan"], 3),
                    })
            results.append({"index": i, "name": bm.name, "conditions": conditions})
    finally:
        conn.close()
    return results


def get_runs_for_map(map_index: int) -> list:
    """Every recorded episode on a benchmark map, for the scrollable run list."""
    by_bm = _runs_by_benchmark()
    runs = []
    if not _db_exists():
        return runs
    run_ids = [rid for rid, _ in by_bm.get(map_index, [])]
    if not run_ids:
        return runs
    conn = _connect()
    try:
        placeholders = ",".join("?" * len(run_ids))
        rows = conn.execute(
            f"SELECT e.episode_id, e.episode_seed, e.reached_goal, "
            f"e.actions_taken, e.replans, e.decisions, e.planning_time, "
            f"r.condition_label "
            f"FROM episodes e JOIN runs r ON e.run_id = r.run_id "
            f"WHERE e.run_id IN ({placeholders}) "
            f"ORDER BY r.condition_label, e.episode_seed",
            run_ids,
        ).fetchall()
        for row in rows:
            runs.append({
                "episode_id": row["episode_id"],
                "seed": row["episode_seed"],
                "condition": row["condition_label"],
                "reached_goal": bool(row["reached_goal"]),
                "actions": row["actions_taken"],
                "replans": row["replans"],
                "decisions": row["decisions"],
                "planning_time": round(row["planning_time"], 3),
            })
    finally:
        conn.close()
    return runs


def _episode_steps(episode_id: int) -> list:
    conn = _connect()
    try:
        return conn.execute(
            "SELECT step_index, source_state, target, success, failed_door, "
            "resulting_state, action_kind, action_payload "
            "FROM steps WHERE episode_id = ? ORDER BY step_index",
            (episode_id,),
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Replay driver — streams a stored trajectory step by step
# ---------------------------------------------------------------------------

def replay_episode_task(episode_id: int, map_index: int, delay: float) -> None:
    steps = _episode_steps(episode_id)

    # Reset the focus maze to a fresh state before replaying.
    env = BENCHMARK_MAPS[map_index].make_env()
    env.reset(seed=0)
    goal_ids = {_nid(g) for g in env.goals}
    socketio.emit("map", serialise_map(env, BENCHMARK_MAPS[map_index].name, map_index))
    socketio.emit("replay_start", {"episode_id": episode_id, "start": _nid(env.start)})

    failed_doors: list = []
    replans = 0
    decisions = 0

    for i, s in enumerate(steps):
        waypoints = json.loads(s["action_payload"]).get("waypoints", [])

        # Reconstruct the planning-decision count: plan() is always called from
        # the current node, so a committed road always starts at the node the
        # agent is standing on.  Hence a step begins a NEW planning decision iff
        # its source equals the road's first waypoint; continuation hops of an
        # already-committed road have source = a later waypoint.
        if waypoints and s["source_state"] == waypoints[0]:
            decisions += 1

        if not s["success"]:
            replans += 1
            if s["failed_door"]:
                u, v = s["failed_door"].split("->")
                failed_doors.append([u, v])

        socketio.emit("step", {
            "source": s["source_state"],
            "target": s["target"],
            "success": bool(s["success"]),
            "done": (i == len(steps) - 1),
            "current": s["resulting_state"],
            "failed_doors": list(failed_doors),
            "plan": waypoints,
            "action_kind": s["action_kind"],
            "actions_taken": i + 1,
            "replans": replans,
            "decisions": decisions,
        })
        socketio.sleep(delay)

    reached = (bool(steps) and steps[-1]["success"] == 1
               and steps[-1]["resulting_state"] in goal_ids)
    # When the agent gives up, the final plan() call (which returned no plan)
    # still counted as a decision but executed no step — add it back so the
    # total matches the stored value.
    total_decisions = decisions if reached else decisions + 1
    socketio.emit("replay_done", {
        "episode_id": episode_id,
        "reached_goal": reached,
        "actions_taken": len(steps),
        "replans": replans,
        "decisions": total_decisions,
    })


# ---------------------------------------------------------------------------
# Live "fresh episode" driver (kept from the original dashboard)
# ---------------------------------------------------------------------------

def run_episode_task(map_index: int, condition_key: str,
                     seed: int, delay: float) -> None:
    benchmark = BENCHMARK_MAPS[map_index]
    label, factory = CONDITIONS[condition_key]

    env = benchmark.make_env()
    env.reset(seed=seed)
    agent = factory(env, n_rollouts=MCTS_ROLLOUTS, rollout_depth=ROLLOUT_DEPTH, seed=0)

    socketio.emit("map", serialise_map(env, benchmark.name, map_index))
    socketio.emit("replay_start", {"episode_id": None, "start": _nid(env.start)})

    def on_step(info: dict) -> None:
        socketio.emit("step", {
            "source": _nid(info["source"]),
            "target": _nid(info["target"]),
            "success": info["success"],
            "done": info["done"],
            "current": _nid(info["current"]),
            "failed_doors": [[_nid(u), _nid(v)] for u, v in info["failed_doors"]],
            "plan": [_nid(n) for n in info["plan"]],
            "action_kind": info["action_kind"],
            "actions_taken": info["actions_taken"],
            "replans": info["replans"],
            "decisions": info["decisions"],
        })
        socketio.sleep(delay)

    result = agent.run_episode(step_callback=on_step)
    socketio.emit("replay_done", {
        "episode_id": None,
        "reached_goal": result["reached_goal"],
        "actions_taken": result["actions_taken"],
        "replans": result["replans"],
        "decisions": result["decisions"],
    })


# ---------------------------------------------------------------------------
# Routes & socket events
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("load")
def handle_load(_data=None):
    """Send everything the UI needs on connect: all maps + the results table."""
    socketio.emit("maps_all", {"maps": serialise_all_maps()})
    socketio.emit("results", {"results": get_results(), "db": _db_exists()})


@socketio.on("select_map")
def handle_select_map(data):
    idx = int(data.get("map_index", 0))
    socketio.emit("runs_list", {
        "map_index": idx,
        "name": BENCHMARK_MAPS[idx].name,
        "runs": get_runs_for_map(idx),
    })


@socketio.on("replay")
def handle_replay(data):
    episode_id = int(data["episode_id"])
    map_index = int(data.get("map_index", 0))
    delay = float(data.get("delay", 0.4))
    socketio.start_background_task(replay_episode_task, episode_id, map_index, delay)


@socketio.on("run")
def handle_run(data):
    """Live mode: run a brand-new episode with the planner."""
    map_index = int(data.get("map_index", 0))
    condition = data.get("condition", "primitive")
    seed = int(data.get("seed", 0))
    delay = float(data.get("delay", 0.4))
    if condition not in CONDITIONS:
        condition = "primitive"
    socketio.start_background_task(run_episode_task, map_index, condition, seed, delay)


if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:5000")
    socketio.run(app, host="127.0.0.1", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
