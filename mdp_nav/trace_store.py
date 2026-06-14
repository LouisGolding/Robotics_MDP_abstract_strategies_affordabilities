"""
Persistent SQLite trace store for MDP navigation runs.

Passive observer: hooks into run_episode(step_callback=...) only.
Does NOT modify MCTSPlanner or OnlineReplanningAgent.

Schema
------
runs     — one row per condition×map×agent configuration
episodes — one row per episode; FK → runs
steps    — one row per door attempt; FK → episodes
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    condition_label TEXT    NOT NULL,
    planner_type    TEXT    NOT NULL,
    params          TEXT    NOT NULL,   -- JSON: n_rollouts, rollout_depth, c, ...
    rows            INTEGER NOT NULL,
    cols            INTEGER NOT NULL,
    cluster_size    INTEGER NOT NULL,
    min_prob        REAL    NOT NULL,
    per_door        TEXT    NOT NULL,   -- JSON
    start           TEXT    NOT NULL,   -- "r,c"
    goals           TEXT    NOT NULL,   -- JSON list of "r,c"
    door_probs      TEXT    NOT NULL    -- JSON: {"r,c->r,c": prob, ...}
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    episode_seed    INTEGER NOT NULL,
    reached_goal    INTEGER NOT NULL,   -- 0/1
    actions_taken   INTEGER NOT NULL,
    replans         INTEGER NOT NULL,
    decisions       INTEGER NOT NULL,
    planning_time   REAL    NOT NULL,
    final_state     TEXT    NOT NULL    -- "r,c"
);

CREATE TABLE IF NOT EXISTS steps (
    step_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(episode_id),
    step_index      INTEGER NOT NULL,
    source_state    TEXT    NOT NULL,   -- "r,c"
    action_kind     TEXT    NOT NULL,   -- 'primitive' | 'macro' | 'strategy'
    action_payload  TEXT    NOT NULL,   -- JSON
    target          TEXT    NOT NULL,   -- "r,c"  (immediate next node attempted)
    success         INTEGER NOT NULL,   -- 0/1
    failed_door     TEXT,               -- "r,c->r,c" or NULL
    resulting_state TEXT    NOT NULL    -- "r,c"
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nid(node: Tuple[int, int]) -> str:
    return f"{node[0]},{node[1]}"


def _edge_key(u: Tuple[int, int], v: Tuple[int, int]) -> str:
    return f"{_nid(u)}->{_nid(v)}"


# ---------------------------------------------------------------------------
# TraceRecorder
# ---------------------------------------------------------------------------

class TraceRecorder:
    """
    Records a full benchmark run to a SQLite database.

    Usage
    -----
    recorder = TraceRecorder("results/trials.db")
    run_id = recorder.begin_run(label, planner_type, params, env)
    for ep_seed in range(n_episodes):
        env.reset(seed=ep_seed)
        cb = recorder.make_step_callback()
        result = agent.run_episode(step_callback=cb)
        recorder.end_episode(run_id, ep_seed, result, env)
    """

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

        # Transient state per episode — populated by make_step_callback()
        self._steps: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Run registration
    # ------------------------------------------------------------------

    def begin_run(
        self,
        condition_label: str,
        planner_type: str,
        params: Dict[str, Any],
        env,
    ) -> int:
        """Insert a run row and return its run_id."""
        G = env.graph
        door_probs = {
            _edge_key(u, v): round(d["prob"], 6)
            for u, v, d in G.edges(data=True)
        }
        spec = env.spec  # DoorProbabilitySpec
        row = (
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            condition_label,
            planner_type,
            json.dumps(params),
            env.rows,
            env.cols,
            getattr(env, "cluster_size", 1),
            spec.min_prob,
            json.dumps({_edge_key(u, v): p for (u, v), p in spec.per_door.items()}),
            _nid(env.start),
            json.dumps([_nid(g) for g in env.goals]),
            json.dumps(door_probs),
        )
        cur = self._conn.execute(
            "INSERT INTO runs (timestamp, condition_label, planner_type, params, "
            "rows, cols, cluster_size, min_prob, per_door, start, goals, door_probs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        self._conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Per-episode step callback
    # ------------------------------------------------------------------

    def make_step_callback(self):
        """Return a fresh step_callback for one episode."""
        self._steps = []
        steps = self._steps  # closure

        def _callback(info: Dict) -> None:
            steps.append({
                "source":    info["source"],
                "target":    info["target"],
                "success":   info["success"],
                "current":   info["current"],
                "action_kind": info["action_kind"],
                "plan":      info["plan"],
                "failed_doors": info["failed_doors"],
            })

        return _callback

    # ------------------------------------------------------------------
    # Episode finalisation
    # ------------------------------------------------------------------

    def end_episode(
        self,
        run_id: int,
        episode_seed: int,
        result: Dict[str, Any],
        env,
    ) -> int:
        """Insert episode + steps rows. Returns episode_id."""
        ep_row = (
            run_id,
            episode_seed,
            int(result["reached_goal"]),
            result["actions_taken"],
            result["replans"],
            result["decisions"],
            round(result["planning_time"], 6),
            _nid(env.current_node),
        )
        cur = self._conn.execute(
            "INSERT INTO episodes "
            "(run_id, episode_seed, reached_goal, actions_taken, replans, "
            "decisions, planning_time, final_state) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ep_row,
        )
        episode_id = cur.lastrowid

        step_rows = []
        for idx, s in enumerate(self._steps):
            src = s["source"]
            tgt = s["target"]
            failed_door = None if s["success"] else _edge_key(src, tgt)
            payload = {
                "waypoints": [_nid(n) for n in s["plan"]],
            }
            step_rows.append((
                episode_id,
                idx,
                _nid(src),
                s["action_kind"],
                json.dumps(payload),
                _nid(tgt),
                int(s["success"]),
                failed_door,
                _nid(s["current"]),
            ))

        self._conn.executemany(
            "INSERT INTO steps "
            "(episode_id, step_index, source_state, action_kind, action_payload, "
            "target, success, failed_door, resulting_state) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            step_rows,
        )
        self._conn.commit()
        self._steps = []
        return episode_id

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
