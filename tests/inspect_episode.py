"""
Inspect a single recorded episode from the trace store (results/trials.db).

Prints the full step-by-step trajectory — source/target rooms, whether each
door opened or locked, the door's open-probability, and the plan MCTS chose —
with a marker on every replanning decision.  Useful for auditing whether the
agent's moves are coherent (re-planning around locked doors) rather than
random wandering.

Usage
-----
    python tests/inspect_episode.py                       # default: 6x6 primitive seed 8
    python tests/inspect_episode.py --map 6x6 --cond primitive --seed 8
    python tests/inspect_episode.py --map 5x5easy --cond macro --seed 3
    python tests/inspect_episode.py --episode 412         # by raw episode_id

Map keys: 5x5easy, 5x5medium, 5x5bottleneck, 6x6   (matched by rows/cols/min_prob)
Cond keys: primitive | macro
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

DB = os.path.join(os.path.dirname(__file__), "..", "results", "trials.db")

# map key -> (rows, cols, min_prob)
MAP_KEYS = {
    "5x5easy":       (5, 5, 0.75),
    "5x5medium":     (5, 5, 0.50),
    "5x5bottleneck": (5, 5, 0.60),
    "6x6":           (6, 6, 0.50),
}
COND_KEYS = {
    "primitive": "MDP alone (primitive)",
    "macro":     "+ Macro-actions",
}


def find_episode(conn, map_key, cond_key, seed):
    rows, cols, min_prob = MAP_KEYS[map_key]
    return conn.execute(
        """SELECT e.* FROM episodes e JOIN runs r ON e.run_id = r.run_id
           WHERE r.condition_label = ? AND r.rows = ? AND r.cols = ?
             AND ABS(r.min_prob - ?) < 1e-6 AND e.episode_seed = ?""",
        (COND_KEYS[cond_key], rows, cols, min_prob, seed),
    ).fetchone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="6x6", choices=list(MAP_KEYS))
    ap.add_argument("--cond", default="primitive", choices=list(COND_KEYS))
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--episode", type=int, default=None,
                    help="look up by raw episode_id instead of map/cond/seed")
    args = ap.parse_args()

    if not os.path.exists(DB):
        sys.exit(f"No database at {DB} — run `python tests/evaluate.py` first.")

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    if args.episode is not None:
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id=?",
                           (args.episode,)).fetchone()
    else:
        ep = find_episode(conn, args.map, args.cond, args.seed)

    if not ep:
        sys.exit("Episode not found — check the DB has the run you expect "
                 "(100-episode evaluate.py run).")

    run = conn.execute("SELECT * FROM runs WHERE run_id=?",
                       (ep["run_id"],)).fetchone()
    door_probs = json.loads(run["door_probs"])
    goals = json.loads(run["goals"])

    print(f"\n{run['condition_label']}  |  {run['rows']}x{run['cols']} "
          f"(min_prob={run['min_prob']})  |  seed={ep['episode_seed']}")
    print(f"start={run['start']}  goal={goals}")
    print(f"reached_goal={bool(ep['reached_goal'])}  actions={ep['actions_taken']}  "
          f"replans={ep['replans']}  decisions={ep['decisions']}\n")

    steps = conn.execute(
        "SELECT step_index, source_state, target, success, resulting_state, "
        "action_kind, action_payload FROM steps WHERE episode_id=? "
        "ORDER BY step_index", (ep["episode_id"],)
    ).fetchall()

    print("Step  Source Target  OK  Result  Prob   Plan (first 4 hops)")
    print("-" * 78)
    revisits = {}
    for s in steps:
        wp = json.loads(s["action_payload"]).get("waypoints", [])
        key  = f"{s['source_state']}->{s['target']}"
        rkey = f"{s['target']}->{s['source_state']}"
        prob = door_probs.get(key) or door_probs.get(rkey) or 0.0
        ok = "OK" if s["success"] else "XX"
        new_plan = bool(wp) and s["source_state"] == wp[0]
        marker = "  <- replan" if new_plan else ""
        print("%4d  %5s %6s  %2s  %6s  %.3f  %-26s%s" % (
            s["step_index"], s["source_state"], s["target"], ok,
            s["resulting_state"], prob, "->".join(wp[:4]), marker))
        revisits[s["resulting_state"]] = revisits.get(s["resulting_state"], 0) + 1

    # quick coherence summary
    print("\nCoherence check:")
    fails = [s for s in steps if not s["success"]]
    print(f"  · {len(steps)} steps, {len(fails)} door failures (forced retreats)")
    most = sorted(revisits.items(), key=lambda kv: -kv[1])[:4]
    print(f"  · most-visited rooms: " +
          ", ".join(f"{r}×{n}" for r, n in most))
    conn.close()


if __name__ == "__main__":
    main()
