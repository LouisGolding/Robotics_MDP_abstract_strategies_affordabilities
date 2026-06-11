# Live Dashboard

A browser dashboard that **animates** an online plan‑act‑observe‑replan
episode step by step: the agent moves room to room, doors are colour‑coded by
open‑probability, locked doors are greyed out, the committed plan (a purple
**macro road** when the planner picks a macro‑action) is drawn ahead of the
agent, and a results table accumulates metrics across runs.

It is a thin visualisation layer over the *same* machinery the evaluation
harness uses — the same `GridWorld`, the same `OnlineReplanningAgent` outer
loop, the same `MCTSPlanner`. Animation is driven entirely by the
`run_episode(step_callback=…)` hook, so **no planner logic is duplicated**:
what you watch is exactly what `tests/evaluate.py` measures.

## Run

```bash
pip install -r requirements.txt          # installs flask + flask-socketio
python dashboard/app.py                   # → http://127.0.0.1:5000
```

Then open <http://127.0.0.1:5000>.

## Controls

| Control | Effect |
|---|---|
| **Map** | One of the 4 fixed benchmark maps from `evaluate.py` |
| **Condition** | `MDP alone (primitive)` or `+ Macro-actions` (Phase 3) |
| **Episode seed** | Seeds the door outcomes for a reproducible run |
| **Speed** | Delay between animated steps |
| **Run episode** | Animate one full episode; its metrics are appended to the table |

## What the visuals mean

- **Edge colour** — door open‑probability, green (reliable) → red (risky).
- **Grey dashed edge** — a door that failed this episode (permanently closed).
- **Green → red flash** on an edge — the door just attempted (opened / locked).
- **Purple road** — a committed *macro‑action* the planner chose; the agent
  follows it without re‑planning until a door along it fails.
- **Blue road** — a committed primitive step.
- **Live metrics** — Actions (door attempts), Replans (door‑failure‑forced
  re‑plans), Decisions (`plan()` calls). Macro‑actions visibly drive
  **Decisions** down by committing several hops per planning decision.

## How it fits together

```
browser  ──socket.io──▶  app.py  ──▶  GridWorld + OnlineReplanningAgent + MCTSPlanner
   ▲                        │                     │
   └──── step / done ───────┘   run_episode(step_callback=emit-per-step)
```

`app.py` serves `templates/index.html` (a single self‑contained SVG + JS
page) and runs each episode in a Socket.IO background task, emitting one
`step` event per door attempt and an `episode_done` event at the end.
