# Project Log & Knowledge Base — README

**Student:** Louis Golding (Advanced Computing MSc)
**Supervisor:** Dr. Khen Elimelech — Autonomous Robots Lab
**Working title:** *Investigating Abstract Strategies for Online Planning in Non‑Deterministic (MDP) Environments*

> **Purpose of this file.** The single source of truth for this thesis project. It records settled decisions, the reasoning behind them, and the full history of changes. If any other document in the knowledge base (including older proposals) contradicts the canonical block below, **the canonical block wins**. When a decision changes, update the canonical block to the new truth and log the reversal in the Changelog with a date and reason.
>
> **📌 Documentation policy (standing instruction).** Every substantive discussion in this project — design debates, direction decisions, "which approach should we use" forks, things we tried and discarded, and the *reasoning* behind each — must be recorded in the **Development Journal** below, even if the idea didn't pan out. Discarded paths are valuable: they explain *why* the final choices were made and become the "alternatives considered" material for the thesis. A large knowledge base is fine and expected; completeness matters more than brevity here. **This logging happens automatically every time, without needing to be asked** — whenever a chat produces an interesting decision or debate, add a dated journal entry before the conversation ends.

---

## ⚠ CANONICAL — current state of truth

*Last updated: 04 Jun 2026*

**Environment.** A static, non‑deterministic navigation environment with repeatable structure (rooms connected by doors). Each door has a known probability of opening. **One attempt per door** — a failed door is not retried; the planner routes around it. No keys, no temporally‑extended goals. Goal = reach a target state with confidence ≥ α (e.g. 0.80).

**Approach.** Online MDP planning. Build on the **concepts and research** from Khen's papers (strategies, affordances, road maps, abstraction keys, ACSD) — but we do **not have access to their code**. We implement our own planner: take an **off‑the‑shelf MDP solver** (MCTS‑style; need not be state‑of‑the‑art) and integrate the strategy/affordance ideas into it. Plan → act → observe → replan. Balance confidence vs. cost. Strategies replace random expansion with experience‑informed expansion.

**Scope boundaries.** Strategies are **assumed given** for now. No grounding / reconstruction work. Stay at the task level of abstraction. The lifelong library is the vision, but start with a fixed library of provided strategies.

**Contribution.** (1) Qualitative: reframing strategy‑based planning for uncertainty — applying a strategy no longer guarantees reaching its end state. (2) Quantitative: extend the **affordance vector with a reliability score** (not just usefulness) so the planner prefers strategies that are both useful and reliable under uncertainty.

**Evaluation.** Three‑way comparison on the same problem set: **MDP alone** (primitive actions) vs. **MDP + macro‑actions** (sequences of actions) vs. **MDP + abstract strategies** (sequences of states). Success = **fewer replans + lower total cost to the goal**.

**NOT in scope (for now).** Offline policy computation. Grounding / reconstruction of strategies. We do not have access to the prior work's codebase and are not trying to modify it — we build our own implementation informed by the research. Dynamic environments (retries would make it dynamic — explicitly excluded).

---

## Changelog

*Append‑only. When a canonical decision changes, update the canonical block above AND add a dated entry here explaining what changed and why.*

| Date | What changed | Why |
|---|---|---|
| 04 Jun 2026 | CLAUDE.md affordance wording corrected to match canon (rank/select strategies, not filter actions); roadmap split so MCTS‑with‑primitives is its own step before macro‑actions | Doc realignment to the 04 Jun decisions; underlying decision already in the 04 Jun Journal entry |
| 04 Jun 2026 | Locked: all three eval conditions use the SAME MCTS planner; Dijkstra is scaffolding/oracle only, never a condition | Controlled experiment — avoid the planner-vs-moves confound. See Journal 2026‑06‑04 |
| 04 Jun 2026 | Corrected affordance framing (selects strategies, not filters actions) + phase sequencing | CLAUDE.md review. See Journal 2026‑06‑04 |
| 03 Jun 2026 | Baseline = online plan‑replan loop with swappable inner planner; Dijkstra most‑reliable‑path kept as scaffolding, MCTS/UCT the target (pending Khen) | See Development Journal 2026‑06‑03 |
| 03 Jun 2026 | Reaffirmed: we are doing **online**, not offline (offline is the later extension) | Corrected a direction mix‑up; matches Khen's notes |
| 03 Jun 2026 | Added Documentation Policy + Development Journal | Keep full traces of debates/decisions for the thesis writeup |
| 31 May 2026 | Clarified: we build on Khen's *research/concepts*, not their code (no access to it) | User clarification — "don't rewrite" means don't try to modify a codebase we don't have; we implement our own planner informed by the papers |
| 31 May 2026 | Canonical block + changelog added to the log | Single source of truth for the full thesis lifecycle |
| 31 May 2026 | Slides for Meeting 1 and Meeting 2 produced | Grader traceability |
| 28 May 2026 | One‑try‑per‑door locked; retry budget dropped from proposal | Khen (Meeting 2): retries make the env dynamic; keep it static |
| 28 May 2026 | Approach: build our own MDP solver, integrate strategy concepts from the papers | Meeting 1: don't try to modify a prior codebase; build on the research instead |
| 28 May 2026 | Grounding marked out of scope for now | Meeting 2: assume strategies are given; stay at task‑level abstraction |
| 28 May 2026 | Three‑way evaluation defined (MDP / macro‑actions / strategies) | Meeting 2: macro‑actions as the middle comparison point |
| 28 May 2026 | Reliability added as a new affordance dimension | Meeting 2: candidate contribution — extend affordance vector |

---

## Development Journal

*Rich, dated entries capturing discussions, debates, and reasoning — including paths not taken. This is the raw material for the thesis writeup (especially "background," "method justification," and "alternatives considered"). Append new entries at the top.*

### 2026‑06‑09 — Added CODE_GUIDE.md (human-readable code companion)

**Context.** Louis asked for a standalone, human-readable explainer of the codebase to consult when in doubt. Created `code/Robotics_MDP_abstract_strategies_affordabilities/CODE_GUIDE.md` (verified against source; all cited identifiers exist). It covers: the one architectural idea (outer loop vs swappable `InnerPlanner`), the theory-in-one-page (SSP-MDP table), a repo map, a file-by-file WHY for every class, a one-episode orchestration trace, a who-uses-whom dependency map, a theory↔code concordance table, the phase status, and a "known gotchas" list. Doc hierarchy now: CLAUDE.md (AI/canonical) · CODE_GUIDE.md (humans/implementation) · PROJECT_LOG.md (decisions) · REFERENCES.md (evidence).

**Gotchas captured in the guide for later action** (documented, not yet changed): (1) `planner.py` line 18 calls the setup "partial observability" — wrong; we are a *fully observable* MDP (only the transition outcome is uncertain). Reword before it reaches the thesis. (2) `AffordanceVector.total` is a naive unweighted sum — the real combination rule (weight? threshold/basin-of-attraction? multiplicative penalty?) is the core Phase-5 modelling decision, still unmade. (3) `planner.py` docstring cites D*-Lite / FF-Replan loosely — tighten before citing.

### 2026‑06‑09 — Relevance audit of REFERENCES.md (+ one discard: ABT)

**Context.** Louis questioned whether every reference is *directly* relevant and asked for a per-entry verdict, discarding any that aren't. Did a full audit and tagged each entry **Core / Supporting**, added an at-a-glance table at the top of REFERENCES.md.

**Tiers.** Core = a thesis claim / the method / an eval condition depends on it (8: elimelech2022wafr, elimelech2022, elimelech2024, bertsekas1991, suttonbarto2018, kocsis2006, sutton1999options, liang2024). Supporting = strengthens but nothing hinges on it (8: elimelech2023, kurniawati2022, silver2010pomcp, yoon2007, barto1995, he2010puma, browne2012, papadimitriou1991).

**Discarded — kurniawati2013abt (ABT).** Its sole role was a static-vs-dynamic *contrast*; that scope boundary is our own design choice and barely needs a citation, and kurniawati2022 §3.1 already describes ABT. Carrying the ABT primary too was redundant → removed (left an in-file tombstone with the full citation so it isn't silently re-added). The contrast role was folded into the kurniawati2022 entry.

**Kept but flagged as next-weakest** (so Louis can trim later if space is tight): browne2012 (convenience MCTS survey) and barto1995 (only surfaces if the "why not RTDP" alternatives-considered paragraph is written). Both have legitimate roles, so not cut unilaterally.

**Caveats made explicit during the audit.** kurniawati2022 and silver2010pomcp are *POMDP* (partial observability); we are a *fully observable* MDP — cite them for landscape / "MCTS-handles-uncertainty," not as our formalism.

### 2026‑06‑09 — Proposal edited: retries removed entirely (one‑try‑per‑door)

**Context.** Louis asked to remove every mention of *retries* from the refined proposal so it matches the canonical one‑try‑per‑door decision (Changelog 28 May 2026). This goes beyond next‑action §10.1 ("edit the retry budget out") — retries are now gone completely, not merely bounded.

**What changed.** Rewrote three locations of `Robotics Project proposal Louis Golding.pdf`: §2.2 (dropped "retries as a decision variable"; replaced with the route‑around‑and‑replan framing), §3 (replaced the two retry paragraphs — "agent may retry a door," "retries appear in the search tree as repeated expansions with decreasing budget" — with one‑try‑per‑door + permanently‑closed‑on‑failure, keeping the forward‑tree‑search description), and §4 challenge 1 ("Avoiding infinite loops / enforce finite retries" → "Reasoning about irreversible outcomes," since a permanently‑closed door removes the infinite‑loop concern). Everything else left untouched — deliberately surgical; the "adapt Khen's algorithms" wording (open tension §9) was **not** changed, as Louis scoped this to retries only.

**Output.** New file `Robotics Project proposal Louis Golding (updated).pdf` (3 pp.), generated with reportlab + DejaVu font so α / ≥ / − render. Original PDF left in place (not overwritten) for reversibility. Verified: `grep "retr"` → 0 matches; glyphs confirmed; visual check of p.1 clean.

**Still open (unchanged by this edit).** §9 open tension: the proposal's "adapt prior‑work algorithms" framing still conflicts with canon ("build our own planner informed by the concepts"). Flag for a future pass if Louis wants the proposal fully reconciled with canon.

### 2026‑06‑09 — Session handover: orientation, 🔴 passage extraction, WAFR flagged, CTP reduction confirmed

**Context.** New session took over the project; read PROJECT_LOG.md, REFERENCES.md, the three Elimelech PDFs (extracted via `pdftotext`), the current proposal, and the `code/` repo (CLAUDE.md + module stubs). Confirmed code state matches canon: Phase 1 done (GridWorld env + online replan loop + `ReliablePathPlanner` Dijkstra oracle); MCTS not yet coded; `macro_actions.py`/`strategies.py`/`affordances.py` are docstring stubs. No canonical changes this session, so no Changelog row.

**1. 🔴 passages extracted (REFERENCES.md updated).** Fetched the source papers (none were in `papers/`) and added real section-level locators:
- *kurniawati2022* (arXiv:2107.07599v1): §2 example domains (Underwater Nav / Manipulation / HRC) and the **Goal-POMDP = Shortest-Path POMDP** definition (supports our SSP framing); §3 five computational difficulties + PSPACE-hardness; §3.1 offline (PBVI/Perseus/HSVI2/SARSOP) vs online (RTDP-Bel/POMCP/ABT/DESPOT) taxonomy + benchmark domains (Tag, RockSample); §3.2 macro-actions for long horizon (supports our +macro-actions condition); §4.3 lessons-learned stat (100% vs 35% success with/without uncertainty at ICRA'18).
- *liang2024* (arXiv:2411.07032v1): the "enumeration is the fundamental constraint" claim (abstract/§2.3/§3); Alg. 1 (generic SBMP-macro-action integration); Alg. 2 + SampleMacroActionSBMP (§4.3); §5.1 benchmark domains (Light-Dark, Maze2D, Random3D, Multi-Drone Tag).
- *silver2010pomcp*: page range verified (NeurIPS 2010, pp. 2164–2172); method captured via a verified secondary description in kurniawati2022 §3.1 (PO-UCT + generative model + particle belief + rollout). Primary PDF fetch timed out — primary internal § locators still 🔴.

**2. Missing foundation paper flagged.** *elimelech2022wafr* ("Automatic Cross-Domain Task Plan Transfer by Caching Abstract Skills," WAFR 2022, DOI 10.1007/978-3-031-21090-7_28) is **not** in `papers/`. Public preprint: http://kavrakilab.org/publications/elimelech2022-wafr-skills.pdf — Louis to download into `papers/` so its passages (original definition of abstract skill + abstraction key) can be extracted.

**3. Canadian Traveller Problem reduction — confirmed, KEEP.** Verified *papadimitriou1991* "Shortest Paths Without a Map" (TCS 84(1):127–150) **is** the CTP paper; stochastic CTP is #P-hard, worst-case competitive-ratio version PSPACE-complete. Our env (graph of doors blocked with known probability, status revealed only on attempt, one try per door = permanently closed, no reset) is exactly a **stochastic CTP instance** — so the reference stays as the hardness/motivation anchor. Caveat noted: we add a confidence-threshold (chance) constraint on unit-cost steps, which makes ours a CTP *variant* rather than the vanilla cost-minimisation objective.

**Status.** Orientation summary delivered to Louis; awaiting his confirmation before any MCTS (Phase 2) coding, per his instruction.

### 2026‑06‑04 — Experimental control: one planner across all three conditions (+ CLAUDE.md corrections)

**Context.** Reviewed the CLAUDE.md Claude Code generated. It's mostly correct (SSP MDP framing, online plan‑replan loop, swappable inner planner, MCTS as target, Dijkstra as scaffolding). Four clarifications were fed back.

**1. The baseline must be MCTS‑with‑primitives, not Dijkstra (most important).** The three evaluation conditions — primitives / +macro‑actions / +strategies — must all run the **same MCTS planner**; only the available *moves* differ. Rationale is controlled experimentation: if condition 1 used the Dijkstra `ReliablePathPlanner` while conditions 2–3 used MCTS, any performance gap could be caused by the *planner change* rather than the *moves* — a confound that makes the result uninterpretable (an examiner would reject it: two variables changed at once). Holding the planner fixed isolates the one independent variable (move vocabulary), so a measured improvement is attributable to macro‑actions/strategies. Consequence: `ReliablePathPlanner` (Dijkstra) is **scaffolding + a validation oracle only** and never appears as an evaluation condition.

**2. Affordance framing corrected.** Affordances are *not* "filtering implausible primitive actions before search." An affordance is a predictive score of how beneficial a **strategy** is from the current state — it defines a strategy's basin of attraction and is used to **rank/select applicable strategies**. Our reliability dimension predicts how reliably a strategy can be grounded under uncertainty.

**3. Phase sequencing.** Land **MCTS‑with‑primitives first** (this *is* the "MDP alone" baseline; validate it against the Dijkstra oracle on small maps), then add macro‑actions, then strategies, then affordance‑based strategy selection.

**4. Decision vs. implementation — two different "trees."** MCTS is *decided* as the planner but not yet *coded*; current code has only the Dijkstra scaffolding (Phase 1 done), and MCTS is Phase 2 — no reversal, just sequencing. Note two distinct trees that must not be conflated: Dijkstra's exact shortest‑path/candidate‑path tree (now, deterministic) vs. the MCTS Monte‑Carlo search tree (later, sampled, strategy‑guided). A visualization of Dijkstra's candidate paths + branch probabilities + chosen path is a good problem‑setup figure, but must be captioned as the Dijkstra baseline, not MCTS.

**Status.** Corrections fed to Claude Code.

### 2026‑06‑03 — Baseline planner: which algorithm, and online vs. offline

**Context.** Claude Code scaffolded the repo and, for the Phase‑1 baseline, implemented a most‑reliable‑path planner. It then paused to ask which algorithm we should formally standardize on as the "MDP alone" baseline, offering Value Iteration vs. determinize‑and‑replan (FF‑Replan) vs. both.

**The Dijkstra scaffolding (important to record).** The initial setup computes the *most reliable path* using **Dijkstra's shortest‑path algorithm with edge weights = −log(door probability)**. The trick: minimizing the sum of −log(p) equals maximizing the product of probabilities, so Dijkstra returns the single highest‑success‑probability path. This is correct and useful **only as scaffolding** to get the environment and the online replan loop working on small maps. *It is not the target method* — it finds one path on a small graph and does not reason about the full sequential, replan‑able problem, and it does not scale or give strategies a place to attach. We keep it as a simple starting baseline but it is essentially a stepping stone, not the contribution.

**Options debated for the real baseline.**
- *Value Iteration (true/textbook MDP, offline policy).* Maximally recognizable to a committee, but it computes a full offline policy — which contradicts the online framing — and the **one‑try‑per‑door rule makes the state space (room × set of failed doors) exponential**, so exact VI is intractable beyond toy sizes. Possible *future* use: compute VI‑optimal expected cost on small instances as a "gold‑standard" reference only.
- *FF‑Replan / determinize‑and‑replan (Yoon, Fern & Givan 2007).* Pretends the world is deterministic, plans, and replans on failure; ignores probabilities. Simple and citable, but strategies don't attach to it cleanly.
- *MCTS / UCT (Kocsis & Szepesvári 2006).* Samples possible futures respecting door probabilities, scales by sampling instead of enumerating, and — decisively — **Khen explicitly described "building a Monte Carlo tree" and strategies guiding which branches to expand instead of choosing randomly.** Strategies plug into MCTS naturally (they bias expansion). This is the substrate the contribution needs.

**Decision (pending Khen confirmation).** Architect the planner as an **online plan‑act‑observe‑replan loop** with a **swappable inner planner**. Keep the Dijkstra most‑reliable‑path planner as the simple starting baseline; make **MCTS/UCT** the target inner planner that strategies will later guide. All three evaluation conditions (primitives / +macro‑actions / +strategies) share this same online loop and inner‑planner interface; only the available "moves" differ. To raise with Khen: *"Using MCTS as the base planner since you mentioned building a Monte Carlo tree, with strategies guiding expansion — and modeling the problem as a stochastic shortest‑path MDP. Right?"*

**Online vs. offline — corrected.** A point of confusion was cleared up: Khen said **online first**, and offline is the *later* "if it works well" extension — not the other way around. (Notes: *"we are looking at ONLINE MDPs… build a plan, not a policy"*; *"Online algorithm. If works well we can take it to offline."*) Offline = compute a full policy for every state before moving (e.g. Value Iteration); online = plan from the current state, act, observe, replan. **We are doing online.** The success metric ("fewer replans, lower total cost") only exists for online planning — an offline policy never replans — so the whole design is consistent around online.

**Possible thesis framing anchor.** The problem (navigate a graph where doors may be blocked, status revealed only on attempt, one try each) resembles the **Stochastic Canadian Traveller Problem** (Papadimitriou & Yannakakis 1991) — a recognized, provably hard problem. Worth checking whether our variant maps onto it; if so, it motivates *why* exact optimal solving is intractable and *why* online planning + learned strategies is a sensible response.

**References surfaced.** UCT — Kocsis & Szepesvári 2006. MCTS survey — Browne et al. 2012. RTDP (alt. online MDP algorithm) — Barto, Bradtke & Singh 1995. FF‑Replan — Yoon, Fern & Givan 2007. Stochastic Shortest Path — Bertsekas & Tsitsiklis 1991. MDP formalism — Sutton & Barto 2018. Canadian Traveller Problem — Papadimitriou & Yannakakis 1991. (Verify each before citing in the thesis.)

---

## 1. One‑paragraph summary

Dr. Elimelech's prior work is a **deterministic** line of research on *abstract strategies*: reusable, generalizable planning experience extracted from past solutions, used to accelerate long‑horizon planning, and (in the 2024 work) guided by *affordances*. This project extends that framework into **uncertainty**, i.e. **Markov Decision Processes (MDPs)**, where actions can fail or have multiple outcomes. Because reaching a goal is no longer guaranteed, the planner must reason about **confidence** and trade it off against **cost**, plan **online** (plan → act → observe → replan), and ideally accumulate a **lifelong** library of strategies. The core research question: *assuming we already have strategies, how do we integrate them into MDP planning so that planning becomes cheaper (fewer replans / lower total cost to the goal)?*

---

## 2. The two project themes (from the poster) and which one was chosen

The supervisor offered two themes, both from the *intersection of planning and learning*:

1. **Learning‑by‑Abstraction — a framework for lifelong learning to plan.** Leveraging learning for planning and vice‑versa: extract abstract strategies/skills from solutions, reuse them to plan faster over a lifetime. (Poster papers [1]–[5].) **← This is the theme chosen.**
2. **Falsification of autonomous systems in rich environments.** Reformulating falsification of black‑box / neural‑component systems (e.g. an RL‑trained autonomous‑car controller) as a "meta‑planning" problem. (Poster paper [6].)

This project sits squarely in Theme 1.

---

## 3. Background — the prior work this project builds on

Three papers are in the knowledge base; they form the deterministic foundation.

**(a) Abstract skills / strategies & abstraction keys — ISRR 2022** (`elimelech2022isrrskills.pdf`, *Efficient task planning using abstract skills and dynamic road map matching*).
- A **skill/strategy** = an **Abstract Road Map (ARM)** — a *sequence of states* in an abstract state space — paired with an **abstraction key** (a projection function `S→Ξ`, its inverse reconstruction `Ξ→S`, and a parameter space).
- An abstraction key lets a road map be **transformed** (e.g. geometric keys: translation, rotation, scaling; or symbolic keys: "attention", "symbol stripping") so a strategy learned in one setting can be **reconstructed** to fit a new one.
- Using a strategy is two problems: **skill matching** (can this ARM be reconstructed to fit my current state/task/domain?) and **action recovery** (fill in the actual actions). Matching is posed as a **constraint‑satisfaction problem**.

**(b) Extracting strategies from one execution — ICRA 2023** (`elimelech2023extractskills.pdf`, *Extracting generalizable skills from a single plan execution using abstraction‑critical state detection (ACSD)*).
- How the library gets built: from a single successful execution trace, **ACSD** detects "abstraction‑critical" states and segments the trace into compact, reusable abstract road maps automatically (rather than hand‑crafting them).

**(c) Affordances & affordance‑directed planning — ICRA 2024** (`elimelech2024skills.pdf`, *Accelerating long‑horizon planning with affordance‑directed dynamic grounding of abstract strategies*).
- **Affordance** = a *predictive* (not retrospective like cost/reward) estimate of how beneficial it is to use a strategy from the current state. Modelled as a **numeric vector** of scores; by convention non‑negative with **0 = best**.
- Example scores: **start affordance** (effort to reach the strategy's first state), **strategy affordance** (effort to refine/traverse the road map), **task affordance** (remaining effort to the goal after the strategy). Thresholding the vector defines the strategy's **applicability region / "basin of attraction."**
- Enables **optimistic, lazy** planning: do high‑level strategy sequencing first (assuming strategies are feasible), then refine ("ground") only the strategies that made it into the skeleton; on a refinement failure, truncate that edge and resume rather than restart.
- Supports the **lifelong plan‑learn loop**: keep solving, keep extracting strategies, keep getting faster.

**Strategies vs. macro‑actions (important framing).** A **macro‑action** is a fixed *sequence of actions*. A **strategy** prescribes a *sequence of states* and lets the planner choose the actions to move between them — more general and more flexible, while still telling you *where you're going*. This distinction is central to the project's evaluation (see §6).

**Everything above is deterministic.** Applying a strategy in the prior work guarantees reaching its final state. That guarantee is exactly what breaks under uncertainty — which is this project's opening.

---

## 4. Vocabulary the meetings pinned down

- **MDP** — a model with **states, actions, transitions**, where transitions are **non‑deterministic** (e.g. "90% chance the door opens"). Nothing more exotic than that is needed here. *State = where you are.*
- **Policy vs. plan.**
  - A **policy** gives the optimal action at *every* state; it solves a whole *family* of problems (e.g. "any maze"). This is the **offline** view, and it is harder. **Not what we want.**
  - A **plan** is a sequence of actions from the *current* state. **Online** planning means: build a plan, execute the first action, observe the outcome, and **replan** if needed. **This is what we want.**
- **Offline vs. online MDP planning.** Offline = compute a full optimal policy for every state *before* execution. Online = generate a plan from the current state only, execute step‑by‑step, and replan on failure. *Start online; if it works well, consider taking it to offline later.*
- **Confidence vs. cost.** Under uncertainty there is always a trade‑off between how *certain* a plan is to succeed and how *expensive* it is. MDP planners balance **exploration and certitude / coverage**.
- **Two senses of "probability of success" (do not conflate these):**
  1. *Probability a plan succeeds in the MDP* — exists even without strategies, straight from the transition probabilities.
  2. *Probability a strategy succeeds* — here "success" means **"I manage to ground/refine the strategy into concrete actions."** Refinement is: pick a strategy, try to refine it; if it can't be refined, **abort, backtrack, and pick another.** Part of the project is seeing how this notion bridges with confidence in MDPs.

---

## 5. Meeting 1 — Realignment of the first proposal (the major course‑correction)

**Context.** The first proposal (`1st_Project_proposal__will_throw_away_1.pdf`, *"Strategy‑Augmented Planning Under Uncertainty: Applying Abstract Strategies to an MDP Key–Door Maze"*) used a key–door maze, **temporally‑extended goals**, value/policy iteration, and assumed the prior‑work algorithms would be modified. The supervisor flagged it as *"not really a proposal"* and reset the framing.

**What was wrong / what changed:**

- **It read as a solution, not a proposal.** A proposal should **identify the gaps and challenges** — *what exists, what is the gap* — not present a finished method. The work is both **qualitative and quantitative**: a new *domain* (uncertainty/MDP) with a framework (strategies) ported from the old (deterministic) setting.
- **Drop temporally‑extended goals.** They are separate from MDPs and not needed. "At the end of the day, I want to reach a certain **state**."
- **Drop the key.** No "having a key" in our world. Doors are simply **locked or unlocked**. (A key is just a *precondition of an action*; we don't need that machinery.)
- **State the generic problem cleanly:** *We are in state x and want to reach state y via several actions, but we are not certain we will get there. That is planning under uncertainty, and we must **quantify the confidence** of successfully reaching y (e.g. 50%).*
- **Don't modify Khen's algorithms.** Originally Louis assumed the affordance/strategy *algorithm family* had to be rewritten for stochasticity. **Correction:** instead, **take an existing MDP algorithm (find one — it need not be state‑of‑the‑art) and apply the strategy ideas to it.** Then show that the strategy framework benefits MDP planning.
- **Include a confidence level in the affordance** (the deterministic affordance had no notion of this).
- **Core question to answer:** *Assuming we have strategies, how can we integrate them into the planning process?*
- **Change the environment.** Move away from a one‑off maze toward a **navigation environment with repeatable patterns** — repeatable structure matters because the point is to **learn and reuse** strategies.
- **Macro‑actions vs. macro‑strategies advantage is even more prominent under uncertainty** — worth foregrounding.
- **Read:** the affordances paper (Khen's 2024 ICRA).

**Supervisor's recommended exploration path (Meeting 1):**
- Google/read *"planning with temporally‑extended goals"* and *"planning in dynamic environments."*
- If going the dynamic‑environments route: plan in **MDPs** (a different planning formulation, with uncertainty in transitions); use **road maps adapted to MDP planning**; this could combine both approaches.
- Mechanism sketch: **grow a tree of plans and choose whichever works** → this yields a road map → then **decompose** it (per the second paper) → thereby showing the strategy framework also benefits MDP planning. (*MDP == decision/planning under uncertainty.*) This is the supervisor's recommendation.

> **Slide seed — "Meeting 1: Realignment" (1–2 slides).**
> *Before:* key–door maze, temporally‑extended goals, "solve it," assumed prior algorithms would be rewritten.
> *Realigned to:* a **proposal that identifies gaps/challenges**; drop keys & temporal goals; doors are just locked/unlocked; reach a target **state** with quantified **confidence**; **don't rewrite Khen's algorithms — take an off‑the‑shelf MDP algorithm and show strategies help it**; move from **policy (offline)** toward **plan (online)**; environment with **repeatable patterns** so strategies can be learned & reused; affordances should carry a **confidence** notion.

---

## 6. Meeting 2 — Q&A with Khen (further refinement)

The refined proposal (`Robotics_Project_proposal_Louis_Golding.pdf`) was submitted; Louis brought questions, and the answers both resolved them and opened up adjacent points.

**Q: Affordance estimates *usefulness* — do we also need to quantify how *reliable* a stored strategy is for the situation at hand?**
- **Yes, and this is a candidate contribution.** Prior work's affordance captured pure usefulness. We can **extend the affordance vector with a reliability score** — same idea (several scores summarized in one vector), now including reliability under uncertainty.
- **Trap to avoid:** even *without* strategies, an MDP planner can already compute the probability of success of a plan. That does **not** map 1:1 onto a *strategy's* probability of success — where "success" means **being able to ground the strategy into actions** (the refine‑or‑backtrack process of §4). The project must show how these two notions of confidence bridge.
- Assume there will always be some **replanning frequency**. Options: **replan after every action**, or **replan only after finishing the current plan**. The trade‑off is *how much to invest in planning*.
- The recipe again: **take an existing MDP solver, take the idea of strategies, integrate them, and show the benefit.**
- **What "success" looks like:** *fewer replans thanks to abstract strategies*, and a **lower total cost of reaching the goal.**
- **Two questions to keep asking** (assume an oracle that knows the correct abstract strategy to pick):
  1. *Given that the system should use strategy X, how does it actually use it?* → We must **extend the MDP algorithm so it can select strategies** (add strategies as something the planner can pick).
  2. *How do we get the system to choose the one we wanted?* → the **selection mechanism**.

**Q: Are limited retries per door allowed, or one try per door?**
- **One try per door.** This keeps the environment **static**; allowing repeated retries on the same door would make it **dynamic**. *(This supersedes the "retry budget as a decision variable" idea in the refined proposal — see §8.)*

**Q: So is it a simple tree search where every edge is a door's probability and we enumerate all paths each time? And can we prune useless branches before computing their probabilities?**
- Roughly yes, and **strategies are exactly what help here** — they bring insight from what has been done before. Normally there are *many* actions to consider, so one **chooses randomly which to expand → a Monte‑Carlo tree (MCTS‑style search).** Strategies guide that choice instead of pure randomness.

**Q: Should the process be lifelong?**
- **Yes.** The strategy **library starts empty** and **grows** as more problems are solved.

**Scope simplifications for now:**
- **Assume the strategies are given.** Focus only on *how to integrate* them. Keep everything at the **same level of abstraction** as the task — **no grounding** work needed for now.
- Just need **some** algorithm that implements an MDP — **not** state‑of‑the‑art.
- **Keep traces of states**, and study **how the MDP can leverage this library (database) of states.**

**Supervisor suggestion (Meeting 2):**
- **Read about macro‑actions.** Strategies are *more general*: a **macro‑action = a sequence of actions**, whereas **what we have is a sequence of states**. Articulate **why strategies (states) beat macro‑actions (actions)**: with states we still **know where we're going**, and that is where the benefit comes from.
- **Final evaluation** should compare three conditions on a set of problems:
  1. **MDP alone** (primitive actions only),
  2. **MDP + macro‑actions** (sequences of actions) — the comparison point,
  3. **MDP + abstract strategies** (sequences of states).
  Expectation: strategies win, and the comparison makes *why* explicit.

> **Slide seed — "Meeting 2: Q&A with Khen" (1–2 slides).**
> Extend the **affordance vector with a reliability score** (new contribution). Distinguish **plan‑success probability** from **strategy‑grounding success** (refine‑or‑backtrack). **One try per door → static environment.** Search is **MCTS‑style**; strategies replace random expansion with informed expansion. **Lifelong** library (starts empty, grows). For now: **strategies are given, no grounding, off‑the‑shelf MDP solver.** Evaluate **MDP vs. macro‑actions vs. abstract strategies**; success = **fewer replans + lower total cost.**

---

## 7. Current consolidated understanding of the problem

- **Environment.** A **static, non‑deterministic navigation environment with repeatable structure** (rooms connected by doors). Each **door has a known probability of opening**; **one attempt per door** (keeps it static). No keys, no temporally‑extended goals. One or more rooms are **goal states** to be reached with confidence **≥ α** (e.g. 0.80).
- **Planner.** **Online** MDP planning over the transition model: build a plan, execute, observe, **replan** (after each action, or after a finished plan — a tunable trade‑off). Search is a **forward / Monte‑Carlo tree** where edges carry door probabilities; useless branches can be pruned by cumulative success probability before expansion.
- **Where strategies enter.** Strategies (assumed given for now) guide which branches to expand — replacing random expansion with experience‑informed expansion. The MDP solver is **extended so it can *select* strategies**, plus a **mechanism to select the right one**.
- **The contributions being staked out.**
  1. **Qualitative:** reframing strategy‑based planning for **uncertainty** — explicit reasoning about success probability, confidence thresholds, and replanning, where applying a strategy is **no longer guaranteed** to reach its end state.
  2. **Quantitative:** **confidence/reliability‑aware affordances** — extending the affordance vector with a **reliability** score so the planner prefers strategies that are both useful *and* reliable under uncertainty.
- **Evaluation.** Same problem set under three planners — **MDP alone**, **MDP + macro‑actions**, **MDP + abstract strategies** — measured by **number of replans** and **total cost to goal**. Possibly test **generalization** to environments with altered door probabilities/layouts, and **online → offline** as a stretch goal.

---

## 8. Locked decisions

All locked decisions are now maintained in the **⚠ CANONICAL** block at the top of this document. See also the **Changelog** for the history of each decision.

---

## 9. Open tensions / things to reconcile

- **Adapting prior algorithms.** The refined proposal lists "adapt Khen's algorithms (matching, affordance computation, reconstruction…) to stochastic outcomes" as a challenge. We do **not have access to their code** — the guidance is to build our own MDP planner informed by the *concepts* from the papers (strategies, affordances, road maps). The proposal's framing of "adapting" specific algorithms should be rewritten as "implementing our own planner that integrates strategy/affordance ideas."
- **Grounding.** For now grounding is **out of scope** (strategies are given, stay at task‑level abstraction). Note this explicitly wherever the proposal currently discusses reconstruction/refinement in detail.

---

## 10. Next actions (low‑risk, high‑value)

1. ~~**Edit the retry budget out of the refined proposal** (`Robotics_Project_proposal_Louis_Golding.pdf`) to match the locked one‑try‑per‑door decision in §8.~~ **Done (09 Jun 2026)** — retries removed *entirely*, not just bounded; see `…(updated).pdf` and the Journal entry. (Outstanding: the "adapt prior‑work algorithms" wording still needs reconciling with canon — §9.)
2. **Read up on macro‑actions** and write the crisp "states vs. actions" argument for why strategies generalize better.
3. **Pick an off‑the‑shelf MDP / online‑planning algorithm** to build on (e.g. an MCTS‑style online planner). It need not be state‑of‑the‑art.
4. **Re‑read the 2024 affordances paper** and draft the **reliability score** as an added affordance dimension.
5. **Specify the environment** concretely: repeatable‑pattern navigation, doors with open‑probabilities, one try per door, confidence threshold α.
6. **Define the evaluation harness**: the three planners and the two metrics (replans, total cost).
7. ~~Later: build the two slide sets from the **Slide seed** boxes in §5 and §6.~~ **Done** — `Meeting1_Recap.pptx` and `Meeting2_Recap.pptx` produced.

---

## 11. Source map

| Source | What it is | Role here |
|---|---|---|
| `KARLPoster_2.pdf` | Lab poster, two themes + paper list | Theme selection (chose Learning‑by‑Abstraction) |
| `elimelech2022isrrskills.pdf` | ISRR 2022 — abstract skills, abstraction keys, matching as CSP | Foundation: strategies & reconstruction |
| `elimelech2023extractskills.pdf` | ICRA 2023 — ACSD strategy extraction | Foundation: how the library is built |
| `elimelech2024skills.pdf` | ICRA 2024 — affordances, optimistic/lazy planning | Foundation: affordances (extend with reliability) |
| `1st_Project_proposal__will_throw_away_1.pdf` | First proposal (key–door maze, temporal goals) | Superseded; what Meeting 1 corrected |
| `Robotics_Project_proposal_Louis_Golding.pdf` | Refined proposal (online MDP, confidence) | Current baseline; refined further by Meeting 2 |
| *Meeting 1 notes* | "Robotics feedback of 1st proposal" | §5 above |
| *Meeting 2 notes* | "Questions for Khen" + answers | §6 above |
| `Meeting1_Recap.pptx` | 2‑slide deck: before/after + refined direction | Grader traceability (Meeting 1) |
| `Meeting2_Recap.pptx` | 2‑slide deck: Q&A + sharpened approach | Grader traceability (Meeting 2) |

*Last updated: 31 May 2026.*
