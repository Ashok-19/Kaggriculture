# P0.2 MACRO INITIATION CEGIS VALIDATED

Date: 2026-08-25
Team: ReCurSiON
Stage: bounded macro/subtree representation + public-state initiation-set CEGIS + safe BKG4 integration

P0.2 validates the first concrete planning-to-Bonsai synthesis link. The P0.1 route435 continuation can be represented as an ordinary bounded BKG4 route macro, given a compact legal public-state initiation gate, and composed with the accepted 345-node parent without protected regressions on the bounded certification panels.

The accepted research-lineage specialist is `route435_pizza_glut_rebalance_v1`. It is **not** made the configured default runtime and is **not** submitted.

## A. Repository state

- Branch at start: `main`.
- Start HEAD: `71b0fa47f58bf8c3c17f771e07d16ddd1490c6c1`.
- The read-only strategic contract `tools/bkg4_search/BONSAI_FIRST_PLACE_MASTER_PLAN.md` was read completely and not modified.
- Five unrelated Aug-25 untracked files were present at start: the paused dual-T4 corpus log plus the bakery/pizza/brunch module, route, distillation script and test. They are not part of P0.2 and were left untouched.
- Final commit is intentionally obtained from git history after this report is committed.

## B. Parent BKG4

The actual accepted parent remained `w33_s172_expansion_v1`:

- 345 nodes;
- 23 blackboard slots;
- 0 NodeState slots;
- max depth 11;
- static max visits 262;
- controller semantic hash `8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034`;
- combined identity `6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925`.

The existing accepted specialist stack remains protected: animal feed deferral, day-2 capital, hire-4 capital cycle, fertilizer/wool expansion and W33/S172 expansion.

## C. P0.1 capability reproduced

The exact route435 source was recovered from `scratch/public_targets/top10_parent_routes/routes_top10_parents.bkv6` rather than reconstructed from prose.

- donor route: 435;
- macro start: step 156;
- duration: 48 turns;
- active window: steps 156..203 inclusive;
- return: normal parent routing from step 204;
- source-fragment SHA256: `77ba16e50fc8634b3ea010e45a7cf308723df12cdbad215665a35fbef97f83ba`.

P0.1's key valid effects were preserved, including hard case9 `-5443 -> +6136`, its paired state `-5178 -> +5630`, and the PET->PIZZA state `-3204 -> +9033`.

The capability remains the coordinated production/worker/market rebalance found in P0.1: reduced strawberry commitment, altered sheep/wheat timing, substantially more productive worker use, and fewer idle turns. Component ablation from P0.1 already showed that market-only and worker-only transplants were less transferable than the full program.

## D. Macro representation

`tools/bkg4_search/bounded_macro.py` adds the deliberately small `BoundedRouteMacro` abstraction (`bounded_route_macro_v1`). It contains:

- macro name and deterministic semantic ID;
- route ID and base route provenance;
- start step and bounded duration;
- one or more small BKG4-native gate clauses;
- identity-free source provenance;
- one module-owned BOOL latch;
- a generated ordinary BKG4 subtree;
- a generated bounded route document.

Execution semantics are precise:

1. initiation is evaluated exactly once at `start_step`;
2. a passing public-state gate sets the macro-owned BOOL latch;
3. while latched and `start_step <= STEP <= end_step`, the subtree writes the existing strategic route-override BB slot;
4. the parent resets that shared route slot every tick before specialists run;
5. the bounded macro is appended after existing specialists, so it owns route authority only during its active window;
6. after the end step it writes nothing and the parent/existing specialists resume automatically;
7. it cannot reactivate because initiation is only tested at the single start step.

The latch remains true after termination, but is inert because the bounded STEP condition is closed. This avoids a second reset/state machine.

No BKG4 VM ABI, feature registry, NodeState ABI or CUDA-only semantic was added. The composed controller uses one additional BOOL BB slot only.

Runtime/provenance safety rejects identity-like keys and string values containing opponent/team/submission/replay/episode/seed/tape terms. Gate conditions use the existing portable BKG4 feature grammar, which has no identity features.

## E. Initiation-state corpus

The initial exact initiation dataset used five existing panels at step156:

| Panel | Cases | Role in CEGIS |
|---|---:|---|
| screen61 | 61 | P0.1 discovery/hard screen |
| historical336 | 336 | broad protected counterexamples |
| current140 | 140 | shifted-current counterexamples/transfer |
| fresh24 | 24 | fresh support/regression reference |
| sealed24 | 24 | prior sealed support/regression reference |

Total: 585 exact macro-labeled cases and 585 exact legal-state captures. Full BKG4 registry + BB + compatibility state was captured for ambiguity checks; hidden-truth output was not used for synthesis.

Ungated route435 was extremely unsafe on these panels, e.g. screen61 had 12 positive / 2 neutral / 47 harmful and 15 W->L; historical336 had 25 positive / 311 harmful and 220 W->L.

Because the final gate was inspected against those five panels, they are not claimed as final sealed certification.

After the gate was frozen, three pre-existing, `(tape, seat, seed)`-disjoint panels were opened:

- Aug-24 top-10 validation40: 40 unseen records, zero final-gate activations;
- Aug-24 top-10 sealed40: 40 unseen records, zero final-gate activations;
- fresh post-949 panel: 80 unseen records, two final-gate activations on one new episode, both seats, both L->W.

An earlier 112-case candidate holdout was explicitly rejected after overlap analysis showed all rows were already represented in the synthesis evidence.

Across all eight panels there are 700 unique `(tape, seat, seed)` records.

## F. Gate grammar

The synthesis language stayed intentionally small and BKG4-native:

- conjunctions first;
- existing legal BKG4 `ConditionGene` predicates;
- comparators `EQ/NE/LE/GE` as already supported;
- no temporal predicates added by contrastive refinement;
- at most four conditions for this POC;
- the representation can hold a small DNF, but route435 required only one conjunction.

Candidate generation reused the existing contrastive CEGIS machinery in `tools/bkg4_search/edit_search/goals.py`. `macro_initiation_cegis.py` adds effect-aware ranking, exact mixed-state rejection and the required lexicographic safety objective.

No broad derived-feature system was added. The only fourth axis is the existing raw shared-market inventory feature, interpreted causally as strawberry glut.

## G. CEGIS iterations

### Iteration 0 — second-shop PIZZA only

Gate:

`SHOP_AT[1] == PIZZA_SHOP`

This was decisively unsafe:

- screen61: 10 active, 4 positive, 6 harmful, 3 L->W;
- historical336: 54 active, 4 positive, 50 harmful, 32 W->L;
- current140: 20 active, 7 positive, 13 harmful, 5 W->L.

Counterexample lesson: shop demand alone does not determine whether the production rebalance is appropriate.

### Iteration 1 — require visible opponent expansion

Gate:

`SHOP_AT[1] == PIZZA_SHOP`
`AND OPP_UNITS >= 7`
`AND OPP_LAND >= 2`

This removed historical/current expansion-mismatch counterexamples:

- historical336: 0 active;
- current140: 6 active, 6 positive, 0 harmful;

but still failed the hard screen:

- screen61: 8 active, 4 positive, 4 harmful;
- sealed24: 4 active, 2 positive, 2 harmful.

Counterexample lesson: even under the same labor+land regime, route435 is wrong when the strawberry market has not reached the oversupply/glut state.

### Iteration 2 — add strawberry glut

Final gate adds:

`MARKET_INVENTORY[STRAWBERRY] >= 9990`

The harmful same-expansion Ryo states were at strawberry inventory 9972 / price 164, while supported states were at inventory 9993 / price 142. A threshold sweep showed every threshold 9973..9993 produced the same observed activation/effect classification on the core panels. `9990` was selected as a round interior threshold rather than the exact observed 9993 value.

Core result:

- screen61: 4 active, 4 positive, 3 L->W, 0 harmful, worst +7353;
- historical336: 0 active;
- current140: 6 active, 6 positive, 1 L->W, 0 harmful, worst +4977;
- fresh24: 2 active, 2 positive, 1 L->W, 0 harmful;
- sealed24: 2 active, 2 positive, 2 L->W, 0 harmful;
- selected mixed exact legal-state cells: 0.

No further refinement was performed because the compact conjunction met the safety objective and later post-freeze transfer succeeded.

## H. Final gate

Human-readable logic:

**At step 156, activate route435 for 48 turns iff:**

`second visible shop is PIZZA_SHOP`
`AND opponent public unit count >= 7`
`AND opponent public land count >= 2`
`AND shared STRAWBERRY market inventory >= 9990`.

Why it is causal rather than an arbitrary fingerprint:

- PIZZA as the second visible shop identifies relevant downstream demand composition;
- opponent units + land indicate that the public competitive economy is already in an expanded labor/land regime, excluding historical five-worker symmetric states where the macro is destructive;
- strawberry inventory near the 10,000 baseline identifies oversupply/glut, exactly the regime where reducing strawberry commitment is economically coherent;
- the strawberry threshold has a wide behaviorally identical plateau rather than matching one exact observed value;
- `OWN_UNITS == 5` was deliberately not added because all core cases already had five own units at step156, making it redundant and fingerprint-like.

The gate contains no opponent identity, team, submission, replay, episode, seed, tape, hidden inventory or future shop information.

## I. Exact CUDA evidence

P0.2 did not repeat P0.1's 61k-game search. Exact CUDA was used only for route435 labeling, state capture and composed certification.

Game-scale macro evaluations:

- five core panels: 585 cases x baseline/macro = 1,170 exact CUDA games;
- post-freeze validation40: 80 games;
- post-freeze sealed40: 80 games;
- post-freeze fresh94980: 160 games;
- compiled composed-controller certification: 745 games;
- total exact game evaluations in this stage: 2,235.

The final composed 745-case campaign used about 3.586 seconds of CUDA kernel time across eight deliberately small independent batches (about 208 games/s aggregate for these under-filled certification batches). Every campaign reported `bad=0`.

Casewise composed semantics over the 745 certification records:

- gate activations: 16 raw panel records;
- inactive records exactly equal parent: 729/729;
- active records exactly equal standalone route435 counterfactual: 16/16;
- piecewise mismatches: 0;
- positive activations: 16/16;
- harmful activations: 0;
- L->W: 9 raw-panel records;
- W->L: 0;
- worst activated delta: +4,977;
- aggregate activated delta: +150,786.

The raw counts include intentional panel overlap. Deduplicated support is reported below.

## J. Independent transfer

Deduplicating all eight panels by `(tape basename, seat, seed)` gives:

- 700 unique records;
- 10 unique final-gate activations;
- 10/10 positive;
- 0 harmful;
- 5 L->W;
- 0 W->L;
- five unique supporting tapes/episodes;
- every supporting tape appears as a both-seat positive pair;
- worst unique delta +4,977;
- aggregate unique delta +89,219.

Supporting tapes are:

- `Ryo_Hasegawa_s1_ep94460852.json` — both seats positive, one L->W;
- `Ryo_Hasegawa_s1_ep94524402.json` — both seats L->W;
- `Subramanya_N_s1_ep94654660.json` — both seats positive;
- `Crop_Dusta_s1_ep94656425.json` — both seats positive;
- **post-freeze unseen** `Ryo_Hasegawa_s1_ep94824465.json` — both seats L->W.

The fresh949 sealed-positive pair is especially important because it was not used to choose or refine the gate:

- seat0: `-1824 -> +6134`, delta +7,958;
- seat1: `-1383 -> +6043`, delta +7,426.

Its visible initiation state is BAKERY then PIZZA, opponent 8 units, opponent 2 land, strawberry inventory 9993. This independently shows that the final gate is not a PET/PIZZA or PIZZA/PIZZA tape selector.

The two Aug-24 top-10 post-freeze panels are inert under the gate, so they contribute exclusion/regression evidence but not positive transfer.

## K. Existing specialist regression

Protection results over the eight-panel composed certification:

- historical336: 336/336 exactly parent, zero activations;
- current140: six activations, all positive, zero W->L;
- screen61 hard panel: four activations, all positive, three L->W;
- fresh/sealed references: no harmful activation;
- post-freeze validation/sealed top-10: no activations and exact parent behavior;
- fresh94980: two activations, both L->W;
- 490 unique parent-winning records protected, 0 W->L;
- eight unique ReCurSiON self-history records, zero activations and zero harm.

Because the parent route-override slot is reset every tick, existing specialists execute before the new macro, and route435 writes last only during 156..203. Any incumbent specialist that is latched during the macro window can resume from step204. The exact 745-row piecewise equivalence test verifies this composition behavior end to end.

## L. Composition result

Accepted research challenger: `route435_pizza_glut_rebalance_v1`.

Compiled ordinary BKG4 assets:

- controller nodes: 363;
- BB slots: 24;
- state slots: 0;
- max depth: 11;
- static max visits: 279;
- max live intents: 0;
- controller semantic hash: `f47ee7034e553e9a835486b283f4f819d857afdc95143a29d84cb07753b7c286`;
- combined identity: `e2a2a0e91035c05c3183e691efee19441b6cee0d86506cbeb0e68a55768e53ce`;
- new strategic route: 216;
- route-pack hash: `ca28049cfbfcb57bbcfb8c74700881737ee3f6133e1ccd3d751bc85d1c415cee`.

One BB BOOL is the only runtime-state cost. Node count rises by 18; static maximum visits rises from 262 to 279. No live intents or NodeState are introduced.

A direct BB trace on one inactive control and the unseen fresh949 active case verified:

- inactive: macro latch 0 / route override 0 throughout;
- active step155: latch0, override0;
- step156: latch1, override216;
- steps157..203: override216;
- step204: override0;
- step215: override0.

The configured `default_runtime` in `CHAMPION_LEDGER.json` remains unchanged; this is accepted lineage, not a submission/default promotion.

## M. CPU/CUDA parity

No official Python Kaggriculture engine was used.

The existing BKG4 v4 C++ and CUDA differential runners were fed 217 consecutive real captured values for the exact macro-decision surface (`STEP`, `SHOP_AT[1]`, `OPP_UNITS`, `OPP_LAND`, strawberry market inventory) for one inactive and one active case.

- records: 434;
- CPU C++ vs CUDA mismatches: 0;
- inactive route decision at step156: unchanged;
- active route decision: route216 at step156 through step203;
- both return to parent route authority at step204.

A focused unittest also reproduces this bounded CPU/CUDA differential when the local runners are available.

## N. Rejected gates

Major rejected selectors:

1. **PIZZA second-shop only** — many historical/current harmful activations and protected W->L.
2. **PIZZA + opponent expansion** — historical/current became safe, but four screen and two sealed harmful activations remained.
3. **Automated 1-3 predicate conjunctions on tape-disjoint historical split** — no gate both activated and transferred safely to the opposite split.
4. **Automated 1-3 predicate conjunctions on tape-disjoint current split** — likewise no safe active transfer.
5. **OWN_UNITS == 5** — rejected as redundant because all core initiation states already shared it.
6. **Thresholds below the safe strawberry boundary** — admit harmful same-expansion states; thresholds 9973..9993 were equivalent on observed core data, so a round 9990 interior value was selected.
7. **Exact state/tape selectors** — never permitted. Mixed exact-state handling is fail-closed in the CEGIS helper.

These rejections matter: the final four-condition gate was not chosen merely for aggregate money; each added axis removes an observed causal counterexample class.

## O. Strategic interpretation

**YES.** Initiation-set CEGIS converted a counterfactual-planning discovery into a reusable deterministic Bonsai specialist.

The important proof is not only that route435 can be represented. It is that the compiled BKG4 candidate uses a compact legal state gate, preserves P0.1 wins, transfers to independent current and post-freeze fresh states, has no harmful activation on the bounded certification universe, exactly falls back to the parent when inactive, and has matching CPU/CUDA macro decisions.

The evidence is still appropriately narrow: five unique supporting episodes/tapes, not a claim of universal safety. That is sufficient for the first synthesizer proof and should guide the next regret-search/synthesis iteration.

## P. Next step

Proceed with the **next regret-search -> bounded-macro -> initiation-CEGIS iteration**, not EditFormer/corpus work.

The smallest useful extension is to make this now-proven path accept the next planner-discovered route fragment as data: instantiate `BoundedRouteMacro`, capture its initiation effects, run the same compact CEGIS objective, and produce another independently gated specialist. Recovery/abort synthesis and automatic multi-specialist composition remain later stages unless a future macro demonstrates a concrete need.

## Q. Tests

Focused commands used:

- `python3 -m unittest tools.bkg4_search.test_bounded_macro_cegis -v` — 11/11 pass, including CPU/CUDA bounded macro differential;
- `python3 -m unittest tools.bkg4_search.test_regret_macro_search tools.bkg4_search.edit_search.test_goals -v` — 13/13 pass;
- `make -C cuda-engine -B bkg4-fragment-eval` — pass;
- deterministic CEGIS reproducer — pass, zero final mixed exact-state cells;
- exact composed CUDA campaign — 745/745 valid, `bad=0`;
- bounded C++/CUDA VM parity — 434/434 records exact.

The official Python game engine was not run.

## R. Commit

The accepted implementation is committed locally after final review. The exact hash is reported by the final task response / `git log -1`.

## S. Git status

P0.2 permanent code, compiled challenger assets, module/macro/route artifacts, report and ledgers are intended to be committed. P0.2 scratch outputs remain under ignored `scratch/`. The five unrelated pre-existing Aug-25 untracked files are intentionally preserved and excluded.

## T. Final recommendation

**Proceed to the next regret-search -> synthesis iteration.** P0.2 establishes the first real autonomous-synthesis primitive the master plan requires: a deeper-planning continuation can be converted into a first-class bounded BKG4 macro, counterexample-gated with legal public state, independently transfer-tested, and safely composed with the accepted Bonsai lineage. The next work should repeat this loop on a new regret-discovered capability rather than widen the infrastructure prematurely.
