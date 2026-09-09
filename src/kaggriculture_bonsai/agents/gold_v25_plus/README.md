# Gold V25 Reinforcement

## Current Validation

The continuation from commit `26284e7` is recorded in rounds 5-8. Its frozen
candidate is `BKG4_GOLD_V25_PLUS_RECOVERY_V8`; acceptance is determined only by
`round8/final_target.json`, `round8/freeze.json`, and `round8/release.json`.
Until those gates pass it is experimental. The earlier audit below is historical,
not the result of the current composition. `active_candidate.json` is written
only after the independent exit gates pass; otherwise the convenience entry
remains the earlier experimental v2.

BUILD diagnostics identified three propagation mechanisms: short wheat pickups
leave later FEED actions unfunded; a weed-blocked structure is dropped from the
parent repair queue before an animal arrives; and a skipped planting can leave
surplus seeds that the schedule buys again. Recovery does not rewrite the tapes.
It adds projected pickup coverage during steps 144-646, missing-structure recovery,
and a remembered construction job performed by the farmer only when a complete
return trip fits inside up to five scheduled PASS actions. Seed purchase reduction
uses the full remaining production requirement and the maximum requirement of
both late routes. A short-horizon seed reduction failed and was discarded.

V7 passed 512 BUILD and 128 fresh DEV games but its official canary was 63 W /
0 T / 1 L. Its official case was not used to tune V8. V8 was derived from fresh
BUILD 1080000-1080127, and passed that 256-game panel plus 512 previous BUILD
games. Its untouched DEV 1090000-1090063 passed 128/128, both seats. Its official
canary uses 1100000-1100031; FINAL_TARGET remains the originally reserved
930000-930127 until the freeze gate opens it.

V8 has a 16-node specialist controller plus the 23-node parent: 39 total nodes,
four I32 blackboard slots, and zero VM node-state slots. Additional overlay state
is 488 bytes per player (last step, 100 target entries, queue length, five actions).
A 12,984-byte per-player cache is derived from immutable own schedules for seed
requirements. No generic runtime or route-pack format changes were made. All
new routing depends on own observations/state and own schedules, never opponent
identity. Parent/champion/baseline assets remain separate and unchanged.

Reproduction for the frozen V8 candidate:

```sh
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.build
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.freeze7 --round 8
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.package --evidence workspace/gold_v25_plus/round8
.venv/bin/python -m unittest kaggriculture_bonsai.agents.gold_v25_plus.test_repair kaggriculture_bonsai.agents.gold_v25_plus.test_plus
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.verify canary --evidence workspace/gold_v25_plus/round8
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.verify final --evidence workspace/gold_v25_plus/round8
```

Do not overwrite a completed official/final panel. The final command is a
one-time gate, not a search loop. Official canaries compare every action and
parent state; focused recovery tests additionally compare the full new queue
and target state on BUILD histories and repeated/backward/per-player resets.

## Historical V2 Stop

No candidate is accepted or promoted. The best frozen experimental composition,
`BKG4_GOLD_V25_PLUS_MARKET_V2`, scores 350 W / 0 T / 2 L over the combined
352-case BUILD/DEV audit, versus PARENT's 15 W / 322 T / 15 L. This combined
audit is not an independent final evaluation. Protected regressions are zero.
FINAL_TARGET remains unopened. The v1 official canary was 62 W / 0 T / 2 L;
v2 did not receive a canary because it already failed the native screen.
The Python convenience entry and standalone package are experimental v2, not
a ready submission. `workspace/gold_v25_plus/audit.json` records the stop gate.

This isolated experiment treats BKG4_GOLD_V25 as immutable PARENT. It never
modifies the four tapes, parent routing, overlays, packaging, or baseline assets.
Public v25 is the requested benchmark; BUILD consists of fresh parent self-play.
No opponent name, seed, fingerprint, source hash, or hidden field enters routing.

## Questions And Panels

Round 1 used 16 BUILD seeds (910000-910015), both seats. PARENT scored 1 W,
30 T, 1 L. Daily telemetry and a targeted per-step diagnostic locate the money
asymmetry at step 403; this is earlier than the proposed endgame repair.
Late cargo and market telemetry motivated a bounded market-order experiment.

48 alternatives received 8-case CUDA screening; six received full 32-case BUILD.
Candidate 16 selected steps 708-718 and projected-stock liquidation with stable
descending current-value order. It scored 32/32 BUILD and 64/64 INTERNAL_DEV
(920000-920031), with no protected regressions. Its independent official canary
(940000-940031, both seats) scored 62 W / 0 T / 2 L. All actions and rewards
matched native CUDA. This candidate was rejected for the 100% objective, not
silently promoted. FINAL_TARGET was not opened for it.

Round 2 asks a different BUILD-only question: prioritize actual available sales
earlier, but only when the original market list consists entirely of SELL/no-op
orders; never move a purchase or hire. Combine that with the fixed endgame
specialist. It uses fresh BUILD 950000-950031, fresh INTERNAL_DEV 960000-960031,
and fresh official canary 980000-980031, always both seats. Its 12 start-window
alternatives are ranked on BUILD only. The v1 canary provides rejection only;
no failing case observations or continuation experiments are used to tune v2.

Round 2 selected start 576 on BUILD (64 W / 0 T / 0 L), then scored 63 W /
0 T / 1 L on INTERNAL_DEV, with zero protected regressions. It was rejected
without opening FINAL_TARGET or tuning from that held-out case.

Round 3 uses BUILD 990000-990063 (128 games), INTERNAL_DEV 1000000-1000063
(128 games), and official canary 1010000-1010031 (64 games), both seats.
The frozen v2 diagnostic scored 127 W / 0 T / 1 L on this fresh BUILD.
The weakest BUILD case first differs at step 187: a weed causes DIG instead
of PLANT. Its economic deficit grows before the late market specialist.
Round 3 tests earlier conversion of available non-input products on sell-only
or idle market turns. Wheat/fertilizer orders and mixed purchase lists remain
unchanged. Eight windows receive a screen on four weakest BUILD seeds, and
four finalists receive the full BUILD panel. No downstream cases drive ranking.

Round 3's immediate sales regress heavily (best 61 W / 0 T / 67 L); all were
rejected. Round 4 reuses only this BUILD panel and anticipates the agent's own
scheduled sales by two or three steps within the current demand cycle. Its
best candidates score 127 W / 0 T / 1 L, with no improvement over v2 on the
failure. Its reserved DEV panel was never opened. Search stopped instead of
multiplying equivalent market candidates. A read-only weed-repair diagnostic
found that simply delaying the missed planting would conflict with subsequent
movement and planting/watering actions; no unit-action specialist was accepted.

FINAL_TARGET is reserved at 930000-930127, both seats (256 games), disjoint from
every BUILD/DEV/canary seed. `verify final` refuses to reopen it and records the
composition/source freeze before executing it. If the frozen final evaluation
fails, its observations must not become candidate construction inputs.

## Runtime

The original 23-node controller always executes and advances parent state first.
An additional synchronous BKG4 controller selects bounded market overrides.
Outside the gate, the complete parent action is returned unchanged. Both
specialists leave every unit action unchanged. Parent routing decisions and
pending repair/suppression state remain live even while market overrides act.

The late specialist replaces the market list with positive projected shed
products, in stable descending current-price times quantity order. Projection
uses the exact parent DROP/PICKUP/PLACE and capacity semantics, preserving
inventory insertion order. The early specialist only reorders existing orders;
quantities and zero/no-op slots are preserved, with projected sellable value as
the key. Mixed buy/hire/sell lists fall back exactly.

The CUDA evaluator reuses the existing native simulator, extended BKV6 loader,
parent host/device runtime, and BKG4 VM. It is a small isolated cohort runner,
not a new discovery framework. Screening uses explicit equivalent bounded
predicates; frozen evaluation also executes the serialized specialist BKG4 gate
and checks predicate equality at every acting step. CPU checks share the same
adapter/overlay and use the existing C++ simulator. Full action-stream hashes
cover both players. Official canaries additionally compare exact action objects
and complete persistent parent state between Python and packaged C++ each step.

## Reproduction

Run from the repository root; GPU commands must not use RTK:

```sh
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.benchmark
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.diagnose
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.build
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.search
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.screen
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.round2
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.round3
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.inspect_build
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.search3
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.search4
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.package --evidence workspace/gold_v25_plus/round2
.venv/bin/python -m unittest kaggriculture_bonsai.agents.gold_v25_plus.test_plus
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.verify canary --evidence workspace/gold_v25_plus/round2
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.verify final --evidence workspace/gold_v25_plus/round2
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25_plus.audit
```

These commands are an experiment sequence, not permission to overwrite frozen
evidence or reuse a final panel for another candidate. Use a separate evidence
directory for a new experiment. Tracked results are compact JSON; full traces
are regenerated. Binaries and the standalone archive live under `/tmp`, not in
champion assets or committed campaign output. `cleanroom_check.py` is copied
beside packaged `main.py` and run from outside the repository with `python -E`.
The package contains a native CPU BKG4 parent-plus-specialist runtime and only
stdlib adapters; no repository Python import or CUDA runtime is required.
The `verify final` command above is intentionally blocked by the failed DEV
gate. It is documented for reproducibility, not an instruction to bypass it.

The final bounded checks passed: three parent test groups, 40 BKG4 semantic
tests, 19 evaluator tests including CUDA parity, two specialist test groups,
and 16 CPU/CUDA full-game probes across the tested market representations.
The v2 controller adds nine nodes and one I32 blackboard slot to the original
23 nodes and three slots: 32 nodes, four slots, zero node-state slots in total.
Parent overlay state is retained unchanged; there are no new state queues,
route-pack modifications, or generic VM/runtime changes.

## Attribution

Parent schedules and chassis remain attributed to yhay81, thomastschinkel,
tetsutani and the notebook's other credited contributors, as documented in
`../gold_v25/PROVENANCE.md`. The inherited Apache-2.0 notice is retained in the
standalone package. ReCurSiON's additions are the bounded specialist composition,
evaluation and packaging work, not authorship of the donor schedules.

Only a passing `final_target.json`, matching official canary, regression/parity
checks and unchanged-parent verification can establish this phase's exit gate.
BUILD or DEV win rates alone do not establish success or general dominance.
