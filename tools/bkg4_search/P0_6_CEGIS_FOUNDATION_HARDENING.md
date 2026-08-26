# P0.6 CEGIS Foundation Hardening

Date: 2026-08-26

Starting HEAD: `9518792d52a05fc735a6eea4ccc111b3ff4742ff`

## Verdict

**CEGIS FOUNDATION HARDENING VALIDATED.** The autonomous synthesizer now uses a real bounded counterexample-feedback loop, treats the original target rescue as required synthesis evidence, requires gated independent transfer, keeps the sealed holdout frozen, preserves exact legal-state ambiguity fail-closed semantics, and retains validated P0.2/P0.3/P0.4/P0.5 behavior.

## Prior behavior

Before this pass, `run_cegis()` was mainly ranked gate enumeration over a fixed evidence set:

- `discovery + counterexamples` was frozen once;
- `enumerate_safe_conjunctions()` produced one static ranked list;
- validation was evaluated against each pre-ranked gate but validation harms were not appended to synthesis evidence;
- no resynthesis occurred from newly discovered validation harms;
- when initial evidence contained no harmful rows, a shared one-predicate gate could be returned even if validation was harmful;
- original target-family discovery states did not participate in gate synthesis;
- the target-capability contract was enforced only later after composition;
- ungated transfer was checked, but final gated transfer support was not independently required;
- trace artifacts did not identify dataset roles, newly added harms or terminal reasons strongly enough for audit.

Broad archive/current-top10/certification falsification happened after gate freeze/composition. It could reject a candidate but could not feed a harmful activation back into initiation-gate synthesis.

## Correctness defects fixed

1. **Validation harm did not drive resynthesis.** Exact validation harms now become counterexamples and the gate is synthesized again from enlarged evidence.
2. **No-harm fast path ignored validation safety.** It now passes through the same validation/counterexample-feedback logic.
3. **Original target capability was absent from synthesis.** Exact target initiation state is captured and target-positive/L->W rows are marked required before CEGIS.
4. **Target-aware ranking alone was insufficient.** Safe convergence fails with `TARGET_CAPABILITY_LOST` if the compact grammar preserves zero required target positives, or zero required target L->W when discovery contained L->W.
5. **Final gate could discard independent transfer.** `assess_transfer_gate_support()` now requires configured gated positive and independent transfer support before sealed holdout.
6. **Sealed role was implicit.** `assess_sealed_holdout()` is an explicit frozen-gate contract; sealed harm rejects without becoming CEGIS training data.

## Strengthened bounded CEGIS algorithm

For a fixed candidate macro:

1. Obtain exact CUDA macro-effect evidence on target, transfer, validation and sealed panels according to their established roles.
2. Capture exact legal initiation state for target as well as transfer/validation/sealed cases.
3. Build synthesis evidence from target + transfer. Mark target positives and target L->W as required classes.
4. Detect exact mixed legal-state cells using the full legal registry + BB + compat signature.
5. Synthesize the best compact conjunction from current evidence using the existing contrastive generator.
6. Evaluate the gate over the exact CUDA validation pool.
7. Add every newly gate-active harmful or W->L validation row to the synthesis counterexample set.
8. Resynthesize from the enlarged set.
9. Repeat until safe convergence or a bounded terminal reason: `MIXED_LEGAL_STATE`, `NO_LEGAL_SEPARATOR`, `TARGET_CAPABILITY_LOST`, `CEGIS_BUDGET_EXHAUSTED`, or safe convergence.
10. Require gated independent transfer support.
11. Check sealed holdout under the frozen gate; never feed it back silently.
12. Run threshold-neighbor robustness.
13. Continue through existing bounded macro/composer, independent target-capability audit, broad certification and parity.

For a fixed macro/case, exact CUDA forced-macro outcome is invariant to symbolic gate resynthesis; the gate changes only activation. Therefore validation macro effects can be computed once by CUDA and harmful gate-active rows can be fed back without replaying identical games every CEGIS iteration.

## Objective ordering

`GateScore` remains safety-first and aggregate-money-last:

1. zero W->L;
2. zero harmful activation;
3. required target L->W;
4. required target positive support;
5. overall L->W;
6. overall positive support;
7. independent/support-group breadth;
8. worst active delta;
9. gate simplicity;
10. aggregate delta.

A separate final-gate transfer report enforces configured minimum positive and independent transfer support. Target support and transfer support cannot replace one another.

## Dataset-role discipline

`MacroEffectCase` now carries `dataset_role`, `required_target_positive` and `required_target_l2w`. Each CEGIS iteration records synthesis roles and counterexample-pool roles.

- `target_capability`: synthesis evidence with required positive/L->W semantics;
- `transfer`: synthesis evidence and gated-transfer acceptance source;
- `validation`: active CEGIS counterexample source;
- `sealed_holdout`: frozen-gate rejection source only;
- certification/archive/current-top10: post-composition broad safety.

No sealed evidence is automatically reclassified. Tuning on a failed sealed case requires a fresh independent holdout.

## Exact legal-state ambiguity and identity

Exact ambiguity remains registry + Bonsai blackboard + compat state. Same exact current legal state with both positive and harmful macro outcome fails closed. Hidden future truth is not used to rescue ambiguity.

Runtime identity-like features remain rejected. `OWN_SEED` remains legal own private information and retains its dedicated exception from the offline seed-identity substring guard.

## Gate grammar

No grammar expansion was justified. Autonomous CEGIS still synthesizes one conjunction with at most four predicates. `BoundedRouteMacro` already supports multiple clauses/Selector at runtime, so a Level-2 DNF can be added later if a real separable two-region failure proves the need.

No decision tree, synthesized termination, abort or recovery mechanism was added. Current evidence does not show that the fixed bounded return-to-parent representation is the blocker; candidate 50 was an initiation/acceptance-contract failure, not an abort/recovery failure.

## Threshold robustness

Numeric predicate sweeps retain the selected threshold and bounded immediate neighbors. Every row now reports activation/positive support, harmful/W->L, target-positive/target-L->W support, support-group breadth, worst delta, and nearest tested unsafe lower/upper threshold.

A sharp boundary is reported rather than automatically rejected. Existing fail-closed rejection remains for an unsafe selected threshold and the configured brittle-neighbor condition; a single unsafe side alone is not automatically invalid.

## Candidate-50 regression

The persisted P0.7 Stage-A candidate-50 exact CUDA audit is now a real regression fixture:

- route212 @ step24 x 24;
- discovery deltas `6271, 6271, 104, 4155, 4155, 104`;
- 6/6 discovery positives;
- 4 discovery L->W;
- old final gate `OPP_MONEY <= 22`;
- all six exact target initiation states have `OPP_MONEY = 170`;
- old gate target activations 0/6;
- preserved target positives 0;
- preserved target L->W 0;
- result `TARGET_CAPABILITY_LOST`.

No candidate-50-specific production logic was added and route219 was not patched.

## Historical regression evidence

The bounded historical/composer suite passed for route435, route410 and route437, including available CPU/CUDA decision parity.

Cheap exact evaluation of the known historical gates against all stored exact effect/state panels produced:

- route435: 14 active / 14 positive / 7 L->W / 0 harmful / 0 W->L / 4 support groups / 0 selected mixed exact cells;
- route410: 12 active / 12 positive / 6 L->W / 0 harmful / 0 W->L / 6 support groups / 0 selected mixed exact cells.

A direct full route435 re-synthesis through the new generic loop was attempted with a 30-second CPU cap and stopped on timeout. It was not rerun because this stage forbids heavy CPU work. This is a performance limitation of full historical symbolic resynthesis, not a semantic regression: exact stored gates, fragments, composition semantics and CPU/CUDA decision behavior all pass bounded regression. Route437 compatibility is covered by exact gate/fragment/composition and 171-step CPU/CUDA collision parity; its historical script does not expose the same reusable CEGIS panel loader.

## Tests

Final bounded regression command:

```text
python3 -m unittest \
  tools.bkg4_search.test_bonsai_synthesizer \
  tools.bkg4_search.test_bounded_macro_cegis \
  tools.bkg4_search.test_synthesizer_repeatability \
  tools.bkg4_search.test_p0_4_third_specialist \
  tools.bkg4_search.test_bonsai_composer \
  tools.bkg4_search.test_regret_macro_search -v
```

Result: **99/99 PASS**.

New focused regressions cover active validation feedback, no-harm validation safety, exact mixed-state failure, explicit dataset roles, required target L->W, target-state tagging, trace roles/counts, deterministic iteration-budget exhaustion, threshold unsafe boundaries, real candidate-50 loss, independent transfer retention, frozen sealed holdout behavior, gated transfer support, and compact-grammar target failure.

No official Python engine, P0.7 Stage B, large autonomous campaign, million-game workload, EditFormer/corpus work, PSRO, push or submission was run.

## CUDA/parity evidence

Historical focused tests exercised available CPU/CUDA differential runners and passed exact decision parity for route435 macro boundaries, route410+route435 boundaries, route437 collision/preemption, and automatic composer collision semantics.

Candidate-50 regression reads persisted exact CUDA `.b4fr` and `.b4is` evidence. No new game-scale CUDA campaign was needed for the symbolic CEGIS contract changes.

## Determinism

Same CEGIS inputs and budget produce the same gate, trace and stop reason. Budget exhaustion is explicit/deterministic. Existing whole-synthesizer run identity/result determinism tests continue to pass.

## Repository organization and cleanup

`SYNTHESIZER_ARCHITECTURE.md` now identifies the authoritative dependency path and separates route-specific historical reproduction scripts from autonomous synthesis.

No tracked historical synthesis script was deleted, no broad directory move was performed, and no evidence/replay/search directory was deleted.

High-confidence cleanup only:

- removed `tools/bkg4_search/bonsai_editformer/.venv` (4.3 GB), a self-ignored reproducible virtual environment with no tracked references;
- removed ignored Python `__pycache__` directories under `tools/bkg4_search` (later tests can regenerate them).

Immediately after cleanup, `tools/bkg4_search` dropped from approximately 4.3 GB to 2.7 MB. `scratch` (~28 GB in the bounded audit) and `data/edit_corpus` (~293 MB) were deliberately retained as protected evidence/data.

Five pre-existing unrelated untracked files remain deliberately preserved as user work and are not part of this hardening commit.

## Remaining limitations

- one-conjunction, <=4-predicate synthesis grammar; no DNF/tree synthesis yet;
- no synthesized termination/abort/recovery state machine;
- active counterexamples come from the configured exact validation pool, not adversarial PSRO;
- compact feature vocabulary can reject a candidate even if a legal feature outside that vocabulary could separate it;
- full historical symbolic resynthesis over hundreds of cases is CPU-expensive enough to exceed the bounded 30-second audit cap;
- sealed failures reject rather than automatically launching a fresh-data resynthesis cycle;
- official-Python champion validation remains a later promotion step and was explicitly excluded here.

## Readiness

**READY for another corrected P0.7 Stage-A CUDA synthesis campaign.** The generic foundation actively incorporates exact validation counterexamples, requires both original target capability and gated independent transfer, keeps sealed evidence frozen, rejects exact legal-state ambiguity and illegal identity features, preserves validated historical specialist/composer semantics, and emits an auditable deterministic trace.
