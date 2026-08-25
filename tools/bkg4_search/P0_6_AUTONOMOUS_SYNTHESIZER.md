# P0.6 Bounded Autonomous Bonsai Specialist Synthesizer

**Final verdict: COMPLETE — terminal strategic result `REJECTED`.** The P0.6 synthesizer ran end-to-end from the P0.5 automatic-composer parent, autonomously selected the highest-priority loss frontier, searched coherent route macros on CUDA, captured legal current state, attempted bounded effect-aware CEGIS gate synthesis, rejected hindsight-ambiguous/unsafe candidates, exhausted its configured eight-candidate attempt budget, and terminated without infrastructure failure.

The terminal reason is:

```text
SEARCH_BUDGET_EXHAUSTED
```

This is a bounded strategic rejection, not a synthesizer failure and not evidence that every possible frontier or specialist has been exhausted.

## 1. Repository baseline

- repository: `/home/nnmax/Desktop/kaggle/kaggriculture`
- team: `ReCurSiON`
- branch: `main`
- P0.5 committed baseline: `c69b60a841f80052514bc2be6a6ce4309f8cf0f8`
- P0.5 evidence: `tools/bkg4_search/P0_5_AUTOMATIC_COMPOSER.md`
- P0.5 parent controller: `bonsai-kg/integrations/bkg4_champion/champions/bounded_automatic_composer_v1/controller.bkgc`
- P0.5 parent routes: `.../bounded_automatic_composer_v1/routes.bkv6`
- P0.5 parent legacy routes: `.../bounded_automatic_composer_v1/legacy_routes.bkv6`
- P0.5 parent structure: 398 nodes, 26 BB slots, 0 state slots, max depth 11

The five unrelated pre-existing untracked files present before P0.6 were not modified or deleted.

## 2. P0.6 objective

P0.6 implements the bounded autonomous Bonsai specialist synthesis loop:

```text
frontier selection
-> coherent CUDA macro search
-> legal state capture
-> effect-aware bounded CEGIS
-> mixed exact-state rejection
-> legal gate derivation
-> macro/declaration creation
-> P0.5 composition
-> certification
-> CPU/CUDA parity
-> bounded retry
-> terminal verdict
```

The top-level terminal taxonomy is:

```text
ACCEPTED
REJECTED
INFRASTRUCTURE_FAILURE
```

A bounded `REJECTED` outcome is valid completion. P0.6 does not force acceptance.

## 3. Architecture and bounded-autonomy contract

Permanent P0.6 files:

- `tools/bkg4_search/bonsai_synthesizer.py`
- `tools/bkg4_search/test_bonsai_synthesizer.py`
- `tools/bkg4_search/p0_6_autonomous_synthesizer_v1.json`
- generic bounded-host change in `tools/bkg4_search/macro_initiation_cegis.py`
- this evidence document

The implementation is declarative and fail-closed:

- no target family is encoded in config;
- no final donor route, start, duration, gate, threshold, opponent identity, replay identity or hidden truth is encoded in config;
- frontier ranking is deterministic from observed baseline losses, both-seat recurrence, independent recurrence and severity;
- CUDA owns large rollout/search/evaluation work;
- CPU work is bounded to orchestration, compact gate proposal/CEGIS and parity if a candidate reaches parity;
- retries are bounded by `budget.max_candidate_routes = 8`;
- gate grammar is bounded to at most four conditions, a compact public-state feature vocabulary and a 24-feature contrastive proposal budget;
- redundant singleton loss-to-win proposal searches are capped at four per CEGIS invocation;
- exact mixed-state detection is not truncated: the exact legal signature contains the full captured `registry + BB + compat` vector;
- candidate-specific intermediate state is reset between retries and preserved in `candidate_attempts` snapshots;
- external commands are run synchronously and failures are checked before output is consumed.

The legal proposal vocabulary includes runtime `OWN_SEED`; offline seed identity remains forbidden.

## 4. Config and final run identity

Config:

```text
tools/bkg4_search/p0_6_autonomous_synthesizer_v1.json
```

Configured evidence banks:

- frontier discovery: `scratch/p0_2_macro_cegis/current140_7.tsv`
- transfer / CEGIS: `scratch/p0_3_repeatability/core585.tsv`
- fresh current-top10 discovery: `scratch/top10_archive_20260824/balanced_panels/discovery.tsv`
- fresh current-top10 validation: `scratch/top10_archive_20260824/balanced_panels/validation.tsv`
- fresh current-top10 sealed holdout: `scratch/top10_archive_20260824/balanced_panels/sealed.tsv`
- broad certification: `scratch/p0_3_repeatability/cert745.tsv`
- historical/archive protection: `scratch/p0_2_macro_cegis/current_archive_aug23_7.tsv`
- coherent donor bank: `scratch/p0_4_third_specialist/donors64.txt`

The coarse `family=0` field in the Aug24 bank is not used as a genuine opponent-family identity for routing.

Final deterministic run ID:

```text
e36da3968e42084e77d77d7a
```

Authoritative run report:

```text
scratch/p0_6_autonomous_synthesizer/e36da3968e42084e77d77d7a/synthesis_run.json
```

After the final closeout fixed the residual `OWN_SEED` no-harm proposal inconsistency, the run directory was deleted and the production config was rerun from clean P0.6 scratch. The rerun reproduced the same run ID, frontier, candidate ordering, eight rejection outcomes and CUDA game count.

## 5. Infrastructure defects found and generic fixes

### 5.1 Capture escaped-newline defect

An earlier P0.6 capture path wrote escaped newline text rather than the required newline-delimited probe input. The capture path was corrected generically so probe files contain real newline-separated steps. Current production code writes the selected candidate start dynamically and does not hardcode a historical probe step.

### 5.2 Original broad CEGIS host explosion

The original contrastive gate proposal path exposed too broad a feature search and could consume excessive host CPU. P0.6 now constrains proposal generation to a compact legal public-state vocabulary and a 24-feature proposal budget while keeping final effect scoring exact over the supplied cases.

This bounds proposal search; it does not truncate the exact legal-state signature used to detect hindsight ambiguity.

### 5.3 Remaining singleton L->W CEGIS explosion

The 24-feature budget alone was insufficient because `enumerate_safe_conjunctions()` still launched one full contrastive search for every individual loss-to-win case in addition to aggregate searches.

Observed real candidate transfer counts included:

- route408/start48/duration24: 120 positive, 455 harmful, 21 L->W;
- route409/410 variants: up to 27 L->W in the affected transfer setting.

That meant roughly 23-29 full contrastive searches for one candidate. The generic fix preserves:

- aggregate positive target search;
- aggregate L->W target search;
- the complete harmful set;
- exact legal state;
- complete effect scoring;

while capping only redundant singleton L->W proposal searches. P0.6 uses:

```text
singleton_l2w_budget = 4
```

A regression test verifies the singleton proposal count is bounded.

### 5.4 `OWN_SEED` legal-runtime false positive

The identity-leakage validator originally treated any feature containing the substring `seed` as offline seed identity. That incorrectly rejected legal runtime feature `OWN_SEED`.

The validator was corrected so `OWN_SEED` is legal while identity-like source/seed/replay/opponent fields remain forbidden. During final closeout, code review found the no-harm CEGIS proposal branch still had an independent substring filter that excluded `OWN_SEED`. A focused red-then-green regression reproduced that inconsistency; the no-harm branch and main validator now share the same identity-feature predicate.

Because that was a runtime gate-synthesis semantic change, the full deterministic P0.6 production config was rerun from clean scratch. The strategic result remained unchanged.

### 5.5 Terminal verdict and evidence bookkeeping

Earlier versions surfaced strategic reasons such as `SEARCH_BUDGET_EXHAUSTED` or `NO_SAFE_SPECIALIST_FOUND` directly as the top-level verdict. P0.6 now uses the required top-level taxonomy:

```text
verdict = REJECTED
terminal_reason = SEARCH_BUDGET_EXHAUSTED
```

Candidate retry bookkeeping was also fixed so `transfer`, `cegis`, gate, threshold, macro, composition, certification and parity evidence are snapshotted per attempt rather than being silently overwritten by the next candidate.

## 6. Final frontier ranking

The final production run ranked the top frontiers as follows:

| Rank | Family | Losses | Both-seat trajectories | Independent trajectories | Severity | Priority score |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 10 | 7 | 3 | 4 | 66,137 | `[3, 7, 4, 66137, -10]` |
| 2 | 4 | 6 | 3 | 3 | 63,390 | `[3, 6, 3, 63390, -4]` |
| 3 | 0 | 6 | 3 | 3 | 10,413 | `[3, 6, 3, 10413, 0]` |

The global candidate-attempt budget was exhausted while processing the highest-priority frontier. P0.6 therefore did not claim that every lower-ranked family or every possible specialist was exhausted.

## 7. Selected family 10 evidence

Family 10 was selected autonomously, not hardcoded.

Selected evidence:

```text
losses                    = 7
severity                  = 66137
independent trajectories  = 4
both-seat trajectories    = 3
priority score            = [3, 7, 4, 66137, -10]
```

Selected baseline margins:

```text
-17341
-16061
-12291
-12217
-4031
-4031
-165
```

The production implementation/config contains no family-10 branch.

## 8. Top autonomous candidate ranking

The highest-ranked target-family candidates were:

| Rank | Donor route | Start | Duration | Positive | Harmful | L->W | W->L | Independent positive | Both-seat positive | Sum delta | Worst delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 408 | 48 | 24 | 5 | 2 | 3 | 0 | 3 | 2 | +9,920 | -19,331 |
| 2 | 409 | 48 | 24 | 5 | 2 | 3 | 0 | 3 | 2 | +9,899 | -19,338 |
| 3 | 410 | 48 | 24 | 5 | 2 | 3 | 0 | 3 | 2 | +9,899 | -19,338 |
| 4 | 407 | 48 | 24 | 5 | 2 | 2 | 0 | 3 | 2 | +15,626 | -16,689 |
| 5 | 409 | 42 | 24 | 2 | 5 | 2 | 0 | 1 | 1 | -56,347 | -30,354 |
| 6 | 410 | 42 | 24 | 2 | 5 | 2 | 0 | 1 | 1 | -56,347 | -30,354 |
| 7 | 409 | 36 | 36 | 2 | 5 | 2 | 0 | 1 | 1 | -61,335 | -30,485 |
| 8 | 410 | 36 | 36 | 2 | 5 | 2 | 0 | 1 | 1 | -61,335 | -30,485 |

These candidates were selected by the search evidence. None is encoded as a final route/start/duration in the config or production code.

## 9. Eight exact bounded rejection summaries

The configured global candidate-attempt budget is eight. The final run exhausted exactly eight attempts:

| Attempt | Route | Start | Duration | Rejection | Stage | Mixed exact-state cells |
|---:|---:|---:|---:|---|---|---:|
| 1 | 408 | 48 | 24 | `NO_SAFE_GATE` | `cegis` | 11 |
| 2 | 409 | 48 | 24 | `NO_SAFE_GATE` | `cegis` | 11 |
| 3 | 410 | 48 | 24 | `NO_SAFE_GATE` | `cegis` | 11 |
| 4 | 407 | 48 | 24 | `NO_SAFE_GATE` | `cegis` | 13 |
| 5 | 409 | 42 | 24 | `NO_SAFE_GATE` | `cegis` | 10 |
| 6 | 410 | 42 | 24 | `NO_SAFE_GATE` | `cegis` | 10 |
| 7 | 409 | 36 | 36 | `MIXED_LEGAL_STATE` | `legal_state` | 3 |
| 8 | 410 | 36 | 36 | `MIXED_LEGAL_STATE` | `legal_state` | 3 |

No infrastructure exception masked these strategic rejections.

## 10. Mixed exact-state and hindsight evidence

The proposal vocabulary is deliberately compact, but mixed-state detection uses the full legal information-state signature:

```text
registry + BB + compat
```

The exact signature is not truncated to the compact proposal vocabulary.

The final run found exact legal-state cells containing both positive and harmful effects for the same candidate macro. At the captured activation state, those cases are indistinguishable using the full legal current information available to the controller, yet the macro has opposite downstream effects.

Therefore selecting only the future-positive cases would require hindsight or a non-runtime label such as opponent identity, replay/source identity, offline family label, hidden truth or future state. P0.6 correctly rejects that routing strategy.

## 11. Why no gate was accepted

The candidate macros contain genuine useful local behavior: the top four improve five of seven target-family cases and convert multiple losses to wins. They are not globally harmless, however.

For six attempts, bounded legal CEGIS could not produce a gate that retained useful support while excluding observed harmful cases and mixed exact-state cells. For the final two attempts, the legal-state check established mixed exact-state ambiguity early enough to reject before CEGIS.

No manual opponent/family/source/seed label was introduced to rescue the candidates.

## 12. CUDA game count and host boundedness

Final production rerun:

```text
CUDA games   = 22106
host seconds = 228.826718
```

The earlier runaway host behavior was removed by bounding proposal breadth and redundant singleton L->W searches. The final deterministic production run completed normally rather than hanging.

No heavy CPU search campaign was used for closeout.

## 13. Final terminal verdict

```text
P0.6 COMPLETE
TERMINAL STRATEGIC RESULT = REJECTED
```

## 14. Terminal reason

```text
SEARCH_BUDGET_EXHAUSTED
```

The eight configured candidate attempts were consumed on the highest-priority frontier.

## 15. Why this is not infrastructure failure

The final run:

- completed frontier CUDA baseline evaluation;
- ranked frontiers deterministically;
- completed coarse and refinement CUDA macro search;
- completed transfer/validation/sealed effect evaluation and legal-state capture for attempted candidates;
- executed bounded CEGIS where applicable;
- preserved per-attempt evidence;
- produced the final machine-readable run report;
- terminated with `REJECTED` and `SEARCH_BUDGET_EXHAUSTED`;
- recorded no `INFRASTRUCTURE_FAILURE` terminal verdict.

The CLI uses exit status 2 for a non-accepted strategic result, but the machine report is complete and contains no infrastructure exception.

## 16. No new specialist was composed

Every candidate failed before a legal safe gate survived. Therefore:

```text
specialist  = null
composition = null
```

No new P0.6 route macro was promoted into the P0.5 parent. The existing P0.5 champion remains unchanged.

## 17. Certification, parity and official Python checks are not applicable

Broad final candidate certification, CPU/CUDA differential parity and the mandatory official-Python 64/128-game submission check apply to an accepted composed candidate.

No P0.6 candidate reached composition. Therefore:

```text
new-specialist broad certification = NOT APPLICABLE
new-specialist CPU/CUDA parity      = NOT APPLICABLE
official Python 64/128 games        = NOT APPLICABLE
```

P0.6 does not claim parity or official-Python validation for a nonexistent accepted specialist.

## 18. Determinism

The config canonicalization produces run ID:

```text
e36da3968e42084e77d77d7a
```

After the final runtime consistency fix, the previous run-id scratch directory was removed and the production config was rerun. The rerun reproduced:

- the same run ID;
- the same 22,106 CUDA games;
- the same family ordering (10, 4, 0);
- the same family-10 margins and priority score;
- the same top candidate ordering;
- the same eight candidate routes/start/durations;
- the same rejection reasons/stages and mixed-cell counts;
- the same final `REJECTED / SEARCH_BUDGET_EXHAUSTED` result.

Deterministic ordering does not depend on filesystem iteration order or manually encoded candidate preference.

## 19. Final code-review findings

Closeout review verified:

- no family-10 hardcoding in production code/config;
- no route408/409/410 hardcoding in production code/config;
- no candidate-specific final gate/start/duration/threshold config field;
- no opponent/team/replay/source/family-label routing gate;
- legal runtime `OWN_SEED` remains permitted;
- exact mixed-state signature is full `registry + BB + compat`;
- proposal vocabulary and feature budget remain bounded;
- singleton L->W proposal searches remain bounded to four;
- retries remain bounded to eight in the production config;
- top-level strategic exhaustion returns `REJECTED` with separate `terminal_reason`;
- attempt-level evidence is preserved;
- process execution is synchronous and command return codes are checked before consuming generated outputs;
- deterministic sort/tie-break paths remain explicit.

The only genuine closeout defect found was the residual no-harm `OWN_SEED` substring filter described above; it was fixed and the production campaign was rerun.

## 20. Test evidence

Final P0.6 lightweight verification commands:

```text
python3 -m unittest tools.bkg4_search.test_bonsai_synthesizer
29/29 PASS

python3 -m unittest tools.bkg4_search.test_bounded_macro_cegis
16/16 PASS

python3 -m py_compile \
  tools/bkg4_search/bonsai_synthesizer.py \
  tools/bkg4_search/test_bonsai_synthesizer.py \
  tools/bkg4_search/macro_initiation_cegis.py
PASS

git diff --check
PASS
```

The `OWN_SEED` regression was observed failing before the production fix and passing after it.

No giant CPU test suite was run.

## 21. Final repository and commit state

The intended P0.6 closeout scope is exactly:

- `tools/bkg4_search/bonsai_synthesizer.py`
- `tools/bkg4_search/test_bonsai_synthesizer.py`
- `tools/bkg4_search/p0_6_autonomous_synthesizer_v1.json`
- `tools/bkg4_search/macro_initiation_cegis.py`
- `tools/bkg4_search/P0_6_AUTONOMOUS_SYNTHESIZER.md`

The closeout commit is local-only; no push and no Kaggle submission are part of P0.6. Because a file cannot stably embed the hash of the commit that contains itself, the exact resulting commit SHA is reported in the final closeout response after commit creation.

Ignored P0.6 scratch evidence remains under `scratch/`. The same five unrelated pre-existing untracked files are intentionally excluded from the P0.6 commit and remain untouched.

## 22. Final interpretation

P0.6 has achieved its intended engineering objective: an autonomous bounded Bonsai specialist synthesizer now connects frontier discovery, coherent CUDA macro search, legal-state capture, compact effect-aware gate synthesis, mixed-state rejection, P0.5 composition, downstream certification/parity hooks, bounded retries and a machine-readable terminal verdict.

This run did not find a legal safe specialist within its configured eight-candidate budget. That is useful strategic evidence rather than infrastructure failure. The correct closeout state is:

```text
P0.6 COMPLETE
TERMINAL STRATEGIC RESULT = REJECTED
TERMINAL REASON = SEARCH_BUDGET_EXHAUSTED
P0.5 CHAMPION = UNCHANGED
```
