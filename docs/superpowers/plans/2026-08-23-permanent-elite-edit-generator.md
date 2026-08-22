# Permanent Elite BKG4 Edit Generator Implementation Plan

> **Execution record:** This file preserves the initial implementation checklist as written. The unchecked boxes are historical planning state, not current readiness status. Final measured completion/validation evidence is recorded in `tools/bkg4_search/edit_search/VALIDATION.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before the terminal commit.

**Goal:** Upgrade the validated permanent BKG4 edit-search subsystem so discovery quality is explicitly separated from certified generalization quality, CEGIS can repair near-feasible candidates, screens remain balanced under counterexample pressure, and rich proposal generation no longer dominates round walltime.

**Architecture:** Preserve BKG4/ModuleGenome, exact CUDA truth, CaseRegistry, transactional corpus/archive persistence, and the existing recursive emitter/QD architecture. Add a certification layer and reusable multi-temporal public-context bank builder, make temporal localization hierarchical rather than mandatory, persist deterministic SearchGoal caches, add bounded exact-CUDA CEGIS repair and quality-aware search signals, then export a schema-versioned high-information corpus with discovery/source/certification outcomes kept separate.

**Tech Stack:** Python 3, PyArrow/Parquet, existing B4GM1/B4EC1 CUDA evaluator, existing `bkg4_information_state_capture.cu`, BKG4 CPU C++ parity harness.

**Spec:** User terminal-pass request in this conversation, sections 0–69.

## Global Constraints

- BKG4/Bonsai remains the backbone; no BKG5/VM ABI change unless a proven grammar-capacity blocker requires it.
- All meaningful game evaluation uses exact CUDA; no heavy CPU simulation campaign.
- No BonsaiEditFormer training, neural edit generator, champion change, Kaggle push, or submission.
- Search/model predicates use only legal public context; opponent/team/replay/seed metadata remains evaluation provenance only.
- Final repository must be locally committed and clean.
- Existing transaction, CaseRegistry, leakage, semantic-dedup, recall-audit, same-parent benchmark, and parity guarantees must remain green.

---

### Task 1: Certified-quality semantics and role separation

**Files:**
- Create: `tools/bkg4_search/edit_search/certification.py`
- Modify: `tools/bkg4_search/edit_search/archive.py`
- Modify: `tools/bkg4_search/edit_search/__init__.py`
- Test: `tools/bkg4_search/edit_search/test_certification.py`
- Test: `tools/bkg4_search/edit_search/test_archive.py`

**Interfaces:**
- Produce `DiscoveryQuality`, `CertificationClass`, `TransferFailure`, `PanelCertification`, `CertificationVector`, `certify_program(...)`.
- Archive persists certification evidence keyed by program ID and validates discovery/certification case disjointness and evidence identity.

- [ ] Write RED tests proving discovery elite is not certified elite without certification evidence, unsafe transfer becomes a generalization hard negative, safe-inert transfer becomes a generalization boundary, and certified elite requires safe+active+useful independent evidence.
- [ ] Run those tests and verify they fail because certification types/storage do not exist.
- [ ] Implement vector/classification semantics using recomputed `SafetySpec` assessments; retain activation/behavior/useful/L2W retention and worst certification protected delta without one magic scalar.
- [ ] Add archive persistence/integrity for certification evidence and bump breaking search/emitter/corpus versions as required.
- [ ] Run focused tests to GREEN.

### Task 2: Reusable multi-temporal public context banks

**Files:**
- Create: `tools/bkg4_search/edit_search/context_bank.py`
- Modify: `tools/bkg4_search/edit_search/cli.py`
- Test: `tools/bkg4_search/edit_search/test_context_bank.py`

**Interfaces:**
- Produce `strategic_probe_schedule()`, `build_context_bank_from_r0(...)`, role enum/policy for `discovery`, `certification`, and `ranker_validation`.
- CLI command builds a CaseBank from an aligned base bank + legal R0 capture; it never reads hidden-truth capture.

- [ ] Write RED tests for five temporally separated strategic windows, both seats, identical public registry, role tagging, and no identity/provenance fields in model context.
- [ ] Run RED tests.
- [ ] Implement schedule derived from 24-step days and 72-step shop cadence using broad strategic windows across 0–719, and reuse `load_r0`/`registry_manifest`.
- [ ] Add CLI surface that only consumes B4ISR01 public capture plus exact baseline margins/case metadata.
- [ ] Run focused tests GREEN and build bounded local discovery/certification/validation banks from existing CUDA R0 infrastructure.

### Task 3: Generalization-first contrastive synthesis

**Files:**
- Modify: `tools/bkg4_search/edit_search/goals.py`
- Modify: `tools/bkg4_search/edit_search/emitters.py`
- Modify: `tools/bkg4_search/edit_search/safety.py`
- Test: `tools/bkg4_search/edit_search/test_goals.py`
- Test: `tools/bkg4_search/edit_search/test_emitters.py`
- Test: `tools/bkg4_search/edit_search/test_safety.py`

**Interfaces:**
- Produce temporal-specificity helpers and synthesis modes that prefer non-temporal semantic separation, then broad phase windows, narrow windows, exact STEP last.
- Safety/production ordering uses temporal specificity only after hard feasibility and strategic quality.

- [ ] Write RED tests proving non-temporal separators outrank equivalent temporal ones and rich contexts no longer force every contrastive program to contain STEP.
- [ ] Write RED tests for broad-window > narrow-window > exact-step ordering and shifted-context activation diagnostics.
- [ ] Run RED tests.
- [ ] Implement non-temporal-first condition synthesis and hierarchical temporal variants; remove DAY/HOUR from primary failure clustering signature.
- [ ] Add temporal specificity as a late lexicographic production/generalization tiebreaker, never ahead of safety/useful quality.
- [ ] Run focused tests GREEN.

### Task 4: Persistent deterministic proposal caches and hot-path batching

**Files:**
- Modify: `tools/bkg4_search/edit_search/archive.py`
- Modify: `tools/bkg4_search/edit_search/search.py`
- Modify: `tools/bkg4_search/edit_search/emitters.py`
- Test: `tools/bkg4_search/edit_search/test_search.py`
- Test: `tools/bkg4_search/edit_search/test_emitters.py`

**Interfaces:**
- Persist synthesized condition tuples keyed by `(case_bank_identity, goal_id, grammar_id, feature_registry_version)`.
- Cache immutable parent typed metadata/pools keyed by `(parent semantic id, grammar/version, route identity, context identity, escalation)`.

- [ ] Write RED tests for SearchGoal cache restore after archive reload and invalidation on bank/goal/schema change.
- [ ] Write RED tests for deterministic parent typed-cache reuse across generations and identical candidate identity/order.
- [ ] Run RED tests.
- [ ] Persist compact deterministic condition cache in runtime/archive state; never persist outcome labels inside the proposal cache.
- [ ] Cache/reuse typed candidate pools and immutable parent metadata; batch emitter access to shared pools rather than rebuilding per micro-budget.
- [ ] Re-profile mature archive and require major reduction from the measured 20.32 s / 3.15 unique/s baseline before moving on.

### Task 5: Balanced counterexample memory and target-aware recall

**Files:**
- Modify: `tools/bkg4_search/edit_search/panels.py`
- Modify: `tools/bkg4_search/edit_search/search.py`
- Test: `tools/bkg4_search/edit_search/test_panels.py`
- Test: `tools/bkg4_search/edit_search/test_multifidelity.py`

**Interfaces:**
- Screening quotas cover counterexamples, target losses, fragile wins, strong wins, random anchors, both seats.
- Counterexample memory is compressed by deterministic public-context/failure/family signature.
- Detection recall adds `target_repair_recall`.

- [ ] Write RED test with hundreds of hard cases proving the screen still contains losses, fragile wins, strong wins, counterexamples, and both seats.
- [ ] Write RED test proving redundant hard cases compress to bounded representatives.
- [ ] Write RED test for target-repair detection recall and composition refresh before raw screen growth.
- [ ] Implement adaptive quota selection and representative counterexample compression.
- [ ] Extend audit recall/governor with target-repair recall and composition refresh.
- [ ] Run focused tests GREEN.

### Task 6: Counterexample-guided repair cascade

**Files:**
- Create: `tools/bkg4_search/edit_search/repair.py`
- Modify: `tools/bkg4_search/edit_search/search.py`
- Modify: `tools/bkg4_search/edit_search/archive.py`
- Modify: `tools/bkg4_search/edit_search/cli.py`
- Test: `tools/bkg4_search/edit_search/test_repair.py`
- Test: `tools/bkg4_search/edit_search/test_multifidelity.py`

**Interfaces:**
- Produce bounded same-frame multi-counterexample separator synthesis and `run_cegis_repair(...)` returning registered exact-evaluated repair programs/chains.
- Repair archive records repairability: violation count/coherence, legal separator availability, remaining grammar capacity, activation localization.

- [ ] Write RED synthetic exact-oracle test where an unsafe promising rule is repairable by one public predicate and becomes root-feasible.
- [ ] Write RED multi-counterexample set-cover/beam test and bounded cascade termination tests.
- [ ] Run RED tests.
- [ ] Implement smallest-guard repair over 1–2 added predicates up to grammar capacity; prioritize target retention, zero CE activation, minimum protected activation, non-temporal separators first.
- [ ] Integrate bounded exact-CUDA CEGIS cascade after discovery full-panel evaluation; track attempts, reduced violations, feasible/useful transitions and chain provenance.
- [ ] Run focused tests GREEN and bounded real CUDA repair campaign; require real unsafe→feasible transitions before acceptance.

### Task 7: Certification-aware motifs, donors, MCTS, and quality stagnation

**Files:**
- Modify: `tools/bkg4_search/edit_search/archive.py`
- Modify: `tools/bkg4_search/edit_search/search.py`
- Modify: `tools/bkg4_search/edit_search/emitters.py`
- Test: `tools/bkg4_search/edit_search/test_search.py`
- Test: `tools/bkg4_search/edit_search/test_emitters.py`

**Interfaces:**
- Certified motif metadata is distinct from discovery motif metadata.
- Recombination/MCTS parent quality lexicographically prefers certified feasible > discovery feasible > repair > exploration.
- `GenerationSignals` carries certified progress and repair→feasible progress; quality escalation has eight bounded levels.

- [ ] Write RED tests for certified motif/donor preference, certification-aware MCTS ordering, and quality stagnation despite unsafe novelty.
- [ ] Run RED tests.
- [ ] Implement certified motif counters/donor ordering and structured quality rank rather than collapsing certification into an opaque reward.
- [ ] Add quality-specific escalation levels: threshold breadth, alternate clusters, non-temporal synthesis, stronger CEGIS, cross-context temporal search, certified motifs, near-feasible repair, grammar-capacity diagnostic.
- [ ] Run focused tests GREEN and bounded anti-saturation campaign demonstrating renewed feasible/useful progress.

### Task 8: Certified corpus schema and high-information sampling

**Files:**
- Modify: `tools/bkg4_search/edit_search/corpus.py`
- Modify: `tools/bkg4_search/edit_search/cli.py`
- Test: `tools/bkg4_search/edit_search/test_corpus.py`
- Test: `tools/bkg4_search/edit_search/test_cli.py`

**Interfaces:**
- Corpus stores separate source-discovery, root-discovery, and certification exact rows using a label-role field.
- Program rows persist discovery class, certification class/vector, generalization class, temporal specificity, and split groups for parent/child/skeleton/target/temporal regime/generation/certification panel.
- Frozen ranker-validation bank identity is metadata only and its cases are never training rows.

- [ ] Write RED tests for discovery elite→safe-inert boundary, discovery elite→unsafe hard negative, and certified elite-only top-positive semantics.
- [ ] Write RED test that frozen validation cases cannot appear in training CaseBank rows.
- [ ] Write RED tests for no-op capping and deliberate retention of generalization failures/near-feasible repairs.
- [ ] Implement schema bump and recomputation validator; all future-truth fields remain outside model feature allowlists.
- [ ] Add deterministic high-information generation sampler; obvious/static/screen no-ops are dropped before expensive final labeling where possible and final no-op fraction is capped at 10% when alternatives exist.
- [ ] Add certified-elite local-neighborhood generation (guard deletion, threshold delta, window width, route/priority alternatives) as bounded exact-evaluated supervision.
- [ ] Run corpus tests GREEN and produce a fresh bounded corpus with certified positives and <=10% no-op.

### Task 9: End-to-end certification workflow and final evidence

**Files:**
- Modify: `tools/bkg4_search/edit_search/benchmark.py`
- Modify: `tools/bkg4_search/edit_search/cli.py`
- Modify: `tools/bkg4_search/edit_search/parity.py`
- Modify: `tools/bkg4_search/edit_search/README.md`
- Modify: `tools/bkg4_search/edit_search/VALIDATION.md`
- Test: relevant benchmark/parity/CLI tests.

**Interfaces:**
- Production search accepts discovery + rotating certification banks and optional frozen ranker-validation identity.
- Final benchmark reports discovery metrics and certified retention separately.

- [ ] Build fresh multi-temporal discovery/certification banks with zero case overlap and reserve a frozen ranker-validation bank not exported to training.
- [ ] Run at least five fresh benchmark seeds excluding all prohibited seeds; compare old shallow/random/fixed/final permanent under identical root/case/CUDA budget.
- [ ] Certify top cohorts on disjoint cases; require >=80% safety, >=50% behavior, >=40% safe-useful and certified elite >0, without weakening gates.
- [ ] Run bounded CEGIS and quality anti-saturation gates, screen-composition/recall evidence, and fresh bounded certified corpus audit.
- [ ] Re-profile full round and require proposal generation <=25% walltime or stronger candidate-throughput evidence.
- [ ] Extend non-empty parity with temporal 3/4-condition overlay and rerun base/native/generic parity.
- [ ] Run 93+ edit-search tests, 59 edit-corpus tests, Bonsai search tests, compileall, `git diff --check`.
- [ ] Update README/VALIDATION from measured final evidence only; review diff, commit all permanent code/docs/tests, verify clean status, release task lock.
