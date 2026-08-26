# RLRMB

## Purpose

Regret-Localized Reactive Macro Beam Search (RLRMB) is the new authoritative
capability candidate-generation system under construction. It replaces the old
candidate manifold:

```text
donor x start x duration
```

with richer structured capability generation. Phases 0 through 3 establish
representation and search-control contracts only; they do not evaluate or improve
candidate quality.

## Frozen baseline

Hardened P0.7 Stage-A run:

```text
run_id = 536dd7c9f4f5b393c99ccf3f
CUDA games = 265,226
refined attempts = 88
accepted = 0
MIXED_LEGAL_STATE = 49
NON_TRANSFERABLE = 38
TARGET_CAPABILITY_LOST = 1
```

This frozen result is the future equal-budget comparison baseline. Phase 0 makes
no claim to improve it.

## Responsibility boundary

RLRMB owns capability candidate generation. It is not a second CEGIS
implementation.

The existing hardened system continues to own outer initiation CEGIS,
target-capability preservation, transfer validation, sealed holdout, threshold
robustness, automatic composition, broad certification, and CPU/CUDA parity.

## Canonical identity contracts

Structural candidate IDs hash canonical semantic content, not donor route IDs,
filenames, artifact paths, or attempt numbers. Fragment-content IDs hash only the
ordered normalized actions in the selected window; candidate IDs separately retain
semantically relevant timing such as start and duration. Market-order order and hand
order remain significant.

Effect signatures identify exact ordered, role-separated panel response vectors.
They include baseline margins, per-case margin deltas, sign and L-to-W/W-to-L
patterns, and per-case action-change counts when available. Donor and attempt
provenance is excluded.

Read-only analysis of the hardened Stage-A run found 52 unique target effect
signatures and 54 unique target-plus-transfer signatures among 88 attempts. These
strict signatures include exact panel identity and action-change vectors. Omitting
action changes produces fewer classes, but would discard a required observed effect
dimension. Thus target effects contain 36 duplicates (40.9% of attempts), while
combined target-plus-transfer effects contain 34 duplicates (38.6%).

The same 88 attempts reduce to 50 structural candidates when identified by
canonical fragment content plus start and duration, leaving 38 duplicate structural
attempts.

The common 64-route donor bank contained the following unique local fragments:

```text
start 24, duration 24: 17
start 48, duration 24: 17
start 60, duration 12: 15
start 84, duration 12: 20
start 120, duration 18: 30
start 156, duration 48: 44
start 264, duration 24: 48
```

This confirms that nominal donor count substantially overstates local behavioral
diversity, especially in early windows. No search behavior changed in Phase 1.

## Checkpoint selection

Phase 2 adds immutable, versioned offline checkpoints for selecting where future
candidate search should begin. Proposals carry a semantic step/lane/reason/state
identity plus separate severity, regret, recurrence, both-seat, family and evidence
bookkeeping. Offline family/provenance labels do not affect semantic identity or
ranking and never become runtime routing state.

Explicit trace flags and current action semantics produce regret, divergence, shop,
day/EOD, land, labor, animal/structure, market, production/payoff and route/phase
events. Only mechanics where a nearby decision boundary is meaningful receive
bounded before/at/after proposals. Equivalent step/state proposals merge their
reasons deterministically.

Ranking is causal-first, then recurrence, both-seat support, recoverability, regret,
severity, causal relevance and proximity to the event. Supported lanes are market
liquidity, worker production, land investment, crop transition, animal transition,
endgame liquidation, mechanic exploit and general. With no causal evidence, the
only fallback is the sparse seven-step set `0,72,144,216,288,432,576`; fallback is
always lower priority when explicitly included beside causal proposals.

Read-only historical validation reached route435 step156, route410 step120 and
route437 step264 at rank 1 in the appropriate worker-production,
worker-production and animal-transition lanes respectively. All were reached from
regret/effect and action-event evidence, not fallback or route-specific production
rules. Phase 2 changes search control contracts only; it adds no candidate program
or runtime behavior.

## Option-Program IR v1

Phase 3 adds a versioned, immutable `MacroProgram` candidate language. A program
starts at one Phase-2 checkpoint, stays in the checkpoint lane, contains one to
four ordered phases, has a positive cap within the remaining 719-action horizon,
and deterministically returns to parent behavior. Each phase uses exactly one of
`PARENT`, `DONOR_BLOCK`, `SEMANTIC_SPLICE`, or
`PARAMETERIZED_PRIMITIVE`. The only override families are `FARMER`, `HANDS`, and
`MARKET`; selected families come from the phase and every unselected family comes
from the parent.

The bounded v1 termination classes are `FIXED_CAP`, `NEXT_EOD`,
`NEXT_SHOP_REVEAL`, `PURCHASE_COMPLETED`, `TARGET_PUBLIC_STATE`,
`PARENT_COMPATIBLE`, and `MAX_STEPS`. Every termination carries a finite phase cap,
and the program cap remains authoritative. V1 deliberately has no branch, abort,
recovery, or general predicate language.

Semantic motifs identify normalized action content and supported action families.
Historical route numbers, filenames, source windows, and artifact provenance are
retained only for offline audit and do not define motif or program identity.
Read-only validation represented the route435 window 156/48, route410 window
120/18, and route437 window 264/24 as generic donor-block motifs with deterministic
program round trips. A two-phase fixture independently splices a MARKET motif and
a HANDS motif while inheriting all other families from the parent, proving the IR
can express a candidate outside the old single-contiguous-fragment manifold. No
games, evaluator, or search behavior were added in Phase 3.

## Exact evaluator v1

Phase 4 executes Option-Program IR directly against the existing BKG4 controller,
route packs, compatibility overlays, and exact simulator. The audit reference is
the Python IR state machine driven turn-by-turn by the parity-validated C++ engine
and CPU BKG4 runtime; it does not reproduce game physics. CUDA consumes a packed
version-1 batch ABI with fixed-width program, phase, motif, action, and result
records. Programs and shared motifs are uploaded once per candidate-by-case batch.

A phase action is submitted first, then its termination is evaluated exactly once
on the legal post-action state. The triggering action belongs to the current phase;
the next phase starts on the following turn. Event completion takes precedence if
it coincides with a cap. `PARENT_COMPATIBLE` means the masked pre-overlay action is
exactly equal to the parent action. V1 purchase completion is limited to `BUY_LAND`
and `HIRE`, whose completion has an exact legal-state counter; item purchases remain
deferred until the IR can name the item without aggregate-inventory ambiguity.

The golden suite covers all four action sources, all seven family masks, all seven
termination classes, two- and four-phase programs, return-to-parent, horizon and
fault behavior, deterministic repetition, batch-order independence, and bounded
route435/410/437 fixtures. Exact overlap validation for route410 step120/duration18
matched the unchanged fragment evaluator on rewards/margins, action changes, and
faults. The validated parity matrix has zero CPU/CUDA mismatches. Phase 4 adds no
search, QD archive, LNS, or BKG4 materialization.

## Effect-vector QD archive

Phase 5 adds a versioned, deterministic one-elite-per-cell archive keyed by the
existing strategic lane, a coarse initiation/economic-state cluster, an exact
response-shape effect cluster, and one of four broad temporal phases. Initiation
clustering uses only available public capital, land, labor, market and causal-event
regimes; absent dimensions remain explicitly unknown. Effect clustering preserves
ordered panel roles, sign/L-to-W/W-to-L patterns, per-case delta buckets, worst-case
regime and action-change regime. `SEALED` evidence is rejected from archive keys.

Within a cell, runtime validity and W-to-L/harm safety precede target L-to-W,
required and independent support, both-seat recurrence, worst delta, available
routability, simplicity and finally aggregate margin. Provenance-only repeats and
exact structure/effect duplicates collapse, while different response patterns and
lanes remain independent. Canonical replay of considered entries makes elites,
metrics and JSON resume state independent of insertion order.

Read-only ingestion of the frozen 88 P0.7 attempts reproduced 50 structural IDs
and 54 exact target-plus-transfer effect signatures. The archive occupied 49 cells:
49 canonical new-cell insertions, 3 replacements, 33 duplicate rejections and 3
inferior same-cell rejections. It retained 5 target-solving cells spanning 3
distinct L-to-W patterns; 48 retained elites had lower aggregate margin than the
global aggregate leader because they represented different strategic niches.
Scalar aggregate margin is therefore deliberately not the archive objective.
Phase 5 adds no beam, candidate generation, game evaluation, or search campaign.

## Hierarchical diverse beam

Phase 6 adds the first constructive search path:

```text
ranked checkpoint -> deterministic seeds -> one semantic edit -> structural dedup
-> one batched CUDA evaluation -> exact TARGET effects -> QD archive -> diverse elites
```

Seeds cover parent-only skeletons, generic historical motifs, masked semantic
donor phases, and the already executable bounded liquidity primitive. The v1
operators insert a phase, replace a phase source, move a boundary to an adjacent
causal event, change only an action-family mask, truncate/extend to an event,
mutate one bounded numeric parameter by one, or swap one semantic motif. Candidate
ordering uses checkpoint rank, enum order, phase position, motif semantic ID, mask
order and program semantic ID. Canonical program IDs remove structural duplicates
before CUDA. One archive elite per effect cell supplies the next frontier, so no
global scalar top-K list controls survival.

Depth, phase count, candidates per layer and per-family CUDA games are hard caps.
A projected batch is truncated before launch if it would exceed budget. Search
stops on depth, budget, no expansions, or configured rounds with neither archive
novelty nor safe target-quality improvement. CUDA faults are recorded as existing
`CUDA_FAULT` failures and never enter the archive. No SEALED evidence is loaded.

The validated bounded probe used one family-00 checkpoint at step 120, two TARGET
cases, two semantic motifs, depth 1 and exactly 40 CUDA games in two launches. It
evaluated 18 unique programs from 22 raw seed/expansion records, produced 11 exact
effect signatures, and occupied 6 archive cells with 6 replacements. A generated
two-phase program (`7d349d725379857cb0fb92d1efe913fc05c38fb7e9662a3eb0a253a82ce8e6c7`)
applied a HANDS-only semantic splice followed by a MARKET-only splice from distinct
motifs. It executed without fault, produced target deltas `[14755, 12452]`, replaced
its archive elite, and remained retained. Its independently sourced ordered phases
cannot be represented by one contiguous all-family donor fragment. The unchanged
Phase-4 parity suite remained at zero CPU/CUDA mismatches. This was a construction
proof, not a specialist search or acceptance claim.

## Semantic LNS repair

Phase 7 repairs one declared semantic neighborhood rather than individual turns.
Versioned destroy descriptors cover one phase, one action family, an explicit
HANDS+MARKET coupled family set, one causal boundary, one bounded parameter group,
or one semantic motif source. Repairs are generated by the existing Phase-6 mask,
source, boundary, parameter and phase constructors and are filtered so all content
outside the declared neighborhood remains unchanged. Canonical program IDs remove
duplicate repairs before CUDA.

Failure policy sends `BAD_CHECKPOINT` back to checkpoint search, defers
`OPEN_LOOP_FRAGILITY` and `MIXED_INITIATION` to Phase 8, permanently withholds LNS
budget from `FUTURE_DEPENDENT`, prioritizes boundaries for `BAD_HORIZON`, and uses
mask/source/phase/coupled semantic repair for `NON_TRANSFERABLE`. Deterministic
operator statistics track validity, evaluation, target preservation, transfer
improvement and archive success. Target gates preserve original positive cases and
L-to-W cases, prohibit new W-to-L, and enforce a worst-delta floor. Aggregate-only
transfer movement is not sufficient for acceptance.

The bounded CUDA repair probe started from the retained Phase-6 two-source program
`7d349d725379857cb0fb92d1efe913fc05c38fb7e9662a3eb0a253a82ce8e6c7`, used two
TARGET and two independent TRANSFER cases, generated five unique coherent repairs,
and evaluated parent plus repairs in two launches and 28 games. Four repairs
improved transfer while preserving the parent's positive target-delta capability.
Removing the HANDS phase produced program `430027c8212b2a9fc7a9fb0bf246f325f52db8342d531e94ded0889b0b48680c`:
target deltas improved from `[14755,12452]` to `[35695,34730]`, transfer changed
from `[-22059,-9844]` to `[-13440,-14461]` with a better worst case, no target
W-to-L was introduced, and the repair entered a new archive cell. A separate exact
fixture proves MARKET-only and HANDS-only repairs can both be insufficient while
the explicit coupled repair succeeds. Evaluator parity remains unchanged.

## Reactive IR v2

Phase 8 adds an explicitly versioned reactive extension while preserving the v1
open-loop encoding and execution contract. A v2 program may contain at most two
latched decisions after bounded phase/event terminations. Each decision evaluates
one legal `PublicCondition` on the post-action observable state and selects a
forward phase, the single permitted abort-to-parent path, or the single permitted
bounded recovery phase. Continuations are forward-only, recovery always returns
to parent, and the program-wide cap still takes precedence, so cycles and
unbounded execution are structurally impossible. Branch, abort, and recovery
semantics share the same fixed host/device record and retain exact CPU/CUDA parity.

The bounded family0 diagnostic used routes 211/212/420/421 around step 24 without
SEALED data. Route 211 and the common parent route 64 share actions at steps 24 and
25; their first material action divergence is step 26 (`CARE` versus `WEST`). A
later-initiation probe at step 26 retained the original six target positives and
four L-to-W outcomes, but the full legal-state registry still contained positive
and harmful trajectories in the same cells. In particular, transfer cases 10 and
83 have identical legal observations at every captured boundary from step 24
through step 48 while route 211 changes their margins by +99 and -2965. A bounded
v2 predicate probe retained the target vector `[6271,6271,104,4155,4155,104]`, but
could not separate that exact pair. Family0 is therefore classified
`FUTURE_DEPENDENT` under legal v2 semantics rather than given an identity- or
future-dependent branch.

## Legal-routability pre-screen

Phase 9 adds a bounded diagnostic before hardened outer CEGIS. It groups
generation-time TARGET, TRANSFER and VALIDATION trajectories by the exact legal
signature supplied by the existing capture path, compares the first required
continuation divergence with the first legal-state divergence, and returns one of
seven machine-readable outcomes: enter CEGIS, later-checkpoint repair,
common-prefix repair, branch-at-reveal repair, abort/recovery repair, permanent
`FUTURE_DEPENDENT` rejection, or suspected gate-vocabulary limitation. SEALED and
CERTIFICATION evidence fail closed. Identity-like legal features also fail closed;
offline provenance never contributes to classification or score.

A tiny deterministic predicate scan reports legal/gate feature separation across
stable support-group and seat partitions. It is diagnostic only and cannot
override an exact collision. The score is likewise advisory: routable is 100,
bounded repair classes are 50-65, gate limitation is 25, and future dependence is
0. No result denotes specialist acceptance; `ROUTABLE_CANDIDATE` means only that
the candidate is worth submitting to the unchanged hardened CEGIS in Phase 10.

Stored non-sealed validation preserved the Phase-8 family0 result: required action
divergence at step 26 with no legal divergence through step 48 is
`FUTURE_DEPENDENT`. Candidate50 remains permanently non-promotable: its required
target support participates in 30 exact untimely conflicts. Route435's 551
non-neutral stored TARGET/TRANSFER/VALIDATION trajectories classify
`ROUTABLE_CANDIDATE`; 64 mixed pairs concern optional positive support that outer
CEGIS may safely decline, consistent with its historical successful gate.

## Hardened CEGIS adapter

Phase 10 adds a thin, versioned adapter from exact RLRMB per-case evidence to the
existing `MacroEffectCase` and `run_cegis()` contracts. TARGET rows retain required
positive/L->W flags, TRANSFER rows retain independent support groups, VALIDATION
rows remain active counterexamples, and SEALED rows are evaluated only after gate
freeze. Exact legal signatures and the existing threshold-neighbor checks pass
through unchanged. The adapter does not enumerate predicates or own a gate type.

Only `ROUTABLE_CANDIDATE` invokes hardened CEGIS. Repair classifications return to
their owning RLRMB layer, `FUTURE_DEPENDENT` is permanently rejected, and suspected
gate-vocabulary limitations remain diagnostic. For reactive v2 candidates, the
program's internal decisions remain opaque candidate semantics; CEGIS controls
only the outer initiation gate.

The bounded integration suite preserved candidate50 rejection through generic
mixed-state/target-capability contracts and the existing exact stored-artifact
regression. A stored route435 subset loaded from its actual `.b4fr`/`.b4is`
panels traversed the adapter, incorporated two validation counterexamples, retained
required target capability and independent transfer, and reached
`SAFE_CONVERGENCE`. Hardened CEGIS itself is unchanged.

## Deterministic BKG4 materialization

Phase 11 maps an accepted v1/v2 `MacroProgram` and its frozen outer CEGIS DNF gate
to a deterministic composer `SpecialistDeclaration`, ordinary bounded-route BKG4
authority module, and an explicit collision-checked state map. The map owns the
active latch, phase index, phase/program counters, up to two latched branch values,
and the optional abort/recovery flags. Program semantics, gate semantics, and this
mapping all enter the materialization ID; audit provenance does not enter runtime
predicates.

The action-side materialized executor preserves the validated post-action timing:
the triggering action belongs to the current phase, a branch is evaluated once on
the resulting legal observation, its continuation starts on the next action, and
abort/final/recovery completion permanently returns to parent. FARMER, HANDS and
MARKET precedence remains identical to the Phase-4 evaluator. The generated
specialist uses the existing bounded-route composer authority and conflict checks;
no second composer, gate grammar, VM ABI, or runtime Python dependency was added.

The bounded validation covered inactive gate fallback, v1 masked multi-phase
execution, every finite termination descriptor, reactive true/false decisions,
branch latching, abort, recovery and parent return. Nine materialization tests,
the existing 23-test composer regression, and the complete RLRMB suite produced
zero semantic mismatches. The fixture costs one existing composer active latch plus
three program-state slots; reactive fixtures add one slot per decision and only
the abort/recovery slots they use.

## Current status

- Phase 0 - isolated package contract and baseline freeze: validated.
- Phase 1 - structural and effect canonicalization: validated.
- Phase 2 - regret/event checkpoint selector: validated.
- Phase 3 - Option-Program IR v1: validated.
- Phase 4 - exact CPU/CUDA evaluator v1: validated.
- Phase 5 - effect-vector quality-diversity archive: validated.
- Phase 6 - hierarchical diverse beam search: validated.
- Phase 7 - semantic large-neighborhood search repair: validated.
- Phase 8 - bounded reactive branch, abort and recovery: validated.
- Phase 9 - legal-routability pre-screen: validated.
- Phase 10 - adapter to existing hardened CEGIS: validated.
- Phase 11 - deterministic BKG4 materialization and composition: validated.
- Phases 12 and later: not implemented.

## Generated outputs

Permanent source belongs under `tools/bkg4_search/rlrmb/`. Future generated runs
belong under `scratch/rlrmb/`.
