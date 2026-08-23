# Elite Specialist B4IM Generator Validation

Validation date: 2026-08-24

This report validates the permanent B4IM specialist factory. It does not promote a
champion, submit to Kaggle, or train BonsaiEditFormer. All game labels are exact CUDA
results. Opponent names are offline evaluation-cohort metadata only; executable modules
and model inputs contain no opponent, tape, replay, episode, submission, or seed identity.

## Permanent Architecture

The factory deliberately separates specialist discovery from globally safe deployment:

1. Identity-free B4IM proposals express bounded strategic mechanisms.
2. A small tape-disjoint exact-CUDA panel screens a large high-recall pool.
3. Specialist-aware quotas retain targeted positives, global-safe edits, boundaries,
   high-risk specialists, hard negatives, inert controls, repairs, and neighborhoods.
4. A larger disjoint exact-CUDA panel supplies detailed source-relative and root-relative
   labels.
5. The corpus preserves applicability, conflicts, lineage, and composition metadata for
   later ranking and Bonsai gating.

Global harm is a label, not an automatic corpus rejection. Production qualification
remains available as the stricter global-safe tier.

## Final Exact-CUDA Campaign

Command surface: `python3 -m tools.bkg4_search.generate_invariant_specialists`.

- case bank: 506 cases in 253 tape groups
- screen panel: 64 cases / 32 tape groups
- label panel: 400 cases / 200 tape groups, tape-disjoint from the screen
- initial programs: 10,000
- recursive/repair/ablation follow-ups: 2,048
- total unique programs screened: 12,048
- selected and fully labeled: 1,536
- exact games: 1,488,000
- evaluator faults: 0
- wall time: 386.332 seconds
- effective end-to-end throughput: 3,851.607 games/second

The ignored, reproducible artifacts are under
`scratch/b4im_elite_factory/final_10000_v2/`.

## Strategic Coverage

The generator emits 24 identity-free mechanism goals covering opening, midgame, lategame,
endgame, financing, liquidation, overbuy control, workforce, expansion, land unlock,
animal acquisition/rescue/collection, crop rescue, weed recovery, seed/fertilizer
procurement, inventory evacuation, order-cap behavior, production deadlines, maintenance,
and engine-mechanic specialists.

It uses all four active B4IM action families (`ROUTE_OVERRIDE`, `COMMITTED_ROUTE`,
`UNIT_REWRITE`, `MARKET_INTENT`), both control modes, and both domains. The coverage matrix
records representation, emitter, CUDA, distillation, and observed-useful status separately;
it does not infer observed coverage from a broad effect-class label.

## Structural Diversity

Across 12,048 proposals:

- 12,048 canonical programs and semantic IDs
- 6,005 threshold-independent structural skeletons
- 8 mechanism families
- 4 action families
- 2 control modes
- 5 guard-feature families
- top-1 skeleton share: 4.98%
- top-10 skeleton share: 21.41%
- top action-family share: 38.85%
- top mechanism-family share: 37.47%

The 1,536-row labeled corpus retains 975 skeletons and four distinct activation-profile
classes. Its top-1 and top-10 skeleton shares are 5.47% and 21.42%.

Novelty did not collapse with proposal scale:

| Programs | Unique skeletons | Top-1 share |
|---:|---:|---:|
| 256 | 160 | 6.25% |
| 1,024 | 461 | 3.32% |
| 4,096 | 1,523 | 8.79% |
| 10,000 | 4,282 | 6.00% |

All programs were unique at every checkpoint; the curve is not threshold-jitter padding.

## Specialist Yield

Screen yield over 12,048 programs:

- valid: 12,048 (100%)
- behavior-changing: 9,707 (80.57%)
- useful on at least one independent group: 5,559 (46.14%)
- strong specialist: 788 (6.54%)
- elite specialist: 620 (5.15%)
- global-safe useful: 199 (1.65%)
- L-to-W: 2,419 (20.08%)
- hard negative: 4,148 (34.43%)
- inert: 2,341 (19.43%)

The diversity-aware labeled set intentionally has higher useful density: 1,103 meaningful
(71.81%), 997 strong (64.91%), 939 elite (61.13%), 88 global-safe (5.73%), 878 L-to-W
(57.16%), 246 hard negatives (16.02%), and 187 inert controls (12.17%). These categories
are nonexclusive where strategically appropriate.

## Corpus Distribution

The primary source classes in the final 1,536 rows are:

| Source | Rows |
|---|---:|
| global-safe specialist | 57 |
| elite specialist | 582 |
| strong specialist | 42 |
| boundary specialist | 42 |
| CEGIS repair | 134 |
| local neighborhood | 298 |
| structural ablation | 80 |
| hard negative | 166 |
| inert control | 135 |

The corpus includes 1,014 mixed-risk/generalization-boundary examples, 847 narrow
specialists, 1,103 counter-specialists, and explicit compatible/conflicting relationships
for every row. There are 382 positive structural skeletons.

Every row stores exact per-case, tape-group, and offline-cohort applicability; L-to-W,
W-to-L, severe regression, activation, and action-change evidence; root-relative and
immediate-parent-relative effects; B4IM/B4IC ABI provenance; structural niche; lineage;
blackboard reads/writes; feature dependencies; authority; side effects; and compatibility.

## Distinct Hard-Cohort Specialists

Fresh examples demonstrate that different mechanisms attack different cohorts. Downside
contexts are deliberately retained:

- Calmracer regime: a price/inventory liquidation plus weed-recovery module improved 7
  target groups, converted 3 losses to wins, and averaged +1,723 on target groups with no
  target harms. Across all groups it helped 93 and harmed 67 (global mean -5,813).
- Arman regime: a legal public `DAY 0..3 -> route 211` opening specialist improved 5 target
  groups, converted 2 losses to wins, averaged +2,122 on target, and had no target harms.
  Across all groups it helped 168, harmed 8, and averaged +277.
- Subramanya regime: a capital-state route specialist improved 4 target groups, converted
  one loss to a win, and averaged +13,178 on target with no target harms. It helped 21 and
  harmed 125 globally, making its narrow activation requirement explicit.
- Historical ReCurSiON regime: a committed financing route plus inventory evacuation
  improved 3 target groups without target harm. Globally it helped 80 and harmed 72.
- Ryo regime boundary: a public early-day sheep-acquisition specialist had 4 useful and 2
  L-to-W target groups but also 3 target harms. It is correctly retained as boundary data,
  not mislabeled as elite.

The policies recognize public strategic state. The cohort identities above never enter
the B4IM AST or model input.

## Recursive Search, QD, And CEGIS

The specialist archive niches on structural mechanism, action, domain, control, guard
families, target failure, phase, activation breadth, support breadth, risk, and complexity;
scalar threshold jitter does not create a new niche.

Useful specialists seed recursive children and source-relative labels. Neighborhoods can
remove/add/replace guards, alter comparators or thresholds, switch route and commitment
boundaries, change market quantities/priorities, change rewritten operations, ablate rules,
or add compatible rules. Recombination rejects competing route controllers, overlapping
market control, and conflicting unit rewrites.

Grouped CEGIS is in the real proposal queue. Mixed-risk parents are retained while repairs
learn additional legal public-state applicability guards. Repairs carry parent lineage and
are labeled against both the root and immediate parent. The capability ladder escalates
through structural exploration, boundary solving, LNS, semantic recombination, MCTS,
grouped CEGIS, engine specialists, legal composition, and new skeleton families.

## Composition Sanity Proof

A separate exact-CUDA spike evaluated 480 phase specialists and 982 legal compositions on
400 cases / 200 tapes. Two individually incompatible route specialists were separated by
disjoint public `DAY` ranges. Specialist A helped 21 and harmed 4 groups; specialist B
helped 17 and harmed 57. The gated composition helped 26 groups, preserving 11 A-exclusive
and 12 B-exclusive benefits. It still harmed 61 groups, which is retained honestly. This
proves the legal Bonsai/B4IM gating direction; it is not a champion promotion.

## Throughput

- proposal generation: 1,906 unique programs/second in the final run; 1,930/second in the
  standalone 10,000-program profile
- initial screen packing (10,001 x 64): 0.162 seconds
- CUDA process wall: 124.433 seconds
- CUDA kernel: 123.973 seconds, approximately 4,899-5,163 games/second by batch
- result decode: 2.754 seconds
- corpus validation: 2.918 seconds
- ranker-array build: 2.684 seconds
- corpus write: 5.423 seconds
- measured screen capacity: 230,824 candidates/hour (5.54 million/day)
- measured large-panel label capacity: 35,328 candidates/hour (847,866/day)

The evaluator uses resident controller/routes/tapes and one compact 256-byte candidate
record per module. It does not compile or launch once per candidate.

## BonsaiEditFormer Contract

The final leakage-free arrays have shapes `categorical=[1536,19,8]`,
`numeric=[1536,19,10]`, and `labels=[1536,9]`. The structural split has 1,266 train rows
across 784 skeletons and 270 validation rows across 191 skeletons, with zero skeleton
overlap. Sample AST-token-array-AST round trips were exact.

Model inputs contain only pre-outcome parent/edit structure. Outcome, identity, cohort,
search rank, holdout, certification, evaluator margin, and support fields are fail-closed
from model input. Detailed outcomes remain labels/metadata. BonsaiEditFormer was not
trained; its intended Python 3.13/JAX environment passed the 15-test compatibility suite.

## Exactness And Regression Evidence

- CUDA target rebuilt for native `sm_86`; `cuobjdump --list-elf` reports only two
  `sm_86.cubin` images.
- Empty B4EC/B4IM: 4/4 payload rows bit-exact, with zero changes, activations, or faults.
- Active CPU C++/CUDA: 16/16 exact money, flags, trace hashes, and activation counts; 5,947
  action changes and 11,504 activations.
- Distilled ordinary BKG4/CUDA: 16/16 exact for route override, committed route, unit
  rewrite, and market intent.
- Native invariant semantics and overlay harnesses compiled and passed.
- edit-search suite: 246 passed.
- focused invariant-corpus suite: 16 passed (also included in the 246-test search suite).
- Bonsai/BKG integration suite: 150 passed.
- BonsaiEditFormer compatibility suite: 15 passed in its intended environment.
- Modified Python modules compile, the final corpus validates, and the structural ranker
  split has no leakage.

## Prior Equal-Budget Baseline

The earlier same-budget 64-child exact-CUDA benchmark remains useful context. Held-out
safe-useful counts were zero for random typed legacy, fixed systematic legacy, and permanent
ModuleGenome, versus four for the seed B4IM system. The earlier seed corpus had only 128
programs, 41 skeletons, and two positive skeletons. The final specialist factory expands
that system to 12,048 screened hypotheses and 975 fully labeled skeletons while changing
the objective correctly from universal safety to explicit specialist applicability.

## Verdict Boundary

The permanent generator is complete for specialist hypothesis generation, exact screening,
recursive refinement, boundary learning, corpus production, and ranker data preparation.
The three-rule/five-guard B4IM1 bound is intentional, and final multi-specialist tree
assembly plus BonsaiEditFormer training remain separate downstream systems. No generated
specialist is promoted by this validation.
