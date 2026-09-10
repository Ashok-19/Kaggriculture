# Phase 0/1: Replay-Seeded Versus Uninformed Production-Route Search

**PHASE 1 FAIL - PRODUCTION ROUTE PATH NOT VALIDATED.**

Neither arm produced a material fresh production improvement over Shop Plus, and
the replay prior added nothing: the two arms' frozen winners are the **same
compiled route pack**. Phase 2 was not implemented. Nothing was promoted; the
frozen reference agent and `assets/champion/` are unchanged.

Team: ReCurSiON. Starting HEAD `bd6a5ca5758e317f88891270d5b75ce61181ad3f`, clean
`main`. The final commit is the commit containing this report.

## Reference identity (verified, unmodified)

| Property | Value |
| --- | --- |
| Agent | `shop_router_0909/plus` |
| Controller nodes / blackboard / node state | **85 / 2 / 0** |
| Controller bytes | **3,224** |
| Controller SHA-256 | `84bbfe9a32ccf4d202664500b28bba787dab18cb18180c539d70f1890b1cc450` |
| Controller semantic hash | `2207e9f02c80e3b5971745426897a78cdfad16abf4d7694b716dfbc9613dd03b` |
| Route pack | **13 x 719** records, SHA-256 `f711680c...eebe696` |
| Arity table | SHA-256 `5e830564...bb882af` |

A candidate is an alternative route pack read **only by the searching seat**. The
frozen executor, controller and opponent assets are byte-identical throughout, so
a zero-edit candidate is the reference by construction and was verified as such.

## Phase 0 - experiment contract

| Check | Result |
| --- | ---: |
| Zero-edit pack bytes identical to the frozen pack | yes |
| Zero-edit action/state semantic mismatch | **0** |
| CPU/CUDA mismatch, zero-edit panel (8 games x 2 backends) | **0** |
| CPU/CUDA mismatch, candidate route semantics (8 games x 2 backends) | **0** |
| Official-engine canary (kaggle-environments 1.32.7), reference and candidate | 16 games, **0** mismatches |
| Mined replay lifecycles versus current engine biology | 12 checked, **0** mismatches |
| Phase-0 native work | 45 FGE (cap 64) |

Job mining does not use a re-implemented simulator. The host evaluator emits an
exact per-step annotation stream (worker positions, tile kind/what/animal/yield,
seeds, shed, produced) for thirteen route-broadcast reference games, and the
compiler links purchases, pickups, placements, structures and service actions
from that stream.

## Replay source

`ashok205/kaggriculture-top10-replay-archive`, public **version 19**, reusing the
existing index in `workspace/research/top10_replay_archive/evidence.json`. No
full-corpus decode.

| Quantity | Value |
| --- | ---: |
| Current-configuration eligible sampled episodes (1.32.7, sell interval 24) | 211 |
| Rejected as old configuration | 137 |
| Replay bodies inspected | **64** |
| Founder economic profiles | **8** |
| Auditable production-job records | **24** |
| Nonduplicate usable templates | **24** (12 animal, 12 crop) |

Templates cover COW, SHEEP and GOOSE placements and WHEAT, CARROT, STRAWBERRY and
MELON plantings across day 6-26 entries. Selection is outcome-blind and balanced
by day; wins and losses both contribute. Episode ids, team names, replay seeds,
opponent-private state and future observations are offline provenance only and are
never runtime features - the deployed object is a route pack, which observes
nothing.

## Shared candidate representation

Both arms use the same job grammar, compiler, route-compatibility rules,
parameter ranges, execution path, CUDA evaluator and candidate complexity
(90 single-operation and 38 two-operation slots per arm). Only the proposal
prior differs.

Operations are dependency-closed and carried through action 718:

- `ANIMAL_CONVERT` rewrites the structure build when the species needs a
  different one, the `BUY_ANIMAL` financing order, the `PICKUP`, the `PLACE`, and
  proportionally remaps the planned product sales (MILK/WOOL/EGG). Feed, care and
  harvest slots stay with the changed animal.
- `CROP_CONVERT` rewrites the `BUY_SEED` financing, the `PLANT`, the service
  cadence for the new crop's yield window, replants a finite crop that clears its
  tile, converts a redundant replant into a harvest for a perennial, replaces the
  lost WHEAT feed with a `BUY_PRODUCT` order, and remaps planned crop sales.
- Either may be retimed by at most one day into an idle slot of the same lane
  that stands on the required tile; otherwise the proposal is rejected, never
  silently repaired.

| Arm | Class composition | Family split | Valid candidates | Compilation attempts | Unique closures |
| --- | --- | --- | ---: | ---: | ---: |
| R (replay-informed) | 64 template, 38 recombination, 26 uninformed | 64 animal / 64 crop | 128 | 465 (cap 512) | 128 |
| U (uninformed control) | identical structure, no prior | 64 animal / 64 crop | 128 | 451 (cap 512) | 128 |

102 of 128 Arm R proposals used the prior; **26 (20.3%)** deliberately ignored it.
Rejections were compiler-legitimate: no compatible one-day slot, retimed
placement preceding its pickup, and duplicate compiled closures.

**Key diagnostic: 52 of 128 compiled closures (40.6%) are identical between the
two arms.** The replay prior reweighted the same small support rather than
reaching a different one.

## Panels

| Panel | Seeds | Seats | Opponents | Games |
| --- | ---: | --- | ---: | ---: |
| Discovery roots (checkpoint 144) | 16, one opponent each | both | 4 | 32 roots |
| Discovery screening per arm | - | - | - | 128 x 32 = **4,096** continuations |
| BUILD confirmation | 32 fresh | both | 4 | 256 per candidate, 2 candidates per arm |
| INTERNAL_DEV | 64 fresh | both | 4 | 512 per frozen winner |
| Whole-job ablation | 16 DEV seeds | both | 4 | 128 per variant |
| External transfer (official engine) | 32 fresh | both | 4 | **512** |

Opponents are live native executors: Shop Plus, the direct Shop port, and Shop
Plus on route-3 and route-12 broadcast packs. Routes 3 and 12 are byte-identical
to route 0 before step 144, so those challengers share the legal opening and then
execute a different production schedule live. No recorded opponent suffixes.

## Discovery

| Arm | Candidates improving cohort win utility | Candidates with positive median paired margin | Best utility change | Best median margin | Mean utility change | Distinct production profiles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | **0 / 128** | 4 / 128 | 0.00 pt | +656.5 | -28.74 pt | 108 |
| U | **0 / 128** | 4 / 128 | 0.00 pt | +656.5 | -27.56 pt | 112 |

8,192 full-horizon CUDA continuations, steps 144-718 inclusive, both agents live.

## Confirmation and INTERNAL_DEV

| Arm | BUILD finalists | DEV winner | DEV utility change | 90% seed-clustered lower bound | DEV median paired margin |
| --- | --- | --- | ---: | ---: | ---: |
| R | `cand_R_024`, `cand_R_108` | `cand_R_024` | **-16.41 pt** | -20.31 pt | -343 |
| U | `cand_U_103`, `cand_U_001` | `cand_U_103` | **-16.41 pt** | -20.31 pt | -343 |

`cand_R_024` and `cand_U_103` are **byte-identical route packs** (closure
`67f91e56...`): convert the fourth post-144 cow acquisition to a sheep, with
purchase, pickup, placement and MILK/WOOL sale obligations updated. Arm R reached
it from a replay template, Arm U from an uninformed draw.

DEV transitions (512 games each arm): L->W 8, T->W 38, W->T 0, W->L 68, T->L 86.

Per opponent (utility points): Shop Plus -19.53, direct Shop -50.78,
route-3 +4.69, route-12 0.00. Per seat: seat 0 -16.80, seat 1 -16.02.

**Arm R minus Arm U: 0.00 points**, 90% lower bound 0.00. The difference is
exactly zero because the frozen winners are the same object, not because the
estimate is noisy.

## Whole-job ablation

The full candidate against a sales-only variant that applies the same output
obligations without the changed production commitment (128 games each):

| Quantity | Value |
| --- | ---: |
| Full candidate utility change | -20.31 pt |
| Sales-only variant utility change | -35.94 pt |
| Production/labour effect | **+15.63 pt** |
| Median paired margin, full candidate | **-801** |
| Seat support (median margin) | seat 0 -801, seat 1 -801 |
| Opponent support (median margin) | -842.5 / -814.5 / -1007.0 / -168.5 |

The mechanism is genuinely productive rather than terminal - changing the herd
moves the economy by 15.6 utility points relative to relabelling sales alone -
but the direction is negative. The required `>= +1,000` median margin fails in
both seats and against all four opponents.

## External transfer

The named external policies other than Gold Plus V8 were not available locally
and `external/` is empty. Four frozen, structurally comparable local policies
were declared **before** any candidate outcome was observed, and the panel is
reported as a live transfer check, **not** four independent external families:
Gold Plus V8 (named in the plan), Gold V25, the authoritative BKG4 champion, and
the direct Shop reference port. 512 official-engine games (768 planned; both arm
winners are byte-identical, so one subject pack covers both arms).

| External policy | Games | Paired utility change | Median paired margin | Seat 0 / seat 1 |
| --- | ---: | ---: | ---: | --- |
| BKG4 champion | 64 | 0.00 pt | -657.5 | 0.00 / 0.00 |
| Gold Plus V8 | 64 | -15.63 pt | -1,058.0 | -15.63 / -15.63 |
| Gold V25 | 64 | -15.63 pt | -1,077.5 | -15.63 / -15.63 |
| Direct Shop port | 64 | **-54.69 pt** | -673.0 | -53.13 / -56.25 |

Mean paired utility change **-21.48 pt**; worst cohort regression **-54.69 pt**,
far past the -10 point veto.

## Predeclared gates

| Gate | Required | Observed | Result |
| --- | --- | --- | --- |
| Absolute production | Arm R >= +5 pt fresh DEV utility, 90% lower bound > 0 | -16.41 pt, lower bound -20.31 | **FAIL** |
| Replay prior increment | Arm R >= Arm U + 3 pt | +0.00 pt | **FAIL** |
| Material economic change | median paired margin >= +1,000, both seats, >= 2 opponents | -801 | **FAIL** |
| External safety | mean >= 0, no cohort worse than -10 pt | mean -21.48, worst -54.69 | **FAIL** |
| Correctness | zero mismatches, faults, hidden-information use | all zero | **PASS** |

## Work

| Quantity | Value |
| --- | ---: |
| Simulated turns | 7,365,667 |
| Native full-game equivalents | **10,244.3** (hard cap 12,000) |
| GPU turns / wall seconds | 7,327,560 / 21.3 s |
| Native CPU turns / wall seconds | 38,107 / 0.8 s |
| Official-engine games | 528 (16 canary + 512 transfer), 394 s wall on 8 workers |
| CPU/CUDA semantic mismatches | 0 |
| Official/native action mismatches | 0 |
| Invalid runtime states / faults / timeouts | 0 |
| Hidden-information dependencies | 0 |

## Interpretation

The pilot answers its question cheaply and negatively.

1. Structured production-route search over this job grammar did not find a fresh
   improvement. Not one of 256 dependency-closed candidates improved cohort win
   utility on the discovery panel.
2. The replay prior added no value. It reweighted the same support, produced
   40.6% identical closures, and converged on the same winner as the control.
   This is an ablation result, not evidence that uninformed search won.
3. The failure is representational, not mechanical. The compiler is exact
   (zero-edit, CPU/CUDA and official parity all clean), the ablation shows the
   edits really move production, and the candidates are economically diverse
   (108-112 distinct production profiles). The inherited Shop schedule is simply
   close to a local optimum for single-lifecycle substitutions of this shape.

**Do not** scale to Phase 2, enlarge the miner, add a neural proposal model, or
widen the budget on this result. Per the plan's kill criteria, the remaining
bounded option is the small ordinary-mechanics executor improvement, which was
not started here.

### Limitations

- Four native discovery opponents share Shop ancestry; they are not four
  independent strategic families.
- The external panel substitutes frozen local policies for unavailable named
  public agents, so it is a transfer check rather than a HOLDOUT.
- One fixed, non-adaptive proposal generation does not measure the ceiling of
  production-route search; it measures this grammar at this budget.
- Job mining uses a single annotated reference game per route. Live weed
  interruptions can displace a lane, which the executor's own repair queue
  absorbs exactly as it does for the inherited tape.

## Files

`manifest.json` pinned identities and frozen configuration; `replay_jobs.json`
selection, profiles, records, templates and rejections; `fidelity.json` and
`parity.json` Phase-0 and candidate-semantics checks; `official.json` engine
canary; `proposal_schedule.json` both frozen arms; `discovery.json`,
`confirmation.json`, `dev.json` panels; `ablation.json` whole-job ablation;
`external.json` transfer panel; `ledger.json` all charged native work;
`summary.json` gates and verdict.
