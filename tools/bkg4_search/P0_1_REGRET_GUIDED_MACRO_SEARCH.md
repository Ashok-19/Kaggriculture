# P0.1 REGRET-GUIDED MACRO SEARCH VALIDATED

Date: 2026-08-25  
Team: ReCurSiON  
Experiment: P0.1 — regret-guided counterfactual CUDA macro search

The P0.1 hypothesis is validated: bounded deeper CUDA counterfactual planning discovered a coherent post-reveal continuation that flips an authoritative U/W/M-unflippable hard loss and shows recurring positive transfer on independent legal-state cohorts. The campaign also exposed a separate search-control limitation where an already-existing BKG4 route can flip a hard case when searched outside the old ORACLE-M checkpoint/horizon grid.

This is capability discovery only. No champion, submission, corpus, EditFormer, or automatic composition change is promoted here.

## A. Repository state

- Start branch: `main`.
- Start HEAD: `a013d639e92ec2ab24ace427ac552ef73e329261`.
- Strongest reproducible parent used: 345-node `w33_s172_expansion_v1`, semantic hash `8c5d7cd0e32574cbe349d5964946a9e2d6e7a75d766708f5d82308cd8507a034`, combined identity `6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925`.
- Starting worktree: five pre-existing untracked Aug-25 files only: `b4im-elite-corpus-dual-t4.log`, the bakery/pizza/brunch module and route JSONs, and their distillation script/test. They were not changed or deleted.
- Final commit is intentionally recorded by git history rather than self-referenced inside this file. The final task report should use `git log -1 --format=%H -- tools/bkg4_search/P0_1_REGRET_GUIDED_MACRO_SEARCH.md`.

## B. Exact hard target set

The authoritative set comes from `tools/bkg4_search/TEMPORAL_INFORMATION_STATE_TERMINAL.json` and `scratch/bkg4_search/optimization_oracle/screen_cases.jsonl`: the ten canary losses in the terminal `NONE` U/W/M cluster. Each had no tested ORACLE-U utility/market, ORACLE-W worker/task, or ORACLE-M macro-route alternative that produced L→W.

The exact cases were recovered from the persisted tape/seed metadata and re-evaluated against the current 345-node parent before search:

| Oracle case | Team | Seat | Tape | Current margin | U/W/M reason |
|---:|---|---:|---|---:|---|
| 2 | Ryo Hasegawa | 0 | `Ryo_Hasegawa_s1_ep94453800.json` | -10,428 | terminal NONE cluster; no U/W/M L→W |
| 3 | Ryo Hasegawa | 1 | `Ryo_Hasegawa_s1_ep94453800.json` | -9,043 | terminal NONE cluster; no U/W/M L→W |
| 6 | Ryo Hasegawa | 0 | `Ryo_Hasegawa_s1_ep94458182.json` | -10,666 | terminal NONE cluster; no U/W/M L→W |
| 7 | Ryo Hasegawa | 1 | `Ryo_Hasegawa_s1_ep94458182.json` | -10,588 | terminal NONE cluster; no U/W/M L→W |
| 9 | Ryo Hasegawa | 1 | `Ryo_Hasegawa_s1_ep94524402.json` | -5,443 | terminal NONE cluster; no U/W/M L→W |
| 12 | Arman Tuganbaev | 0 | `Arman_Tuganbaev_s0_ep94541153.json` | -3,908 | terminal NONE cluster; no U/W/M L→W |
| 13 | Arman Tuganbaev | 1 | `Arman_Tuganbaev_s0_ep94541153.json` | -3,908 | terminal NONE cluster; no U/W/M L→W |
| 18 | Arman Tuganbaev | 0 | `Arman_Tuganbaev_s0_ep94507639.json` | -15,530 | terminal NONE cluster; no U/W/M L→W |
| 19 | Arman Tuganbaev | 1 | `Arman_Tuganbaev_s0_ep94507639.json` | -15,665 | terminal NONE cluster; no U/W/M L→W |
| 20 | ReCurSiON | 0 | `ReCurSiON_s0_ep94324507.json` | -49,506 | terminal NONE cluster; no U/W/M L→W |

The first five Ryo references improve only +94…+98 under the later 345-node parent; none flips. The remaining five margins are unchanged. The historical hard-set classification therefore remains meaningful.

## C. Existing infrastructure reused

No universal planner was built. The experiment reused:

- `cuda-engine/src/bkg4_fragment_eval.cu`: exact CUDA live-BKG4 rollout, bounded donor-fragment substitution, then automatic return to current BKG4;
- `cuda-engine/src/bkg4_information_state_capture.cu`: complete legal observation + BKG4 runtime-state capture for initiation-state ambiguity checks;
- the current champion controller, route pack, legacy compatibility layer and mechanics reconcilers;
- the Aug-24 source-backed top-10 parent-route archive (routes 400–437);
- existing current BKG4 routes and public hybrid routes 300–310;
- persisted optimization-oracle screen tapes/seeds and public-feature artifacts.

The route435 donor is source-backed: rank-10 `カワシギ`, episode `97637527`, source seat 0, seed `877972500`, action hash `13cb3f29b7139832c96f0238e4163ce91683f8feec46381f27e3e91ad49282b4`. Its archived route had zero clipped hand or market actions.

## D. Minimal implementation added

Permanent implementation is deliberately small:

- `tools/bkg4_search/cuda_io.py`: adds the existing `B4FR1` fragment-result ABI parser.
- `tools/bkg4_search/regret_macro_search.py`: small host helper to summarize candidate-vs-baseline regret and sanitize native tape JSON metadata for the CUDA parser.
- `tools/bkg4_search/test_regret_macro_search.py`: focused deterministic parser/summarizer/sanitizer tests.
- `cuda-engine/Makefile`: exposes the already-existing `bkg4_fragment_eval` and `bkg4_information_state_capture` binaries as normal build targets.
- this terminal report and a compact append to `EXPERIMENT_LEDGER.md`.

No policy language, controller, route pack, corpus generator, EditFormer, or submission code was changed.

## E. Search design

### Checkpoints

The first coarse pass used strategic day/decision windows at steps:

`96,120,144,168,192,216,240,264,288,336,384,432,480,528,576,624,672`.

Refinement used 12-step spacing around promising early/midgame regions. After evidence showed that several losses were already strategically committed by step96, one bounded correction added starts `0,24,48,72`.

### Horizons

Initial horizons were exactly the research recommendation: `24,48,72`. Refinement added `36,60,84,96,120,144`, and the early-checkpoint diagnostic included up to 192 turns.

### Macro vocabulary

Candidates were coherent route fragments, not random per-turn strings:

1. 38 source-backed top-10 continuations, routes 400–437;
2. current BKG4 routes including 22/23/29/32/34/46/63/64/70/77/79/211–214;
3. public hybrid routes 300–310;
4. two bounded two-stage compositions for evidence-backed Arman basins.

### Search algorithm

The search was hierarchical enumeration + evidence-guided refinement:

coarse donor macro enumeration → exact CUDA evaluation → inspect per-case regret → refine promising checkpoint/horizon basins → limited two-stage composition → full-screen transfer → complete legal-state ambiguity check → component ablation.

### Rollout policy

Before and after each searched window, the live current 345-node BKG4 remained the policy. The experiment never replaced the full-game controller.

## F. Search scale

Completed macro evaluations total **61,787 exact CUDA games**. Two complete legal-state capture campaigns added 122 CUDA trajectories.

| Campaign | Exact CUDA games |
|---|---:|
| current-parent baseline | 10 |
| coarse 38-donor search | 19,390 |
| refined donor search | 12,160 |
| two-stage composition A | 1,530 |
| two-stage composition B | 1,530 |
| public-hybrid grid | 5,620 |
| early all-route grid | 20,490 |
| step48 full-screen transfer | 305 |
| step48 ablation | 48 |
| window minimization | 306 |
| three post-reveal transfer runs | 366 |
| route435 component ablation | 32 |
| **Total** | **61,787** |

Measured CUDA kernel time for those macro evaluations was approximately **16.45 s**. The two complete state captures used another approximately **0.98 s** of GPU kernel time. Saturated macro-search batches reached roughly 4,100–5,300 games/s on the available RTX 3050 Laptop GPU. Small transfer/ablation batches were intentionally underfilled and are not representative throughput measures.

Every accepted CUDA invocation reported `bad=0`; no game/runtime fault entered the evidence.

## G. Results per hard loss

| Hard case | Baseline | Best coherent result | L→W? | Delta | Macro summary |
|---:|---:|---:|---|---:|---|
| 2 | -10,428 | -3,537 | No | +6,891 | existing route79, early long continuation |
| 3 | -9,043 | +581 | **Yes** | +9,624 | route419, step96 ×48 |
| 6 | -10,666 | -5,859 | No | +4,807 | route424 early refinement |
| 7 | -10,588 | -5,859 | No | +4,729 | route424 early refinement |
| 9 | -5,443 | **+6,136** | **Yes** | **+11,579** | route435, step156 ×48 |
| 12 | -3,908 | -535 | No | +3,373 | route421 midgame continuation |
| 13 | -3,908 | -535 | No | +3,373 | route421 midgame continuation |
| 18 | -15,530 | +9,557 hindsight | **Rejected** | +25,087 | step48 macro; future-dependent initiation cell |
| 19 | -15,665 | +10,218 hindsight | **Rejected** | +25,883 | step48 macro; future-dependent initiation cell |
| 20 | -49,506 | -5,930 hindsight | No | +43,576 | long early public-hybrid continuation |

Cases 18/19 are deliberately not counted as valid P0.1 L→W discoveries despite their terminal wins because the exact initiation-state test demonstrated indistinguishable legal states with opposite future effects.

## H. New capabilities discovered

### Valid capability: shop-conditioned production rebalance

The strongest valid macro is route435 applied at step156 for 48 turns, then returning to BKG4.

Relative to incumbent routes 63/64 over that window, route435 changes 32/48 route steps and coordinates both market and worker behavior. Mechanically/economically it:

- reduces strawberry seed commitment sharply (12 planned strawberry seeds in the incumbent window versus 4 in route435);
- delays one sheep acquisition instead of buying both at the first expansion transaction;
- reallocates wheat procurement across the production cycle;
- roughly doubles PLANT actions in the window (12 versus 6);
- reduces worker PASS/idle actions (26 versus 43);
- increases active watering/production routing while keeping the same five-worker labor base at initiation.

This is therefore a **co-adapted production/worker/market rebalance**, not a single market-order exploit.

Component ablation on eight supported cases:

| Variant | Positive | Harmful | L→W |
|---|---:|---:|---:|
| full route435 | **8/8** | **0/8** | **3** |
| route435 market + incumbent units | 5/8 | 3/8 | 2 |
| route435 units + incumbent market | 4/8 | 4/8 | 2 |

Both halves contain useful information, but only the coordinated continuation transfers without harm across this supported set.

### Is it new relative to B4IM?

**Yes at the current representation boundary.** Current B4IM local edits/route overrides do not contain this external route435 continuation as an executable bounded macro. Expressing it cleanly requires adding a coherent macro/subtree/route-fragment object rather than dozens of independent local action edits. Once such a macro object exists, ordinary public-state predicates can later gate it.

## I. Transfer/generalization

Ungated route435 at step156 ×48 was evaluated across the full 61-case optimization screen. Global application is unsafe: 12 positive, 2 neutral, 47 harmful, with 15 W→L. P0.1 therefore does **not** propose an unconditional patch.

The important result is that effects recur in already-visible public shop regimes:

| First two visible shops | Cases | Positive | Harmful | L→W | Aggregate delta |
|---|---:|---:|---:|---:|---:|
| PIZZA → PIZZA | 2 | **2** | **0** | **2** | **+22,387** |
| PET → PIZZA | 2 | **2** | **0** | **1** | **+19,590** |
| PET → FARMERS_MARKET | 2 | **2** | **0** | 0 | **+27,974** |
| BRUNCH → PET | 2 | **2** | **0** | 0 | **+7,514** |

The authoritative hard-target success is case9:

`-5,443 → +6,136`, delta `+11,579`.

Its paired PIZZA→PIZZA state, case8, independently changes:

`-5,178 → +5,630`, delta `+10,808`.

A separate PET→PIZZA trajectory, case16, changes:

`-3,204 → +9,033`, delta `+12,237`.

The paired PET→PIZZA case17 improves from `-7,518` to `-165`.

Both seats are represented in the PIZZA→PIZZA pair. The effect also recurs in additional two-seat non-flip cohorts. These are legal/public state distinctions available before the route435 window; no opponent identity is used.

A simple scalar-only gate was tested and rejected because it selected harmful states too. Full initiation-set synthesis is intentionally deferred to the next stage.

## J. Rejected discoveries

### Step48 five-worker launch macro

The largest raw hindsight result was an early 12–24-turn macro around steps48–71. It could flip Arman cases18/19 in both seats by roughly +25k and also flipped other Ryo losses.

Ablation showed the effect required a coordinated five-worker/market launch rather than either market-only or units-only edits. However, complete CUDA legal-state capture at step48 found only nine distinct legal-state cells across the 61-case bank, and the support cells were mixed-sign: identical complete legal observation + BKG4 runtime state led to both large gains and large regressions depending on unseen future shop realization.

**Decision: reject as hindsight-only/future-dependent.** It is not counted toward validation.

### Other rejected branches

- route421 two-stage capital→production compositions improved deep Arman deficits by up to roughly +9.2k but produced no new L→W;
- public hybrids 300–310 produced no hard-set L→W;
- route79/route419 post-reveal flips were globally harmful without a selector and therefore are discovery evidence only, not ship candidates.

## K. Comparison with old edit manifold

The evidence supports **both** a search-control and a representation bottleneck.

### Search-control bottleneck

Hard case3 can be flipped with **existing route79**:

`-9,043 → +344`.

The useful continuation starts at step72 and runs 168 turns. The old ORACLE-M grid used a narrower checkpoint/horizon family, so the route existed but the search did not expose this continuation. This is a concrete search-control/horizon failure rather than a representation failure.

### Representation bottleneck

The route435 production rebalance is external to the current BKG4 route set and requires a coordinated bounded worker+market program. Current local B4IM edits do not cleanly represent it as one reusable executable object. This is a concrete macro/subtree representation gap.

## L. Research hypothesis verdict

**YES.**

Deeper counterfactual planning can discover genuinely new winning capability that the current BKG4/B4IM local search does not expose.

The proof does not rely on the rejected clairvoyant step48 wins. The valid evidence is the post-reveal route435 continuation: it flips an authoritative U/W/M-unflippable hard case, independently flips its paired PIZZA→PIZZA state, flips another legal PET→PIZZA state, and produces repeated positive effects in several other public cohorts.

The result also demonstrates that planning is a teacher, not a safe selector: the same macro is disastrous in unrelated public regimes.

## M. Next synthesizer requirement

The smallest next feature should be:

1. **bounded macro/subtree representation** capable of storing a coherent route fragment such as route435 steps156–203 as one executable program object;
2. **initiation-set CEGIS** over legal public state, especially visible shop regime plus relevant capital/labor/production features.

Do not implement general automatic composition, recovery synthesis, or EditFormer retraining yet. First prove that the new macro representation plus CEGIS can retain the positive route435 cohorts while excluding the harmful ones.

## N. Tests

Experimental validation completed:

- exact current-parent baseline on all 10 hard cases;
- 61,787 exact CUDA macro games;
- two complete CUDA legal-state capture campaigns;
- full-screen transfer for the key early and post-reveal candidates;
- route435 component ablation;
- all accepted CUDA runs `bad=0`;
- no official Python engine used;
- no large CPU grid used.

Permanent tooling validation is provided by `tools/bkg4_search/test_regret_macro_search.py` plus Makefile builds for the two existing CUDA binaries.

## O. Commit

This file cannot contain the hash of the commit that contains itself without a self-reference cycle. The local commit hash is therefore reported in the task-completion response and is recoverable with:

`git log -1 --format=%H -- tools/bkg4_search/P0_1_REGRET_GUIDED_MACRO_SEARCH.md`

No push or Kaggle submission is part of P0.1.

## P. Final recommendation

**Proceed to the synthesizer stage**, but only after treating P0.1 as capability discovery rather than a promotion. The experiment establishes that deeper CUDA planning can uncover recurring coherent continuations beyond the current local B4IM edit manifold, while the exact-state rejection of the step48 macro demonstrates why initiation-set synthesis is mandatory. The next work should be the smallest bounded macro/subtree representation plus public-state CEGIS needed to turn route435-like discoveries into safe Bonsai specialists.
