# Continue Kaggriculture — use this as the entry prompt in the new ChatGPT session

Continue the Kaggriculture Kaggle competition project from the **repository-grounded canonical handoff**. Do not reconstruct the project from memory, assumptions, stale leaderboard data, previous assistant summaries, or unverified public claims.

Use `Local_mcp` with repository id:

```text
kaggriculture
```

Repository root:

```text
/home/nnmax/Desktop/kaggle/kaggriculture
```

## Mandatory first reads

Read these **fully and in order** before proposing or implementing the next experiment:

1. `.chatgpt/handoffs/current.local.md`
2. `.chatgpt/handoffs/KAGGRICULTURE_MASTER_PROMPT.local.md`
3. `.chatgpt/handoffs/2026-08-14-0103-kaggriculture-q8-live-meta-market-restructuring-continuation.local.md`
4. `.chatgpt/handoffs/2026-08-12-2204-kaggriculture-evidence-first-gold-meta-continuation.local.md`
5. `RULES.md`
6. `AGENTS.md`
7. `README.md`
8. `agent_system/README.md`
9. this `continue_prompt.md`

The master prompt intentionally contains an older historical body below a new 2026-08-14 authoritative update. Preserve both: the new top section is current operational state; the older body contains important mechanics, rejected experiments, hidden-future results and provenance that must not be forgotten.

## Verify current state before any research

Use read-only Git/Kaggle checks first:

```text
repo_git_status(repo_id="kaggriculture")
```

and:

```bash
git log --oneline --decorate -12
git branch -vv
git remote -v
```

Immediately before the continuation-prompt commit, the verified baseline was:

```text
branch: main
HEAD: 6b53de3284dfa218a6217045d8034833f5c6c11d
origin/main: 6b53de3284dfa218a6217045d8034833f5c6c11d
status: clean
```

`continue_prompt.md` itself was then committed locally, so the final handoff HEAD is necessarily newer than `6b53de3`. **Do not infer that final SHA from this file; run `repo_git_status` / `git log` at startup.** No push was requested, so also verify whether local `main` is ahead of `origin/main`.

Recent important commits:

```text
6b53de3 Ignore local Kaggriculture research artifacts
595fbc6 Promote q8 fertilizer minimax successor
4f99359 Add evidence-first gold medal rules
7934ce7 Promote minimax meta successor
11b5245 Fix arena module isolation
```

Do not trust these blindly if the repository changed. Verify.

Then refresh current Kaggle truth:

- all ReCurSiON submissions and current scores;
- leaderboard top 20;
- current public-safe submission ids for Jason, Kawashigi, researchstudio, Mohamed, Furious Monk, JALKARNA, Yusuke, Arda and other top agents;
- current episode counts/replay availability;
- whether a newer top-agent submission has appeared since this handoff.

At the handoff snapshot, the latest submitted ReCurSiON agent was:

```text
submission id: 55480892
lineage: q8_fert_v601
production commit: 595fbc677db3a377885fe7ef6e2983b49d3d0c33
archive SHA-256: 08c176d10fe077710860dc3d48155cddf1765d6b8563f41801ff10f72bd269bb
archive size: 77,222 bytes
status: COMPLETE
public score at snapshot: 2213.3
```

Older `55462815` was 2982.8 at the same snapshot. These values are mutable.

Leaderboard snapshot at handoff, historical only:

```text
1  Kawashigi              3235.4
2  researchstudio.site    3146.2
3  jasonstillchasin       3145.3
4  JALKARNA GAUTAM        3141.2
5  Mohamed abdelrazik     3118.2
6  Furious Monk           3116.0
7  Yusuke Hayashi         3103.9
8  MD. Nazmus Sakib Anik  3097.0
9  Arda Ceylan            3091.3
10 Suda                   3082.0
```

Refresh before using.

## Mission

The objective is explicit:

> **Finish top 10, earn Kaggle gold, and maximize prize-money probability.**

The user does not want another cosmetic successor that wins self-play by +50 or produces only a modest live improvement. The current hard-meta target is to push the dominant current-top league toward **128/128 wins**, while remaining generalized and causal.

Do **not** fake that target by overfitting. A real winner must be strong across multiple current public agents, both seats, fresh seeds, historical opponents and held-out replays.

## Required four-lane research style

Continue the Kaito-Fukami-inspired four-lane approach in `RULES.md`:

### Lane 1 — mechanics / invariants / seat asymmetry

Use exact failure replays and official-engine state to repair genuine mechanics/logistics problems. Weed, worker, shed, capacity, debt, market-order and day-boundary invariants belong here.

### Lane 2 — elite current-top adversaries

Jason, Mohamed, Kawashigi, researchstudio, JALKARNA, Furious, Yusuke, Arda and any newly emerging top agents are first-class adversaries/rule sources.

Download replays whenever useful. Build populations, not one-tape mythology. Keep newest distinct streams as holdouts where practical.

### Lane 3 — historical / worst-of-population robustness

Keep current submitted q8, previous ReCurSiON packages, WHEAT14 families, rank1/rank7 families, Kaito/core, Aug-11 exact-current tapes and the historical league in promotion gates.

Evaluate worst opponent per fixture, both seats.

### Lane 4 — live meta / architecture discovery

Do not remain trapped in q8 micro-patches. Maintain a structural lane: market controller, production portfolio, causal selector, bounded search, alternative economic architecture, or other coherent meta change.

The user explicitly said that **50/128 hard-league wins are not enough** and expects a meta restructuring if necessary.

## Strict evidence rules

Treat these as non-negotiable:

- repository + pinned official engine + exact local simulations + verified Kaggle replay output are source of truth;
- no opponent/team/submission id in policy;
- no episode id or seed logic;
- no future shops or hidden RNG;
- no fake classifier when observations are identical at the decision point;
- replay tape = realized trajectory, not hidden executable policy;
- both seats always;
- fresh disjoint seed blocks;
- failed/timed-out/safety-blocked wrapper = **no policy conclusion**;
- geometry changes can change future weed/shop RNG;
- market cash changes can change affordability and route;
- accepted tracked source/tool/test changes get small incremental commits;
- rejected scratch stays scratch;
- keep repo clean and delete obsolete tmp files/wrappers when no longer useful;
- preserve exact replays, tapes, useful baselines and negative-result evidence;
- no push unless explicitly requested.

The arena isolation fix at commit `11b5245` is mandatory for monkeypatch/wrapper comparisons.

## Current submitted q8 anchor

The strict future successor gate is against **submission 55480892 / q8_fert_v601**, not the older v29 submission.

q8 opening market basket:

```text
BUY_PRODUCT WHEAT 8
HIRE x4
BUY_ANIMAL COW 1
BUY_ANIMAL SHEEP 4
BUY_SEED MELON 5 requested; 4 execute
BUY_SEED WHEAT 5
```

Turn1 q8-vs-q8 reference:

```text
money 1
hands 4
shed WHEAT8 COW1 SHEEP4
seeds WHEAT5 MELON4
```

Future promotion still requires 100% wins against this exact incumbent on a large paired-seat disjoint range, but that is only the first gate.

## Important validated successor history

The current scratch research is far beyond the submitted q8.

### v5/v6/v7/v8 meta lineage

A q8 farm/worker backbone plus adaptive market meta achieved:

```text
v5 direct q8 gate: 128/128 fresh wins
mean ~+1259
worst game ~+200
```

Historical/core v6:

```text
96/96 nonnegative fixture deltas
```

Aug-11 exact-current population:

```text
52 fixtures
0 negative deltas
```

Exact q8 shadow branches are used for public early families that were not yet safely improved:

```text
5 hands / $17 -> rank1/rank7-style elite
5 hands / $20 -> Arda-like
>=6 hands / >=$600 -> high-capital conservative family
```

Dominant WHEAT14 branch is 5 hands / $0.

### Safe WHEAT14 FERT bundle

Older three-shift bundle was:

```text
302->290
326->312
350->340
```

Later live testing proved `350->340` causes exact -2 against Furious/Mohamed. Current safe bundle keeps only:

```text
302->290
326->312
```

Current live panel: zero negative deltas; beneficial families gain about +84.

### v9 STRAWBERRY terminal controller

Candidate:

```text
scratch/live_top10/meta_routeclean_adaptive576_v9_w14straw/
```

WHEAT14-only bounded sales:

```text
576 @ $1 -> 586
600 @ $1 -> 604
624 @ <=$25 -> 640
648 @ <=$10 -> 668
```

Two fresh 64-fixture WHEAT14 blocks:

```text
128/128 non-regressing vs v8
max activated gain ~+1153
wins 38 -> 50
```

This is useful but **50/128 wins is not the endpoint**.

### v11/v12/v13 restructuring

Broad season-long strawberry debt was rejected due shed capacity. Bounded late strawberry tranches were strong.

v11 consolidated terminal portfolio:

```text
24/24 diagnostic fixture improvements
+801..+2054
```

v12 delays ninth cow purchase from step361 to about505 and buys only if observable MILK quote >$50:

```text
24/24 non-regressing
max +529
wins 8/24 -> 13/24
```

v13 widens bounded step624 STRAWBERRY hold to <=$40:

```text
scratch/live_top10/meta_w14_terminal_v13_straw40/
24/24 non-regressing
mean +428
max +700
wins 13/24 -> 19/24
worst margin -1938 -> -1238
```

v13 is a key robust baseline.

## Mohamed population — do not tune to one tape

Current Mohamed submission used:

```text
55476469
```

Exact replay population:

```text
92699406 vs Furious
92659241 vs researchstudio
92634088 vs Malel
92607971 vs JALKARNA
92616295 vs Filip
```

Files:

```text
scratch/live_top10/mohamed_aug13_expand/
scratch/live_top10/mohamed_aug13_expand/tapes/
```

All exact source rewards reproduce locally.

Fresh paired population result for v13:

```text
9 wins / 31 losses
mean -100.4
floor -4225
v13-v12 delta min0 mean+68.9 max+697
0 negative deltas
```

This is a major current weakness.

Exact market accounting on Mohamed hard seed48420 found the decisive product gap:

```text
WOOL opponent revenue 21202
WOOL candidate revenue 17900
candidate deficit -3302
```

Candidate WHEAT and STRAWBERRY revenue were actually better, so do not misdiagnose the whole problem as farm production.

## Jason is now first-class elite data

Jason submission used:

```text
55470275
```

High-value exact replays already downloaded:

```text
92692927 vs Kawashigi -> Jason wins 91716-86690
92708754 vs researchstudio -> Jason wins 79807-76792
92725401 vs JALKARNA -> Jason wins 69564-68618
92547299 older Jason-vs-Malel tape
```

Files:

```text
scratch/live_top10/jason_aug13_expand/
scratch/live_top10/jason_aug13_expand/tapes/
```

The Kawashigi and research wins share Jason action hash prefix `c3b5b90418606d18`; JALK win uses a second stream `856d6cf4cb5248e5`.

Jason's step240 mature farm matches our backbone closely:

```text
4 sheep
8 cows
13 WHEAT
20 STRAWBERRY
5 MELON
```

Use Jason for **rule mining**, not exact schedule transplant.

Fresh three-stream Jason screen seeds48440..48443 both seats:

```text
v13 18/24 wins
mean +6449
floor -2042
submitted q8 18/24 mean +5053.8 floor -3435
v13-q8 positive on 24/24 by +638..+2745
```

All six v13 losses are the same town seed `48440`, both seats across all three Jason streams.

Hard48440 exact accounting candidate minus Jason:

```text
extra buy cost: +1663
WOOL sell revenue: -1063
WHEAT sell revenue: +861
STRAWBERRY: -318
FERT: -328
MILK: +392
MELON: +69
```

The hard Jason deficit is mostly **late WHEAT overtrading and WOOL opportunity cost**.

Do not copy Jason's WHEAT/WOOL market schedule wholesale: direct transplantation produced catastrophic losses (roughly -20k WHEAT-only, -5k WOOL-only, ~-101k both) because market paths are state-coupled.

## v14 safe WHEAT gate

Candidate:

```text
scratch/live_top10/meta_w14_terminal_v14_wheat46/
```

Rule:

> after step576, suppress `BUY_PRODUCT WHEAT` only while observed WHEAT quote <=$46.

Jason diagnostic result:

```text
24/24 non-regressing
+40..+101 where active
exact tie in high-WHEAT-demand seed48442
```

Safe but small.

## CURRENT FRONTIER — v15 WOOL market-impact controller

This is where the new session must resume.

Official WOOL market curve:

```text
base 200
I0 10000
T 105
below log / target .2
above sq / target 3.2
```

The quadratic glut side makes large WOOL sells highly destructive.

Experimental candidate:

```text
scratch/live_top10/meta_w14_terminal_v15_wool40/
```

Concept:

- inspect planned WHEAT14 WOOL sales;
- calculate predicted marginal WOOL quote from official public market inventory;
- cap units whose marginal quote would fall below ~$40;
- carry unsold WOOL as debt and try to release later.

Jason diagnostic result:

```text
seed48440: +1774..+2010 delta
all three Jason continuations flip positive
final margins roughly +48 / +245 / +1211

seed48441: +5..+9
seed48443: +5

seed48442: catastrophic -21699..-22259 delta
resulting margins roughly -12k..-20k
```

**v15 is unsafe and must not be promoted.**

Its importance is mechanistic: a correct WOOL controller has enough value to flip the hard Jason town by roughly +2k.

### Critical hidden-future caution

At step309:

```text
48440 WOOL price18 inventory10056
shops: PET, ICE, FARMERS, BAKERY
future step360 later reveals YARN

48442 WOOL price18 inventory10056
shops: BAKERY, PIZZA, PIZZA, SMOOTHIE
future step360 reveals ICE, no YARN yet
```

At 311/312 both can be WOOL price1/inventory10059.

The full observations differ, but **future YARN is not known at step309**. Never build a rule that effectively uses future YARN or memorizes a shop prefix as a proxy for random future RNG.

### Immediate research question

Can the +2k WOOL gain be made safe using only current observable economics:

- current marginal WOOL price curve;
- current unlocked demand;
- shed occupancy/capacity;
- carried/physical WOOL;
- next known canonical wool availability;
- bounded debt/tranche size;
- current cash/opportunity cost;
- later observed recovery?

Or is early holding itself a hidden-future minimax conflict and the correct commitment point must be later, after demand actually reveals?

### First actions for the new session

1. Diagnose **why 48442 goes toxic** in v15:
   - record `_WOOL_DEBT` over time;
   - shed occupancy and peak capacity;
   - carried WOOL and sheep `yield_units`;
   - suppressed canonical WOOL sales;
   - actual quote path;
   - release attempts;
   - cash/affordability divergence;
   - first farm/market structural divergence vs v14.
2. Compare 48440 and 48442 at the exact first v15 divergence using full legal observations.
3. Test bounded WOOL tranches / later commitment before any classifier.
4. Run broad fresh activation sampling across Jason + Mohamed + 16-continuation WHEAT14 before choosing thresholds.
5. If two futures are observationally equivalent at commitment, stop classification and solve minimax.

Do **not** simply lower the v15 floor until 48442 stops failing. That would be cosmetic tuning without understanding the mechanism.

## Replay workflow

The system Kaggle CLI is old and lacks simulation commands. Use isolated modern CLI:

```bash
uvx --python 3.11 --from kaggle==2.2.2 kaggle competitions episodes <SUBMISSION_ID>
uvx --python 3.11 --from kaggle==2.2.2 kaggle competitions replay <EPISODE_ID> -p <DIR>
```

Replay mapping:

```text
observation t -> steps[t+1].action
719 real actions
```

Require exact coin-for-coin source parity with `agent_system/tools/official_oracle.py` before accepting a tape.

For Jason specifically, keep downloading high-value replays against current #1/#2/top agents whenever a new distinct action stream appears. The user explicitly authorized this and wants Jason used for both rule mining and promotion gates.

## C++ guidance

C++ has **not** produced a validated policy improvement. Current runtime is already roughly 0.4ms/turn.

Do not rewrite Python merely for speed.

C++ is only justified for a new exact search backend, such as:

- bounded market-order search;
- marginal liquidation optimizer;
- exact reserve/debt solver;
- offline lookup-table generation;
- bounded route-repair search.

Any C++ search must first match official market mechanics exactly, then beat the current policy in paired arena tests. Otherwise it gets no credit.

## Promotion target

Do not submit the current scratch research yet.

A real successor should pass, in order:

1. **Exact latest incumbent gate:** 100% vs `55480892` / q8_fert_v601 on large fresh paired-seat seeds.
2. All accumulated hard mechanics/seat seeds.
3. WHEAT14 16-continuation worst-of population on at least two disjoint blocks.
4. Expanded Jason distinct-stream population.
5. Expanded Mohamed distinct-stream population.
6. rank1/rank7 elite families.
7. Aug-11 exact-current 26-tape population.
8. historical 12-opponent league.
9. Kaito/core and other current-live families.
10. blind replay holdout.
11. exact packaged `submission/build/main.py` parity/gates.

The user wants the hard current-meta league pushed toward **128/128 wins**, not merely 128/128 non-regressing deltas.

If the candidate only ties q8 on a difficult elite family through a shadow branch, treat that as safe but incomplete. Continue looking for a positive causal edge.

## Repo hygiene and commit discipline

Local handoffs and `research/` tooling are intentionally ignored by commit `6b53de3`.

Keep the repo clean:

- remove temporary failed wrappers and obsolete generated files after diagnosis;
- use `workspace_reap_processes` after timed-out arena batches if needed;
- preserve active scratch candidates and verified replays/tapes;
- do not mass-delete scratch evidence;
- do not mass-stage;
- accepted tracked source/tool/test changes get small exact-path commits;
- rejected scratch research does not get committed;
- no Git push unless explicitly requested.

Before a production commit use `repo_git_review`, then commit only intended paths.

Before a Kaggle submission:

```bash
python3 agent_system/tools/build_submission.py
```

Then test the exact `submission/build/main.py`, inspect tar contents, record size + SHA-256, and submit that exact archive.

## Do not repeat these known failures without a materially new premise

- WHEAT14 no-13th-cow simple patch;
- rank7/W14 extra opening MELON;
- `dailyclean` universal cleaner;
- naive full WHEAT14 worker splice;
- full replay market transplant;
- global WHEAT buy suppression;
- generic WHEAT timing/preemption;
- season-long strawberry debt bucket;
- naive 8-cow deletion;
- exact Jason WHEAT/WOOL schedule transplant;
- v15 rolling WOOL debt as currently implemented;
- future-YARN classifier at step309;
- identity/seed selectors;
- C++ rewrite without search/parity evidence.

## Final operating principle

Do not optimize for the appearance of progress.

The strongest current evidence says:

- the submitted q8 is not enough for the live target;
- v13/v14 are robustly stronger but still lose too many current hard fixtures;
- the dominant remaining matched-architecture weakness is **premium WOOL opportunity-cost control**;
- v15 proves that fixing WOOL can add ~+$2k and flip Jason hard48440;
- v15 also proves that naive holding can be catastrophically wrong in another legal future;
- Jason is now one of the most valuable public rule sources and must stay in the elite lane.

Continue through diagnosis, structural market redesign, broad live-population testing and the full four-lane gate. Do not stop because one local panel looks good. Do not submit until the evidence supports a genuinely stronger meta.
