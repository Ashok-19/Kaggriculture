# Continue Kaggriculture — v33 hierarchical lane continuation

Continue the Kaggriculture Kaggle competition project from the **repository-grounded canonical handoff**. Do not reconstruct the project from memory, assumptions, stale leaderboard values, previous assistant summaries, or unverified public claims.

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
3. `.chatgpt/handoffs/2026-08-15-0147-kaggriculture-v33-hierarchical-lane-continuation.local.md`
4. `.chatgpt/handoffs/2026-08-14-1710-kaggriculture-v29-expanded-current-best-continuation.local.md`
5. `.chatgpt/handoffs/2026-08-14-0103-kaggriculture-q8-live-meta-market-restructuring-continuation.local.md`
6. `.chatgpt/handoffs/2026-08-12-2204-kaggriculture-evidence-first-gold-meta-continuation.local.md`
7. `RULES.md`
8. `AGENTS.md`
9. `README.md`
10. `agent_system/README.md`
11. this `continue_prompt.md`

The master prompt intentionally preserves older historical bodies. The **2026-08-15 authoritative section at the top** supersedes stale operational state while the older sections remain valuable negative-result/mechanics/provenance memory.

## Mission

The objective is explicit:

> **Finish top 10, earn Kaggle gold, and maximize prize-money probability.**

The user does not want cosmetic gains or a narrow best response. The research objective is to create a causal current-meta architecture that drives the hardest public population toward 100%/128-of-128-style win conversion while surviving fresh seeds, both seats, historical leagues and blind replay holdouts.

## Hard local-PC compute constraint

This is non-negotiable:

> **Never run more than 2 simulation workers.**

Use `Pool(2)` explicitly or run sequentially. Never use bare `multiprocessing.Pool()`, all cores, or an unbounded heavy `-j`. Split large leagues into small batches. After a timeout/interruption, use `workspace_reap_processes` to verify no orphaned workers remain.

The user's local PC previously crashed from all-core arena jobs. Do not repeat this.

## Most important gold-tier technique: lane-based causal restructuring

The project is no longer trying to recognize individual opponents. The key method is:

> **shared causal prefix + option-value/minimax while futures are indistinguishable + branch at the first legal public divergence + coherent lane-specific continuation + adversarial population gates**

A lane is a population sharing an observable public-state/economic regime. It is not a team name or submission id.

Rules:

- If complete public observations are identical, the policy must remain common.
- Never use seed, team/opponent/submission/episode id, future shops, hidden RNG, or a prefix that merely predicts future RNG.
- Branch only after a public state difference is actually visible before the changed action.
- Prefer coherent actor+market continuation resets when geometry/economy is state-coupled.
- A lane expert must be tested against every available member of the same public-state cohort, both seats and disjoint seeds.
- Add new live agents with the same prefix to the cohort; do not route around them by identity.
- If micro-patches cannot solve a lane robustly, restructure opening economy, production geometry, livestock mix, market controller or full continuation architecture.

Read the new sections in `RULES.md` before research.

## Verify current repository state immediately

Run:

```text
repo_git_status(repo_id="kaggriculture")
```

and verify:

```bash
git log --oneline --decorate -12
git branch -vv
git remote -v
```

At this handoff the final local HEAD is expected to be newer than production v32 because durable rules/continuation docs were committed. Verify the exact SHA; do not trust this prompt blindly.

Production v32 itself was promoted at:

```text
d52526d8122a1848ab364f0f072b5ba650adff18  Promote v32 four-lane agent
```

Durable lane/compute rules were then committed as:

```text
1cbc881f9221b3ff63fd94b944ed5d4bd786f8f5  Document lane method and local compute cap
```

No push was requested during the handoff. `origin/main` was `6b53de3` at verification time. Refresh remote state before saying anything about push status.

The only remaining worktree noise after releasing the v33 task claim was the unrelated untracked notebook:

```text
public-notebooks/read-the-market-choose-the-farm.ipynb
```

Do not stage or modify it unless the user explicitly asks.

## Refresh Kaggle state before strategy conclusions

At the handoff snapshot around 2026-08-15 01:41 IST:

```text
55507122  v32 / commit d52526d  COMPLETE  2706.6
55491717  f120 d4 fix          COMPLETE  2897.3
55462815  old meta minimax     COMPLETE  2975.5
```

Latest v32 archive metadata:

```text
submission: 55507122
size: 212,630 bytes
SHA-256: cd25d9457a3ab3551931d807543e2e526ae2a021f5bf53c90084171cd2f9e2ce
```

Leaderboard snapshot at handoff, historical only:

```text
1  Kawashigi              3265.2
2  researchstudio.site    3134.4
3  Ueddy                  3101.7
4  somewhere after        3093.4
5  Mohamed abdelrazik     3085.0
6  Utkarsh #2             3080.1
7  Ignat                  3066.5
8  Aaweg                  3061.9
9  uri_kkyhr              3059.6
10 One-For-All            3055.3
13 jasonstillchasin       3033.6
```

These values are mutable. Refresh current leaderboard, ReCurSiON submissions, current public-safe top-agent submission ids and replay availability before using them.

## Current promoted/submitted v32 source

Exact research/promoted source lineage:

```text
scratch/live_top10/current_best_aug14/fourlane_router_v32/main.py
```

Do not reimplement v32 from prose. Open the exact source/assets.

Important corpus:

```text
scratch/live_top10/current_best_aug14/manifest_tapes.json
scratch/live_top10/current_best_aug14/manifest_multi_tapes.json
scratch/live_top10/current_best_aug14/tapes/
scratch/live_top10/current_best_aug14/tapes_multi/
scratch/live_top10/current_best_aug14/one_for_all_public/main.py
```

## Current v33 frontier

Primary research directory:

```text
scratch/live_top10/v33_moe/
```

Naive integrated wrapper:

```text
scratch/live_top10/v33_moe/hier_v33/main.py
```

**Do not promote this file as-is.** It fails the exact five-live-loss gate 8/10 because its mirror routing begins too early.

### BAKERY option-value prefix

A cap3-like WHEAT-retention/light-farm prefix is strong for old BAKERY families. The relevant BAKERY futures cannot legally be distinguished before the second shop.

The first useful legal divergence is at **step144** when the second shop becomes public.

Do not restore the lighter state to cap4/cap5 before the reveal; that destroys the old-family advantage.

### BAKERY -> PIZZA expert

Strong component:

```text
scratch/live_top10/v33_moe/bakery_pizza_itmo/r0/main.py
```

Full ITMO-r0 continuation after the legal step144 BAKERY->PIZZA reveal:

```text
gogogo cap3 baseline  -5449 / -5449
ITMO-r0               +9244 / +9244
```

Market-only ITMO is catastrophic. The full actor+market continuation is the mechanism.

Before mirror integration this architecture passed:

```text
five exact v32 live losses: 10/10
mean ~+7939.8
floor +204

representative old BAKERY family: 18/18
floor +9749
```

Preserve this mechanism.

### BAKERY -> ICE_CREAM high-cash expert

At step144, public opponent capital separates a high-cash ICE regime from low-cash Malelizarazo.

ITMO-r1 continuation gives:

```text
Erfan +2219 / +2219
eason +1355 / +1355
```

Integrate only through a public-state lane, not names.

### Low-cash ICE / Malelizarazo unresolved

Best current coherent continuation:

```text
One-For-All-r0 from step144/168
```

Approximate result:

```text
cap3 -8414
OFA full -2894
OFA actors-only -3121
OFA market-only -7970
```

The remaining gap is structural. Around step648:

```text
ours ~31 WHEAT / 22 STRAWBERRY
Mal  ~56 WHEAT / 5 STRAWBERRY
late WHEAT harvest ours ~160 vs Mal ~298
```

Mass STRAWBERRY->WHEAT conversion inside OFA geometry is catastrophic. Late donor/tape transplants are incompatible. Search a new coherent production basin; do not keep applying crop micro-patches.

### Mirror/Kawashigi current-meta expert

Strong component:

```text
scratch/live_top10/v33_moe/kaw_clone_sellrank/main.py
```

A reconstructed Kawashigi continuation gives 0-margin ties; generic SELL-slot ranking turns it positive.

Kawashigi results:

```text
60200 +1344/+1344
60201 +1267/+513
60202 +644/+644
60203 +801/+801
```

Against the seven original Pasture/HIRE5 mirror opponents — Kawashigi, Anik, One-For-All, Mohan, Ddere, OnlyFarms, Patrick — over four seeds/both seats:

```text
56/56
mean +2721.7
floor +513
```

This is a proven cohort expert, not an identity counter.

### Critical mirror collision / exact next task

The seven original mirror opponents, live Hax and live Wang are **public-state identical through step71**.

First public divergence occurs only at **step72**, when the first shop unlocks.

Examples:

```text
Hax: PET_CAFE first
Wang: ICE_CREAM_SHOP first
Kaw examples vary: BRUNCH / ICE / PIZZA across seeds
```

Therefore the step1 Pasture/HIRE5 selector in the naive `hier_v33` is too early. No legal policy can route Hax/Wang away from the mirror cohort before step72.

Naive integrated exact live gate:

```text
excluding +5743 / +642
gogogo    +9244 / +9244
Hax       -2973 / +33531
Muhammed  +13165 / +3338
Wang      +14932 / -108
TOTAL 8/10
```

Do **not** patch those failures by opponent identity.

A screen of 15 state-compatible current-best continuations found no single common route that dominates the combined old mirror cohort + Hax + Wang.

## Exact next research task

Resume here:

### 1. PET-first mirror lane

At step72 after `PET_CAFE` becomes public, collect **all** Pasture/HIRE5 mirror-opening trajectories that are PET-first, including Hax. Search a coherent full continuation/expert for the whole cohort.

Do not tune to Hax alone.

Success condition:

- Hax positive both seats;
- every PET-first mirror-cohort fixture positive on development and disjoint seeds;
- no changed action before the step72 public divergence.

### 2. ICE-first mirror lane

At step72 after `ICE_CREAM_SHOP` becomes public, collect all ICE-first mirror trajectories, including Wang. Search a coherent continuation for the whole cohort.

Success condition:

- Wang positive both seats;
- every ICE-first mirror-cohort fixture positive;
- old mirror 56/56 strength retained across integrated routing.

### 3. Revalidate other first-shop mirror regimes

Keep Kaw clone+SELL-rank where it is strongest for BRUNCH/PIZZA/other first-shop regimes, or replace only with a cohort-validated alternative.

### 4. Integrate high-cash BAKERY->ICE

Add ITMO-r1 using only public step144 state; validate all high-cash ICE cohort members.

### 5. Continue structural low-cash ICE search

Malelizarazo requires a coherent WHEAT-heavy production architecture, not terminal market surgery.

### 6. Rebuild hierarchical v33

Replace the naive early mirror branch with legal step72 routing. First gate is to recover the exact five-live-loss **10/10** result.

### 7. Broad gates

Run, in bounded <=2-worker batches:

- original 28-opponent × 4-seed × both-seat 224-game gate;
- expanded `manifest_multi_tapes.json` current-best population;
- fresh/disjoint seed blocks;
- newest replay holdouts;
- exact current incumbent comparisons.

The goal remains hard-league conversion toward 100%, not merely non-regression.

## Key negative results from current v33 work

Do not repeat without a materially new premise:

- post144 WHEAT dump to repair cap3;
- cap4/cap5 uncertainty financing;
- fourth-STRAWBERRY seed reconstruction;
- fertilizer-funded missing crop geometry;
- simple WHEAT crop substitution;
- empty-tile TOMATO insertion;
- broad TOMATO/MELON substitution;
- extended PIZZA WHEAT hold;
- early terminal liquidation;
- isolated or mass STRAWBERRY->WHEAT edits inside wrong geometry;
- extra terminal scavenger workers;
- surplus WHEAT watering worker;
- same-turn WHEAT sell/buy netting;
- suppressing structural late WHEAT buys;
- simple extra-cow deletion;
- late current-best actor transplants;
- Mal's own late tape transplant;
- any step1 mirror selector that tries to distinguish Hax/Wang from the old mirror cohort.

The full negative ledger and causal details are in the 2026-08-15 master update and structured handoff.

## Replay workflow

Use the modern CLI:

```bash
uvx --python 3.11 --from kaggle==2.2.2 kaggle competitions episodes <SUBMISSION_ID> -v
uvx --python 3.11 --from kaggle==2.2.2 kaggle competitions replay <EPISODE_ID> -p <DIR>
uvx --python 3.11 --from kaggle==2.2.2 kaggle competitions logs <EPISODE_ID> <AGENT_INDEX> -p <DIR>
```

Replay mapping:

```text
observation step t -> replay steps[t+1].action
719 real actions
```

Require coin-for-coin official-oracle parity before accepting a tape as evidence.

## Arena / mechanics workflow

Use:

```text
agent_system/tools/arena.py                  # C++ backend by default
agent_system/tools/cpp_oracle.py             # Python-agent bridge to maintained C++ engine
cpp-engine/                                  # 1.32.7 parity-validated C++ engine
agent_system/tools/official_oracle.py        # final Python correctness oracle / --backend official
agent_system/.cache/official/kaggriculture.py
```

The competition host's 2026-08-15 balance patch is `kaggle-environments>=1.32.7` / Kaggle merge commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`. It adds the hinge scarcity curve for CARROT/TOMATO/EGG. The old 1.32.6 notebook port and the old official-oracle pin are stale and must not be used.

The maintained C++ engine has passed exhaustive market-price parity, 15 full 719-action official traces, complete observation/action parity canaries, and current v32/v33 matchup checks. `arena.py` therefore defaults to C++, while `--backend official` remains the correctness canary. Never use more than two arena workers; the tool now rejects `--workers > 2`.

Arena isolation commit `11b5245` is mandatory for wrappers/monkeypatches.

Every meaningful simulation result must include both seats and exact seed/opponent provenance. Failed wrappers or incomplete timed-out batches yield no conclusion.

## Submission workflow

Do not submit current scratch v33 yet.

Before future promotion/submission:

1. acquire the official/promotion lock;
2. make minimal evidence-backed tracked changes;
3. build with `agent_system/tools/build_submission.py`;
4. test exact `submission/build/main.py`;
5. raw Kaggle loader smoke: execute exact packaged main with `env={}`, no `__file__`, build directory on `sys.path`, retrieve `agent`, call initial observation;
6. verify exact research/package parity on both-seat canaries;
7. inspect tar contents;
8. record archive size + SHA-256;
9. `repo_git_review` and commit explicit intended paths only;
10. submit the exact archive;
11. verify Kaggle submission id/status/description;
12. re-query live score;
13. do not push unless explicitly requested.

The raw-loader smoke is mandatory because submission `55491619` failed validation from undefined `__file__`.

## Final operating rule

Do not optimize for the appearance of progress.

The strongest current evidence is that **legal lane decomposition and coherent continuation resets** produce order-of-magnitude larger gains than cosmetic thresholds:

- BAKERY->PIZZA coherent reset moved gogogo from roughly -5.4k to +9.2k;
- reconstructing the current Kawashigi basin and improving only SELL-slot execution beats the original mirror cohort 56/56;
- the naive integration failed because a selector was placed before legal public divergence.

Continue by building legal step72 PET/ICE mirror lanes, then integrate and attack the remaining low-cash ICE structural basin. Do not stop at a locally good panel; continue until the current adversarial population is genuinely dominated.
