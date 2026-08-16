# Continue Kaggriculture — v34 CUDA + Bonsai

Continue the Kaggriculture competition project from the current compact repository-grounded handoff. Do **not** reconstruct project state from memory, old assistant messages, stale leaderboard values, or the older giant handoff files.

Use `Local_mcp` with:

```text
repo_id = kaggriculture
root = /home/nnmax/Desktop/kaggle/kaggriculture
```

## Mandatory reads

Read fully, in order:

1. `.chatgpt/handoffs/current.local.md`
2. `.chatgpt/handoffs/KAGGRICULTURE_MASTER_PROMPT.local.md`
3. `.chatgpt/handoffs/2026-08-16-0039-kaggriculture-v34-cuda-bonsai-continuation.local.md`
4. `RULES.md`
5. `AGENTS.md`
6. `README.md`
7. `cpp-engine/README.md`
8. `cuda-engine/README.md`
9. `agent_system/README.md`
10. `continue_prompt.md`

Optional only when an old route/result becomes relevant:

```text
.chatgpt/handoffs/KAGGRICULTURE_HISTORY_COMPACT.local.md
```

Do not automatically ingest the older 2026-08-12/14/15 giant handoffs.

## Mission

> **Finish top 10, earn gold, maximize prize-money probability.**

The strict promotion requirement is **100% wins, zero ties/losses** against a current diverse promotion league containing major-top, top-mid, mid and low-mid/historical agents, both seats, followed by a disjoint fresh seed block and blind newest-replay holdout. This is a measured gate, not a promise.

Do not settle for cosmetic gains. Never get stuck in a local optimum. Prefer structural improvements to opening economy, production/livestock mix, geometry, market controller, reactive execution, or behavior-tree program structure over endless threshold patches.

## Mandatory research method

Follow Kaggle discussion 734212's verified closed-loop research process:

```text
real losses
→ isolate one mechanism
→ challengers
→ multiple teams + both seats
→ reject most
→ freeze winner
→ later-episode holdout
```

Maintain two lanes in parallel:

1. **current-meta optimization**;
2. **new-meta creation**.

The core causal architecture is:

```text
shared public prefix
→ preserve option value while futures are indistinguishable
→ branch only at first legal public divergence
→ coherent lane-specific expert
→ later legal subbranches if evidence justifies them
→ adversarial population gates
```

Never use team/opponent/submission/episode identity, seed, hidden state, future shop, or future RNG in production routing.

## Current technical assets

- Exact official semantics: Kaggriculture 1.32.7 / merge commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`.
- Exact CPU engine: `cpp-engine/`, final performance commit `ff6b585`.
- Exact CUDA engine: `cuda-engine/`, commit `19658376eb4733b2919c2492a51528dbe9625e13`.
- Bonsai reference clone: `bonsai/`, upstream `8fe964fca85d584c7575fcba2754ff2e9f6fd031`.
- Current research: `scratch/live_top10/v34_bonsai/`.
- Production incumbent: `scratch/live_top10/current_best_aug14/fourlane_router_v32/main.py`, commit `d52526d`, submission lineage `55507122`.

Max **2 CPU simulation workers**. Use CUDA for large `state/seed × candidate-program` branch populations.

## Current v34 evidence

Sparse shop trees overfit. Broad 128-seed evidence:

```text
simple two-level tree                 3275/3584
step144 best-final oracle             3440/3584
best of all 13 mids × all 33 finals  3505/3584
```

Six proven third-level option-value subtrees raised cached projection to `3330/3584`, and the compiled hierarchy scored `213/224` on fresh 610xx. Fixed hierarchy scored `170/224` on fresh 612xx; universal reactive root improved it to `179/224`.

The strongest new concept is **reactive executable leaves** rather than replay tapes. A Ueddy YARN economic route augmented with generic weed/state-drift recovery + SELL ranking scored `251/252` on fresh YARN-first fixtures with mean `+18,947.5`, floor `-153`.

Ueddy is mandatory in the promotion population. Current Ueddy public submissions at the handoff were `55528957` and `55528783`; refresh them.

## Exact next task

PET→BAKERY was missing from the original broad training tuple coverage and caused the severe fresh seed-61207 seat0 failure cluster.

Broad fresh cohort:

```text
Ueddy PET fixed leaf     160/168
Ueddy PET reactive leaf  164/168
fixed-library oracle     164/168
```

But integrated 61207 seat0 remains unresolved, and a deeper fixed replay subtree was rejected.

**Resume with this exact experiment:**

For PET→BAKERY seed 61207 seat0, determine whether any combination of:

```text
13 legal first-shop mids × 33 final continuations
```

can win representative strong opponents. Use CUDA/native branch search, not a slow Python brute force.

- If oracle-positive, find the earliest legal public-state feature that routes the whole causal cohort to the winning basin.
- If oracle-negative, stop replay recombination and design a genuinely new PET reactive/economic program/opening.

Do not fit a rule to seed 61207 or a player name.

## After that

- Keep converting strong donor economics into reactive executable leaves.
- Expand program search beyond shop identity to legal public features: money, market prices/inventory, farm composition, workers/land, and seat only when cohort evidence justifies it.
- Keep a separate new-meta search active so the project does not converge to a polished version of the same mirror meta.
- Refresh current top/top-mid/mid/low-mid league before final gates.
- No v34 promotion until the strict 100% multi-tier + disjoint + blind-holdout requirement is actually satisfied.

## Git/worktree caution

Expected handoff HEAD: `19658376eb4733b2919c2492a51528dbe9625e13`, but verify.

Known worktree items at handoff:

```text
M  cuda-engine/src/tape_branch.cu
?? bonsai/
?? task lock files
?? public-notebooks/4000x-environment-speedup-kaggriculture.ipynb
?? public-notebooks/read-the-market-choose-the-farm.ipynb
```

`cuda-engine/src/tape_branch.cu` contains an intentional diagnostic raw/public snapshot export used by v34 synthesis. Review its diff before changing. Do not stage the nested Bonsai repo or unrelated notebooks accidentally.

No push unless explicitly requested.
