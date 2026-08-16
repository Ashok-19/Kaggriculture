# Kaggriculture Gold Research Rules

This file contains the durable operating rules for every future ChatGPT/Codex/agent session in this repository. Current experiment state belongs in `.chatgpt/handoffs/`; do not bloat this file with mutable leaderboard snapshots.

## 1. Objective

The objective is to finish **top 10**, earn **Kaggle gold**, and maximize prize-money probability. Optimize for robust tournament strength, not local score, code volume, dashboard polish, or the appearance of progress.

The aspirational promotion gate is **100% wins, zero ties/losses** on a broad current+historical league, both seats, followed by disjoint fresh seeds and blind newest-replay holdouts. Treat this as a measured gate, never a promise.

## 2. Evidence first — no assumptions

Use this evidence order:

1. repository source/state;
2. official Kaggriculture 1.32.7 semantics;
3. exact CUDA/C++ engine output;
4. parity-verified public replay evidence;
5. freshly queried Kaggle leaderboard/submission/episode state;
6. causal inference supported by the above.

Never treat memory, an old assistant message, stale leaderboard value, filename, intuition, or an incomplete run as truth when it can be checked.

Rules:

- Mutable facts must be refreshed before important decisions: leaderboard, active submissions, episode streams, replay availability, current top population, live score.
- A timed-out/crashed/parser-failed run gives **no policy conclusion** unless a complete persisted artifact is independently verified.
- Distinguish exact result, causal inference, and hypothesis.
- Do not fabricate success, commits, uploads, scores, replays, or tool output.
- If observations are identical, deterministic production behavior must be identical.

## 3. CUDA-first research

For serious policy research, **default to the exact CUDA engine**.

Use CUDA for:

- evolutionary/genetic program search;
- beam search;
- MCTS/tree-search teachers;
- minimax/option-value search;
- population search over many seeds/opponents/candidates;
- exact same-state counterfactual continuation oracles;
- Bonsai program/feature/leaf search;
- opening/economic architecture search.

Do not use slow Python/CPU brute force when the same experiment can be expressed in CUDA.

CPU/C++/official execution is allowed for correctness canaries, unsupported policy interfaces, tiny diagnostic cases, packaging parity, or independent verification. It is not the default search engine.

**Hard local CPU limit: at most 2 simulation workers.** Never use bare `Pool()`, all-core sweeps, or an unbounded heavy `-j`. After an interrupted worker batch, inspect/reap stale repo-local processes.

Never use CUDA fast-math. Any simulator-semantic change requires the full parity gate.

## 4. Search must expand Bonsai, not replace it

The intended architecture is:

> exact CUDA search/evolution/MCTS/beam teacher -> distilled deterministic Bonsai-style policy -> final deterministic Python runtime.

Bonsai is the policy/control backbone, not the search algorithm. If search finds a strong action/continuation, convert the result into legal observation-derived rules/features/leaves or a coherent executable expert.

Do not ship a runtime MCTS merely because MCTS was useful offline. Runtime should remain deterministic unless strong evidence proves otherwise.

Bonsai capacity is not a current practical bottleneck. Expand/factor the rule language when evidence requires it. The limiting factors are usually search quality, feature language, action/expert expressiveness, and economic basin coverage.

## 5. Causal lane architecture

The mandatory pattern is:

> shared public prefix -> preserve option value while futures are indistinguishable -> branch at first legal public divergence -> coherent lane-specific continuation -> later legal subbranches -> adversarial population validation.

A lane is a public-state/economic cohort, **not a player name**.

Never use in production:

- team/opponent name;
- submission id;
- episode id;
- seed;
- hidden state;
- future shop/RNG;
- replay fingerprints or proxies whose only purpose is hidden identity recognition.

If two trajectories have identical legal observations, the policy must remain common. When they diverge publicly, branch only on features visible before the changed action.

Own private shed/seeds/inventories are legal because they are part of the agent observation. Opponent private state is not.

## 6. Structural changes over cosmetic patches

Do not spend the project on tiny thresholds, comments, naming, formatting, or wrappers.

Prioritize interventions that can move whole failure lanes:

- opening economy;
- worker/land geometry;
- crop/livestock mix;
- shared-market timing/order structure;
- coherent actor+market continuation;
- reactive recovery/execution;
- reusable economic experts;
- new legal features;
- new Bonsai hierarchy;
- new meta basins.

If repeated micro-patches only move margin slightly, stop and restructure. If an exact oracle shows no winning continuation from the current state, intervene earlier or create a new expert/opening instead of tuning the selector.

Do not overengineer trivial infrastructure. Add a tool only when it directly unlocks a high-value experiment or materially reduces repeated research cost.

## 7. Code quality

Prefer **small, clear, direct code** over framework-building or bloated abstractions.

- Keep experimental CUDA kernels and host drivers explicit and auditable.
- Avoid duplicate generic layers unless they eliminate real repeated complexity.
- Do not rewrite working engines for style.
- Do not perform cosmetic refactors during active search unless they remove a concrete blocker.
- Preserve exact semantics before performance or elegance.

New research source files must be tracked/staged promptly; do not leave unexplained untracked repo files/folders. Large generated experiments belong under ignored `scratch/`.

## 8. Evolutionary / beam / MCTS discipline

Use large-scale search aggressively when the action/program space is too coupled for manual reasoning.

Recommended sequence:

1. exact small oracle to measure available capacity;
2. broad CUDA beam/evolution/MCTS search to discover structure;
3. freeze candidates;
4. test on disjoint seeds/opponents;
5. inspect causal features/first divergence;
6. distill into deterministic Bonsai/expert logic;
7. rerun broad regressions and holdouts.

Use multiple discovery seeds/islands. A candidate that is perfect on one seed block is not trusted until it survives out-of-sample aggregate testing.

Balanced fitness should protect floor/tail/mean as well as win count. Pure win-count specialists with catastrophic margins are normally rejected.

## 9. Replay and live-meta discipline

Public replays are realized trajectories, not executable opponent policies.

- Verify action alignment and exact source parity before trusting a tape.
- Replay mapping is `observation step t -> replay steps[t+1].action`; 719 real actions.
- Use multiple streams/replays to mine mechanisms.
- Keep the newest distinct elite episodes frozen as holdouts whenever possible.
- Refresh current top replays frequently; the meta changes quickly.
- Current #1/top agents are mandatory research sources when public data exists.
- Preserve historically informative agents such as `jasonstillchasin` when they expose distinct mechanisms.

Do not inspect a frozen holdout before the candidate/policy decision it is intended to test.

## 10. Regression and overfit gates

Every meaningful candidate must be challenged for overfit.

At minimum, when applicable:

- both seats;
- multiple opponents from the same causal cohort;
- broad current league;
- historical/diverse league;
- multiple discovery seeds;
- disjoint validation seeds;
- newest-replay holdout after freeze;
- per-opponent/per-seed breakdown;
- wins/ties/losses, mean margin, floor/tail;
- first-trigger/routing audit for false entries;
- exact policy coherence where a latch is intended to persist.

A 128/128-style hard panel is a research target, not enough by itself. It must remain causal and survive fresh populations.

Do not promote a candidate merely because it beats the incumbent or one top player 100%.

## 11. Market/state coupling

The market is adversarial and path-dependent. Order-slot alignment, inventory, price, future farm consumption, and opponent quote impact matter.

Actor geometry and market/economy are often co-adapted. Prefer whole coherent continuations over isolated market-only or unit-only transplants when experiments show coupling.

Farm geometry can change weed RNG consumption and later public town/shop RNG, so geometry-changing edits are whole-policy interventions.

## 12. Established engine workflow

Pinned official semantics: Kaggriculture / `kaggle-environments==1.32.7`.

Exact engines:

- `cuda-engine/` — default serious research/search engine;
- `cpp-engine/` — exact CPU engine for small/unsupported/canary work;
- official Python interpreter — final correctness canary, not the research workhorse.

For CUDA semantic/source changes, run:

```bash
make -C cuda-engine check
```

This must retain exact market, trajectory, logical-state, snapshot/resume, cached/uncached RNG, and device-policy parity.

## 13. Version control / workspace

- Use `Local_mcp` with repo id `kaggriculture`.
- Claim a fresh task before focused work; do not reuse expired lock files.
- Do not delete lock files manually just because they are old; use the workspace task APIs when needed.
- Commit meaningful tracked research-engine/source changes after review and validation.
- Stage explicit intended paths only.
- Do not accidentally stage nested reference repos, notebooks, generated scratch, or unrelated user edits.
- Do not push unless the user explicitly asks.
- Do not clean/reset/restore unrelated dirty files blindly.

## 14. No routine SHA/hash work

**Do not spend research time on SHA checks, artifact hashing, hash ledgers, or hash-based handoff ceremony.** The user explicitly does not want this overhead.

Use ordinary file/path/version/tool evidence. Only perform a hash/integrity check if the user explicitly asks or an external submission/tool workflow strictly requires it for a concrete reason.

Do not include hash computation as a default handoff, validation, or research step.

## 15. Submission/promotion

Do not submit scratch research automatically.

Before promotion:

1. refresh/freeze the current promotion league;
2. pass broad current + historical regression gates;
3. pass disjoint fresh seeds;
4. freeze the candidate;
5. expose blind newest-replay holdouts;
6. verify exact packaged runtime behavior;
7. review exact intended source changes;
8. submit only when the evidence justifies it and the user wants submission.

No identity leakage, seed leakage, future information, or non-deterministic hidden lookup is acceptable.

## 16. Research logging

For important experiments record enough to reproduce the conclusion:

- exact candidate/program path;
- route/opponent list path;
- seed start/count;
- both-seat status;
- wins/ties/losses;
- mean/floor/tail when relevant;
- first legal branch point / features;
- promote/reject/hold decision;
- failure reason for rejected branches.

Preserve negative results so future sessions do not rerun dead ends without a materially new premise.

## 17. Final rule

**Do not optimize for the appearance of progress. Search for the strongest causal meta the exact engine can support, then prove it on adversarial unseen evidence.**
