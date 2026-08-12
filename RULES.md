# Kaggriculture Gold-Medal Rules

This file contains durable operating rules for every future ChatGPT/Codex/agent session working on this repository.

## 1. Objective

The objective is not merely to improve local score. The objective is to finish **top 10**, earn **gold**, and maximize prize-money probability in the Kaggriculture Kaggle competition.

Every research, engineering, validation, and submission decision must be evaluated against that objective.

## 2. Evidence First — Never Work From Assumptions

1. Treat repository state, pinned official engine behavior, exact local simulations, verified Kaggle tool output, and downloaded replay evidence as primary sources of truth.
2. Do not infer a game mechanic from intuition, a forum comment, a notebook headline, leaderboard folklore, or a previous assistant message when it can be checked directly.
3. Mutable facts must be re-verified before use: leaderboard rank/score, active submissions, current top-10 teams, episode counts, Kaggle submission status, public replay availability, and opponent strategy fingerprints.
4. When a wrapper/script fails before producing a result, record **no policy conclusion** from that run.
5. When a batch times out before returning complete auditable output, do not use guessed/partial results unless the partial results were explicitly materialized and independently verified.
6. Distinguish clearly between:
   - official-engine fact;
   - exact empirical result;
   - causal inference supported by evidence;
   - hypothesis awaiting validation.
7. Never fabricate a replay path, submission status, score, artifact hash, or successful push/upload.

## 3. Four-Lane Research Program

All major successor development must proceed through four complementary lanes. Do not collapse the project into one best-response loop.

### Lane 1 — Mechanics, Invariants, and Failure Repair

Purpose: eliminate genuine execution/recovery defects and discover engine-level edges.

Work includes:
- exact failure replay and first-divergence diagnosis;
- weed/dependency recovery;
- day-boundary and roster-reset invariants;
- worker synchronization;
- shed/logistics invariants;
- atomic market/order semantics;
- price-function and scarcity mechanics;
- safe preemption/debt accounting;
- official-engine fact checking of public notebooks/discussions.

Rules:
- prefer a general invariant over a seed/opponent patch;
- prove the original action is impossible/no-op or that stolen route slack is genuinely free before replacing it;
- after a repair, rerun every accumulated historical hard seed plus fresh disjoint seeds.

### Lane 2 — Elite Adversary / Kaito / Current-Top Specialist Lane

Purpose: learn from and counter the strongest publicly observable agents without overfitting to identity.

Work includes:
- download public replays from current elite submissions;
- reconstruct coherent route/action families from cross-replay consensus;
- build fixed-tape fixtures only as one layer, never confuse replay tape with executable policy;
- reconstruct/surrogate conditional branches where possible;
- derive selectors only from public information observable **before** the relevant commitment;
- explicitly test hidden-future/minimax cases where observations are identical but future RNG differs.

Rules:
- never key production logic to team name, submission id, seed, or episode id;
- opponent-family recognition must use causal public state signatures;
- if complete observations are identical at a decision point, do not invent a classifier for different futures; solve the minimax/option-value problem instead;
- hold out newest elite replays until a candidate is frozen.

### Lane 3 — Population Robustness / Historical League

Purpose: prevent a successor from being a narrow best response.

The league should include, whenever locally available:
- current submitted ReCurSiON agent;
- previous ReCurSiON agents;
- v27 family;
- Lev/current-control variants;
- Boatlee;
- Rayk/Lev descendants;
- Oleg/THUNDER;
- Pilkwang;
- Breaking;
- Tetsu;
- Junaid;
- Jince/Lev/v27 coherent route experts;
- reconstructed elite surrogates;
- any newly discovered coherent meta families.

Promotion should be evaluated on the **worst opponent per seed/seat**, not only average win rate. The league is allowed to adversarially choose the hardest local opponent for each fixture.

### Lane 4 — Meta Discovery / Live Top-10 Evolution

Purpose: avoid getting trapped in one route family or local optimum while the leaderboard meta changes.

Work includes:
- refresh the actual current top-10 leaderboard and active submission ids;
- mine their public episodes/replays;
- fingerprint routes, opening economics, livestock mix, market timing, worker/land counts, shop branches, and recovery behavior;
- search coherent alternative production/economic architectures, not only patches to the current route;
- maintain at least one research thread that asks whether the current backbone itself is obsolete;
- periodically reopen route splice/production architecture search against the **current population**, not only against an old baseline;
- consider deterministic search, evolutionary optimization, macro-policy search, learned selectors/value functions, or RL/hybrid methods where they add real evidence-backed value.

The goal is continual creation of a **new better meta**, not merely exploitation of the previous agent.

## 4. Anti-Local-Optimum Rules

1. A challenger beating the current agent 100% is **necessary evidence but never sufficient evidence**.
2. Do not repeat the mistake where a locally dominant best response moved live Elo only marginally.
3. Large improvement versus one controller can be an exploit of that controller rather than a general strength gain.
4. Maintain an expanding population of historical and external opponents.
5. Test both seats. Seat asymmetry is a first-class failure mode.
6. Use disjoint seed blocks for development, validation, and final holdout.
7. Preserve a blind holdout whenever possible, especially for Kaito/current-top replay data.
8. Prefer large intrinsic margins. A 100% panel of +5 coin wins is much weaker evidence than a 100% panel with thousands of margin.
9. Investigate deterministic/symmetric losses differently from one-seat stochastic tails.
10. Do not keep stacking wrappers around a flawed mechanism. Periodically decompose and simplify the policy.
11. If two experts are complementary, first search for a coherent minimax policy or causal late selector before adding brittle classification.
12. Never use future shop information, hidden RNG, seed ids, or opponent ids to make an earlier decision.

## 5. Promotion Gates

No scratch experiment becomes official production merely because it looks promising.

Before promotion, aim for all of the following when feasible:

1. **Current-baseline gate** — strict paired-seat wins over the current submitted/frozen baseline on a fresh disjoint seed range.
2. **Historical hard-set gate** — all accumulated known failure seeds re-pass.
3. **Worst-of-league gate** — zero losses/ties against the available local opponent league, evaluated per seed and seat.
4. **Fresh disjoint league gate** — repeat on a seed block not used to design the candidate.
5. **Elite-surrogate gate** — zero observed losses against reconstructed Kaito/current-top strategy families on development and disjoint panels.
6. **Blind holdout gate** — only after candidate freeze, expose reserved newest replay/episode data.
7. **Packaging gate** — run the exact built `submission/build/main.py`, not only a scratch wrapper.
8. **Native/parity gate** — all relevant C++/official-oracle tests pass.
9. **Archive integrity** — verify top-level contents, size, and SHA-256 before submission.

If the user explicitly instructs submission before every aspirational gate is perfect, freeze the **best broadly validated candidate**, exclude speculative probes, document known residual risk, and submit exactly the tested package.

## 6. Opponent Replay Rules

1. A public replay contains realized actions on one trajectory, not the opponent's executable policy.
2. Fixed-tape wins are useful but must not be represented as guaranteed wins against the real adaptive opponent.
3. Verify replay alignment and observation reconstruction before using replay fixtures.
4. Seat-1 replay frames may be deltas/private overlays; reconstruct shared fields correctly.
5. Whenever possible, verify our own archived submission reproduces all recorded actions before trusting the replay bench.
6. Cross-replay consensus is preferred over copying one trajectory.
7. Surrogates must be validated on held-out replays.

## 7. Arena Isolation Rule

Scratch wrappers frequently monkeypatch `runtime_gateway`. Candidate and opponent must never share the same cached Python gateway module.

`agent_system/tools/arena.py` contains an isolation fix. Preserve it and verify isolation when evaluating monkeypatch-based wrappers. If candidate/opponent module contamination is possible, controller-ablation results are invalid until rerun under isolation.

## 8. Official Engine / Mechanics Rules Already Established

Unless repository/official engine verification proves otherwise:
- season = 720 steps, 24 turns/day;
- act timeout = 1 second;
- market order cap = 10;
- shed cap = 100;
- starting money = 3000;
- unit actions execute before market;
- hires happen during market; a newly hired hand acts starting next turn;
- HIRE/BUY_LAND are atomic in player order;
- SELL/BUY_PRODUCT resolve per order slot across players;
- same product in same slot shares the same precommit quote;
- market prices refresh after each slot;
- SELL at $1 removes the item and pays $1 but does not increase market inventory;
- only WHEAT and FERTILIZER are BUY_PRODUCT items;
- plants/animals die/escape after two consecutive unwatered/unfed day refreshes;
- planting day counts as the first unwatered day, so a newly planted crop must be watered the same day;
- midnight resets farmer position and dismisses hands; stale route-recovery transactions must not blindly survive roster reset;
- DROP at shed access deposits the worker's entire carried inventory before market, subject to shed capacity;
- DIG cannot remove a pasture/coop that currently contains an animal.

Always inspect README.md/AGENTS.md/official interpreter before changing a mechanics assumption.

## 9. Market Research Rules

1. The market is adversarial and path-dependent. Maximizing our own immediate sale value is not always maximizing match margin.
2. Shared-market timing, order-slot alignment, scarcity denial, and opponent quote impact matter.
3. Public notebook claims must be independently reproduced against the official interpreter.
4. The `strawberry pays 24x` notebook's scarcity arithmetic was independently verified as mathematically real in the appropriate scarcity-hole scenario, but naive scarcity-hole sale capping regressed our agent and was rejected.
5. Do not import a public notebook executor wholesale merely because its arithmetic insight is correct.
6. When advancing a sale, reserve future farm consumption/pickups and account for debt correctly.

## 10. RNG / Geometry Rule

Farm geometry can change how many weed RNG draws are consumed, which can alter later town/shop RNG. Therefore a seemingly harmless extra crop/pasture can change future shop sequences.

Any geometry-changing experiment must be evaluated as a whole-policy intervention. Do not assume it only changes production on that tile.

## 11. Version-Control Rules

1. Commit every meaningful tracked source/test/tool change, however small.
2. Scratch experiments remain scratch/research-only unless promoted deliberately.
3. Do not commit unrelated pre-existing files.
4. The untracked `research/` package may exist in the worktree; leave it untouched unless the user explicitly asks to promote it.
5. Before commit, use `repo_git_review` and stage explicit intended paths only.
6. Never claim a commit/push/upload without tool evidence.
7. Do not push unless explicitly requested or project policy says to do so.
8. Keep generated `submission/build/` and tarballs out of source commits unless intentionally tracked.

## 12. Submission Rules

1. Build with `agent_system/tools/build_submission.py`.
2. Test the exact packaged `submission/build/main.py`.
3. Verify `tar -tzf submission/submission.tar.gz` and SHA-256.
4. Record production commit and archive hash in the submission description whenever practical.
5. Use Kaggle connector/CLI evidence to verify the submission id and status.
6. After submission, treat live score as mutable and re-query before drawing conclusions.

## 13. Current Strategic Principle

The likely strongest architecture is hybrid:

**deterministic execution/recovery + coherent production routes + exact market logic + causal opponent/town regime selection + population/meta search**, with learned components added only where they outperform evidence-backed deterministic alternatives.

Do not assume the final top 10 will be pure RL, and do not reject RL/search either. Evaluate methods empirically.

## 14. Communication / Research Logging

For every meaningful experiment, record:
- candidate file;
- baseline/opponent file;
- exact seed range;
- both-seat count;
- wins/losses/ties;
- mean margin;
- minimum margin;
- intrinsic/worst-seat margin where available;
- whether the run used corrected arena isolation;
- causal interpretation;
- promote/reject/hold decision.

Negative results are valuable. Preserve them in handoffs so future sessions do not repeat failed branches.

## 15. Final Rule

**Never optimize for the appearance of progress. Optimize for robust tournament strength.**

A top-10/gold candidate must survive adversarial evidence from multiple metas, not merely dominate yesterday's agent.
