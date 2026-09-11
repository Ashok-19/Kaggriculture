# Agent Instructions

- The active discovery architecture is **LLM-guided lane evolution over Bonsai/BKG4**, governed by `docs/FINAL_LLM_LANE_EVOLUTION_PLAN.md`.
- Bonsai/BKG4 remains the final runtime. LLMs are proposal/program-generation agents only; exact simulator evidence decides candidate survival. No LLM or neural dependency may exist in the submitted runtime.
- Preserve the lane-based approach. Specialists may be strong in a narrow causal regime and weak elsewhere; the complete routed/composed policy is the release-level regression unit.
- Use CUDA for large evaluation and the C++ oracle for bounded reference checks. CPU/CUDA semantics must match exactly; official Python/Kaggle canaries remain mandatory before release.
- BUILD is fresh internal self-play/adversarial evidence by default. A public/historical source may be used for LLM hypothesis generation only when explicitly declared `TRAINING_SOURCE`; once exposed it cannot count as independent HOLDOUT evidence for that epoch.
- Runtime routing must use legal causal observations only. Never route on opponent/team/submission identity, replay hash, episode ID, hidden RNG seed, future observations, or opponent-private state.
- Preserve the safety flow: lane BUILD -> fresh INTERNAL_DEV -> routing/composition -> TRANSFER/VALIDATION/CEGIS/SEALED -> official canary -> reviewed promotion. Any panel whose result feeds later design is retired as unseen evidence.
- INTERNAL_DEV and downstream panels stay disjoint from BUILD. SEALED is unavailable to proposal agents until its designated release stage.
- Final protected win-to-loss must remain zero unless an explicit future governance decision changes that standard. Individual specialists need not satisfy global zero-regression before composition.
- `assets/champion/` is the only authoritative champion. Never overwrite it from an incomplete lane experiment.
- The retained learned-policy research alternative is `docs/FINAL_POLICY_DISCOVERY_ARCHITECTURE.md`; it is not the active execution plan.
- RLRMB, Direct-Bonsai, CCPR, route mutation, the generic task executive, and the service-tour representation are historical diagnostics/falsified discovery paths, not defaults to revive under new names.
- Keep agent tasks evidence-first: inspect actual source, state one causal hypothesis, implement an executable candidate, evaluate, and stop at the assigned budget. When stuck, reconsider the hypothesis and inspect source/research rather than increasing brute-force search.
- `workspace/` stores compact tracked evidence. Commit or remove every non-ignored file before finishing. Never create `scratch/`.
- Import live Python through `kaggriculture_bonsai`; do not add `sys.path` manipulation or depend on deleted repository history.
- Keep the root and dependency set small. Commit only source, current configs, compact assets, tests, and current documentation. Never commit caches or campaign output.
- For GPU-related commands, run without rtk to avoid errors.
- Do not push or submit unless explicitly requested.
