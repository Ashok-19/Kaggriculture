# Agent Instructions

- The active architecture is self-play Bonsai/BKG4. Do not revive old neural
  RL, BC, PPO, replay-corpus, edit-former, or meta-search systems without an
  explicit strategic decision.
- BUILD comes only from fresh self-play and the internal adversarial league.
  Never use public or historical opponent identity to construct policy.
- External opponents are independent HOLDOUT/generalization evidence only.
- Preserve the flow: multi-regime BUILD -> INTERNAL_DEV -> RLRMB search ->
  binary routing -> TRANSFER/VALIDATION/CEGIS/SEALED -> composition.
- INTERNAL_DEV and all downstream panels stay disjoint from BUILD. SEALED is a
  write-only safety firewall until its designated stage.
- Use CUDA for large evaluation and the C++ oracle for reference checks. CPU
  and CUDA semantics must match exactly; semantic mismatches and final
  protected win-to-loss must both remain zero.
- Binary routing already exists. Do not casually add routing complexity or
  weaken safety thresholds.
- `assets/champion/` is the only authoritative champion. Generated tapes,
  binaries, checkpoints, and results go under ignored `workspace/`. Optional
  HOLDOUT inputs go under ignored `external/`.
- Import live Python through `kaggriculture_bonsai`; do not add `sys.path`
  manipulation or depend on deleted repository history.
- Keep the root and dependency set small. Commit only source, current configs,
  compact assets, tests, and current documentation. Never commit caches or
  campaign output.
