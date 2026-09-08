# BKG4_GOLD_V25

Isolated baseline translation for ReCurSiON. This candidate is not the authoritative
champion and must not be treated as BUILD/discovery evidence. No schedules or
strategy improvements were authored by ReCurSiON for this port.

## Source And License

- Source: `public-notebooks/current-gold-sept-8.ipynb`, final callable from its
  `%%writefile main.py` cell, frozen September 8, 2026.
- Notebook SHA-256:
  `330085c464e992b4e1cfd4c0cbb7c4d5f97a67fd5c6f754c56bec5a6eed12820`.
- Extracted main.py SHA-256:
  `bb3109331d4087d5e5890515f9b1d1e0d4f9ea6015ee3d1b172a93b8b1d8dbb4`.
- Production schedules and routing: Yusuke Hayashi / yhay81,
  [Shop Router 0908](https://www.kaggle.com/code/yhay81/shop-router-0908).
- Inherited runtime: thomastschinkel, yhay81, tetsutani, Apache-2.0.
  The complete inherited Apache-2.0 notice/license is retained at the beginning
  of `overlay.py`. Native translations carry SPDX notices referencing it.
- Sale-leading ideas: yhay81's Fieldbook. Weed/inventory work: tetsutani and
  thomastschinkel, as credited by the notebook.
- Integration, packing, ablations and original validation: EXP-149 work for
  Ahmed Berat Ozer (the notebook credits Ahmed Berat Özer).
- Notebook research credits retained: mohui666's v66 and TITAN informed terminal
  experiments; destbreso and nikital7 informed simulator/measurement research.
  Their tools, donor terminal code, and hidden-state signatures are not shipped.

The notebook's 582/42/0, 93.27%, 624-game result is donor metadata, not a result
of this port's validation. The port neither reproduced that study nor used its
opponent identities to construct policy.

## Exact Scope

The BKG4 controller has 23 nodes, three I32 blackboard slots (`route`, `day6`,
`day27`), zero node-state slots, and emits no action intents. The existing BKV6
route-pack path supplies the four 719-step schedules. No VM/ABI changes were made.

Route IDs 0/1 are the absent/present YARN_STORE branches latched at/after 144.
Route IDs 2/3 are the EGG <= 9888 / > 9888 branches latched at/after 648.
Day6 is evaluated first, then day27, including when the first call skips directly
to step 648 or later. Missing EGG defaults to 10000; invalid integer values use
the donor's zero fallback. Only literal YARN_STORE matches, not a shop alias.
All canonical JSON and packed route hashes are in
`workspace/gold_v25_port/identity.json`.

Active overlays: hand alignment, weed repair with ordered per-unit pending queues,
sell lead with next-step per-item suppression, and custom step >= 718 rescue.
Suppression retains zero-quantity orders and their market race slots. Sale lead
is forbidden when next step > 718, next step is divisible by 72, or current step
is divisible by four. Terminal rescue follows ordinary state updates, replaces
all unit actions with adjacent/nonempty-inventory DROP or PASS, projects shed
capacity in unit and inventory insertion order, then stably sorts product SELLs
by descending integer current price times quantity.

Inactive: front_run, budget_guard, room_guard, clamp_sells, dead_stock, and chassis
terminal_liquidation. Their default configuration declarations and dead branches
remain in the mechanically extracted Python chassis, but candidate settings
disable them; their implementations are not incorporated. The custom final
rescue is separate from the disabled chassis terminal layer.

State is per player. Initial calls, step 0, and repeated/backward steps reset
route/latches, pending repairs and suppression exactly as in the donor. New games
reset through this same step-based rule; the donor has no separate game-ID reset.
The Python entry preserves donor layer/entry/final-rescue exception fallbacks.
Native input is the official legal observation domain along donor-generated
histories, not a replacement interpreter for arbitrary malformed Python objects.

## Representation Details

The BKV6 records represent quantities, not omitted-argument spelling. A 5,752-byte
donor-only `arity.bin` restores omitted PICKUP/PLACE quantities. The isolated
decoder also restores packed no-op market slots to `[]`, rather than PASS or
dropping them. All 2,876 source actions are checked for exact lossless round-trip.
No donor payload literal or decoded route JSON is duplicated in source control.

The native overlay uses the existing extended route-pack loader and BKG4 VM on
both CPU and CUDA, with a shared host/device implementation. It retains input
inventory key order explicitly. The per-player queue bound is 719 entries per
unit: at most one entry can be added per increasing in-range acting step, replay
cannot increase queue length, and repeated/backward steps reset. The donor tapes
attempt at most 12 hires per day; the official engine dismisses hands daily.
Route decisions happen at day boundaries, so the existing 40-unit BKG4 bound
cannot truncate a donor-reachable workforce. These bounds do not simplify any
reachable donor action or control state.

## Reproduction

From the repository root, with the pinned notebook present:

```sh
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25.extract --write-overlay
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25.native
.venv/bin/python -m unittest kaggriculture_bonsai.agents.gold_v25.test_gold_v25
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25.validate panel
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25.validate canary
.venv/bin/python -m kaggriculture_bonsai.agents.gold_v25.finalize
```

Run GPU commands without RTK. CUDA access may require sandbox approval.
Extraction deliberately resets candidate validation status; finalize rechecks
the evidence. Binaries are disposable under `/tmp/gold_v25_native`, never under
champion/baseline assets or tracked evidence. The pinned notebook is an ignored
user-supplied input; regenerate from the same SHA-256, not a newer download.

Python entry: `kaggriculture_bonsai.agents.gold_v25.agent.agent`.
Native entry: `NativeAgent('/tmp/gold_v25_native/gold_cpu.so')` from `native.py`.
The CUDA entry uses `gold_cuda.so` with `gpu=True`; native hosts can instead use
`gold_v25::tick` with the validated BKG4 policy and route-pack views directly.

Validation compares every farmer/hand action, exact market order sequence and
quantity, route, both latches, last step, suppression, and complete pending queues.
Queue hashes are trace metadata only, not a substitute for queue equality.
The 64-case canary means 128 official-engine runs: independent notebook/native
arms on each matched case, with observations, actions, state and rewards equal.
The panel includes both seats and internal champion/reference opponents; the
canary adds the official deterministic starter. Neither panel is strategy search.
