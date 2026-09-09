# BKG4_SHOP_ROUTER_0909

Isolated port and reinforcement for ReCurSiON. Neither candidate here is the
authoritative champion, and neither replaces `BKG4_GOLD_V25`, `GOLD_V25_PLUS`,
the packaged `GOLD_V25` submission, `assets/champion/` or `assets/baseline/`.
Those remain untouched protected comparison references.

## Source And License

- Source: `public-notebooks/current-gold-spet-9.ipynb`, the final callable from
  its `%%writefile main.py` cell, downloaded September 9, 2026.
- Notebook SHA-256:
  `1d7f6a17bad05dafe27002fb20a0a650603f571c2a3d0408431d1a7d0ab2701b`.
- Extracted `main.py` SHA-256:
  `d6d74997dc5b483db63d8e39cafa1afeec0f366824e75107e109123f111e866b`.
- Decoded `actions.json` SHA-256:
  `17d503f2fd20d59f9c0f14024d1e74a8add8bb9b5561d4d908b45deecb5495ef`.
  All three are the notebook's own published hashes.
- Production schedules and shop-pair routing: Yusuke Hayashi / yhay81,
  Shop Router 0909.
- Sell timing and shed projection follow aurax7's public Reactive Router, as the
  donor's own docstring credits; the donor's Apache-2.0 `LICENSE.txt` cell is the
  licence for that inherited runtime. Native translations carry SPDX notices.
- Porting, controller synthesis, panels and the reinforcement search here are
  ReCurSiON work. No donor validation result is reused as evidence.

The notebook publishes no head-to-head record that this port reproduces or
relies on. No opponent identity, notebook identity, file identity, seed, hash or
replay fingerprint reaches runtime routing in either candidate.

## Layout

- `reference_port/` - the frozen, immutable exact PARENT.
- `plus/` - the separate reinforced candidate. It never edits the parent.

Evidence lives under `workspace/shop_router_0909/`; the parent's own assets and
panels are under `workspace/shop_router_0909/port/`.

## Exact Scope Of The Port

The donor callable is reproduced without simplification. The donor's 13 complete
719-turn tapes are held in the existing BKV6 route pack, one route id per tape;
no scheduled action is encoded in controller nodes. The BKG4 controller holds
only the routing decision: 61 nodes, one `I32` blackboard slot (`route`), zero
node-state slots, and no emitted action intents. No VM or ABI change was made,
and the existing extended route-pack loader, BKG4 VM, native CPU path and CUDA
path are reused unchanged.

Route ids 0..12 are the donor's plan indices. The controller latches the plan at
exactly step 144 from `SHOP_AT(0)` and `SHOP_AT(1)` - the ordered first two
unlocked shops - reproducing the donor's 15-entry `SHOP_PLANS` table, with plan 0
for every other pair. Fewer than two shops, or a shop name outside the eight
known shops, yields -1 from `SHOP_AT` and therefore plan 0, exactly as the donor's
dictionary lookup misses. At exactly step 648 the route becomes plan 2. Both are
equality tests, not thresholds, because the donor tests `step ==`.

Every other donor rule is transcribed as-is, in the donor's own order: per-worker
same-day weed queues that insert `DIG` without consuming the blocked action and
are cleared at each dawn; removal of quantities already sold a turn early, which
keeps zero-quantity orders and their market slots; the one-turn sale lead, with
the donor's `next > 718`, `next % 72 == 0` and `step % 4 == 0` prohibitions and
its `WHEAT`/`FERTILIZER` exclusion; the ten-order truncation; and the step-718
liquidation, which drops reachable inventory, projects the shed and stably sorts
product sells by descending integer price times quantity.

State is per player. A fresh state, including the controller VM, is created on
the first call and whenever the step repeats or goes backward, exactly as the
donor rebuilds `DayState`. The donor has no separate game-id reset and neither
does the port.

### Representation details

The BKV6 records store quantities, not omitted-argument spelling, so a donor-only
`arity.bin` restores omitted `PICKUP`/`PLACE` quantities, and packed no-op market
slots decode back to `[]` rather than `PASS`. The donor keeps each tape's own hand
count rather than resizing to the live farm, so the pack view returns the tape
entry verbatim. All 9,347 source actions are checked for exact lossless
round-trip during extraction.

The native translation shares one host/device implementation. The per-worker
repair queue is bounded at 32 entries: the queue is cleared at every dawn and a
day is 24 turns, so at most 24 entries are reachable and the bound cannot
truncate donor-reachable behaviour. Worker inventory key order is carried
explicitly, because limited shed capacity makes insertion order observable.
Native input is the official legal observation domain, not a replacement
interpreter for arbitrary malformed Python objects.

## Reinforcement Scope

`plus/` keeps the parent pipeline unchanged and adds one compact BKG4 controller
that reads only legal observable state. It may select a different donor plan per
observable shop-pair regime, and may open a step-windowed gate for bounded
terminal repairs. With the donor table and no gate the candidate is the parent
action for action, which `plus/test_plus.py` pins directly.

## Reproduction

From the repository root, with the pinned notebook present:

```sh
.venv/bin/python -m kaggriculture_bonsai.agents.shop_router_0909.reference_port.extract
.venv/bin/python -m kaggriculture_bonsai.agents.shop_router_0909.reference_port.native
.venv/bin/python -m unittest kaggriculture_bonsai.agents.shop_router_0909.reference_port.test_port
.venv/bin/python -m kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate panel
.venv/bin/python -m kaggriculture_bonsai.agents.shop_router_0909.reference_port.validate canary
```

Run GPU commands without RTK. CUDA access may require sandbox approval. Binaries
are disposable under `/tmp/shop_router_0909_native`, never under champion or
baseline assets. The pinned notebook is an ignored user-supplied input;
regenerate from the same SHA-256, not a newer download.

Python entry: `...shop_router_0909.reference_port.agent.agent`.
Native entry: `NativeAgent('/tmp/shop_router_0909_native/shop_cpu.so')`, and the
same class with `shop_cuda.so` and `gpu=True` for the CUDA path.

The differential panel compares every farmer and hand action, the exact market
order sequence and quantities, the route, the day latch, the sale-due step, the
advanced-sale quantities and the complete per-worker repair queues, across the
donor, the Python port, the CPU port and the CUDA port on identical inputs.
Neither panel is strategy search.

## Recorded Results

Panels are disjoint by seed. BUILD (2000-2399) drove all search; INTERNAL_DEV
(3000-3031) only screened frozen candidates; FINAL_TARGET (5000-5063) was
untouched until the composition was frozen. Parity and canary seeds are
separate again. No FINAL_TARGET failure was ever fed back into search.

### Exact port

Differential panel: 24 games, 17,256 acting steps. Zero action mismatches, zero
state mismatches, zero CPU/CUDA mismatches, zero faults, across the donor, the
Python port, the CPU port and the CUDA port on identical inputs. All 13 routes
and all three repair branches were exercised.

Official canary: stopped early by request after 42 of 64 planned matched cases
(84 engine runs, 30,198 acting steps compared). Every completed case matched
exactly; see `workspace/shop_router_0909/port/canary.json`.

### Reinforced candidate

Composition: the donor route table with one shop-pair specialist
(`YARN_STORE|FARMERS_MARKET` takes plan 9 rather than the donor's plan 1) plus
one bounded terminal repair (project the shed and top every product sell up to
it) gated to steps 704-717. Controller: 85 nodes, two `I32` blackboard slots,
zero node-state slots, reading only `STEP` and `SHOP_AT`. Route pack unchanged.

BUILD, 60 seeds, both seats, 360 games: new notebook 118/0/2, GOLD_V25_PLUS
120/0/0, GOLD_V25 120/0/0.

INTERNAL_DEV, 32 seeds, both seats, 192 games: 64/0/0 against all three targets.

FINAL_TARGET, 64 untouched seeds, both seats, 384 games:

| Target | W | T | L |
| --- | --- | --- | --- |
| NEW NOTEBOOK | 128 | 0 | 0 |
| GOLD_V25_PLUS | 124 | 0 | 4 |
| GOLD_V25 | 127 | 0 | 1 |

The five losses come from two seeds only, 5010 (`BRUNCH_SPOT|YARN_STORE`) and
5054 (`YARN_STORE|YARN_STORE`), with margins from 577 to 2,604. The 100% / 0 T /
0 L target is therefore met against the new notebook but **not** against either
GOLD agent, so this candidate is not promoted and is not a champion.

CPU/CUDA parity: 12 games, 8,628 acting steps, zero mismatches, zero faults,
with the gate firing on 168 steps. Official canary: 32 matched cases, 64 engine
runs, 23,008 acting steps, zero mismatches, zero faults.

### Rejected by BUILD evidence

Recorded so the search is not repeated: richer terminal repair masks (adding
forced drops, shed approach, multi-step sale lead or value sorting to the
projected-shed top-up) each removed no failure and added new ones; and route
overrides for the `ICE_CREAM_SHOP|PIZZA_SHOP` and `BRUNCH_SPOT|YARN_STORE`
regimes were strictly worse than the donor's own choice for those pairs.

### Baselines

`BKG4_GOLD_V25`, `GOLD_V25_PLUS`, the packaged `GOLD_V25` submission,
`assets/champion/` and `assets/baseline/` are unmodified by this work.
The `gold_v25` extraction tests require `public-notebooks/current-gold-sept-8.ipynb`,
which is not present in this environment; that ignored input predates this work
and is unrelated to it. The `gold_v25` agent itself loads and plays normally.
