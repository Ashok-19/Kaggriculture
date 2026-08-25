# P0.5 Bounded Automatic Bonsai Composer

**Verdict: VALIDATED.** The bounded automatic composer consumes the current parent, three existing `BoundedRouteMacro` documents, one embedded route213 authority declaration and one evidence-backed route218-over-route213 rule. It emits a controller and route pack that are byte-identical to the manually composed P0.4 policy, while rejecting unresolved authority conflicts instead of inheriting append order.

## A. Repository state

- repository: `/home/nnmax/Desktop/kaggle/kaggriculture`;
- team: ReCurSiON;
- branch: `main`;
- starting HEAD: `2d7891393cfe5713f1e9fcd509fa25260c35000c`;
- starting status: exactly the five known unrelated untracked files;
- master plan: read completely and left unchanged;
- no specialist search, official Python game evaluation, PSRO, EditFormer, corpus work, push or submission.

The final commit and final status are reported after this file is committed.

## B. P0.4 reference controller

The exact manual oracle is `route437_wheat_scarcity_sheep_shift_v1`:

- 398 nodes, 26 BB slots, 0 NodeState slots;
- max depth 11 and static max visits 312;
- controller semantic hash `4e611d80f0bbb38bd696aa3dbb8a5e9e903be556e4ceafa233b5cb5685597be6`;
- combined identity `357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`;
- route-pack hash `068dca3988994778dc2c3d96b36a30b078b85f6ac329b91dc1a6fbf9d2a323e9`;
- strategic routes include accepted parent routes211..214 plus synthesized routes216,217,218;
- shared strategic authority is BB `day2_capital_specialist_v1_route_override`, compiled slot19.

The P0.5 declaration explicitly models route213, route216, route217 and route218. Earlier accepted parent mechanics remain an immutable validated parent structure.

## C. Manual composition being replaced

P0.2, P0.3 and P0.4 each deep-copied the previous giant module, appended one BOOL latch, appended one specialist subtree, appended one strategic route source and rewrote the module manifest. P0.4 additionally instantiated `ExactRoutePreemption` in its route437-specific distiller.

This required the implementation agent to choose the child position and manually wire the route213 latch clear. The new authoritative path reads those relationships from data. Historical distillers remain unchanged as reproduction evidence, but a future accepted macro no longer needs a route-specific composition function.

## D. Composer architecture

Permanent implementation:

- `tools/bkg4_search/bonsai_composer.py`: declaration loader, public API, static analyzer, deterministic generator and CLI;
- `tools/bkg4_search/test_bonsai_composer.py`: positive, negative, structural and CPU/CUDA tests;
- `bonsai-kg/integrations/bkg4_champion/compositions/p0_5_current_specialists_v1.json`: authoritative inputs;
- `p0_5_current_specialists_v1.manifest.json`: deterministic machine-readable conflict/output report;
- `...plus_automatic_composer_v1.json`: generated BKG4 module;
- `champions/bounded_automatic_composer_v1/`: compiled generated artifacts.

Public API:

```python
compose_specialists(parent, specialists, preemptions,
                    strategic_authorities=..., output_name=...)
compose_declaration(declaration, repo_root=...)
load_and_compose(path, repo_root=...)
```

The return value contains the composed module, canonical report, composition semantic ID and construction latency. P0.6 can construct a `SpecialistDeclaration` directly from a `BoundedRouteMacro` through `SpecialistDeclaration.from_macro`; it does not import a route-specific distiller.

CLI:

```text
python3 -m tools.bkg4_search.bonsai_composer \
  --declaration DECLARATION.json \
  --out-module MODULE.json \
  --out-manifest REPORT.json
```

`--validate-only` performs the same fail-closed validation without writing output. Errors return non-zero.

## E. Specialist declaration schema

Each specialist declares:

- `name`, `kind`, route ID and semantic ID;
- authorization step and inclusive active window;
- one owner-controlled BOOL latch;
- named strategic authorities;
- referenced route document;
- referenced bounded macro document for generated specialists;
- end policy;
- identity-free acceptance provenance;
- legal gate clauses for embedded authorities.

Example shape:

```json
{
  "name": "route437_wheat_scarcity_sheep_shift_v1",
  "kind": "bounded_route_macro",
  "route_id": 218,
  "authorization_step": 264,
  "active_window": [264, 287],
  "active_slot": "route437_wheat_scarcity_sheep_shift_v1_active",
  "authorities": ["strategic_route_override"],
  "macro_source": ".../route437_wheat_scarcity_sheep_shift_v1.json",
  "route_source": ".../route437_wheat_scarcity_sheep_shift_v1.json",
  "semantic_id": "3321dcaa...",
  "end_policy": "RETURN_TO_PARENT"
}
```

The macro payload is referenced rather than duplicated. Summary fields are deliberately repeated and checked against the referenced macro so stale declarations fail.

## F. Authority model

`StrategicAuthority` maps a logical authority name to one typed BB slot. Every declared specialist names its authority and its persistent active latch. The composer validates:

- authority slot existence and type;
- unique parent BB names;
- embedded-latch owner and BOOL type;
- new latch absence and ownership;
- route ID uniqueness against parent and peers;
- authorization/window equality with referenced macro;
- suppression target ownership.

Active windows, not only authorization steps, determine whether persistent latches can collide.

## G. Conflict detection

The report separates:

- `WINDOW_OVERLAP`: inclusive active ranges intersect;
- `AUTHORITY_CONFLICT`: declarations own a common strategic authority;
- `GATE_OVERLAP`: bounded legal conjunction analysis finds a satisfiable joint authorization state;
- `DECLARED_PREEMPTION`: an explicit evidence-backed edge resolves the authority collision.

Gate analysis reports `PROVEN_DISJOINT`, `POSSIBLE_OVERLAP`, `OBSERVED_OVERLAP` or `UNKNOWN`. It includes the implicit authorization-step equality. Missing gate data produces `UNKNOWN`, never a safe assumption.

For the current six declaration pairs, five are window-disjoint and gate-proven-disjoint. Route213/route218 has window overlap, shared authority, possible gate overlap and declared preemption.

## H. Fail-closed behavior

Composition rejects:

- unresolved overlapping authority owners, including unknown gates;
- missing preemptor or target;
- self-preemption and precedence cycles;
- conflicting pair directions or multiple suppression targets;
- a preemptor that is not a generated bounded macro;
- a precedence pair without the named shared authority or window conflict;
- mismatched declaration, macro or preemption windows;
- unsupported end/resume policies;
- missing/wrong authority or latch owner/type;
- duplicate BB names, specialist names, route IDs or semantic IDs;
- route documents that disagree with declarations;
- observed-overlap evidence contradicting proven-disjoint gates;
- identity-like specialist, composition or evidence provenance.

There is no best-effort or append-order fallback.

## I. Precedence model

The current rule is data:

```json
{
  "preemptor": "route437_wheat_scarcity_sheep_shift_v1",
  "preempted": "fert_wool_expansion_v1",
  "authority": "strategic_route_override",
  "overlap_window": [264, 287],
  "resume_policy": "BOTH_WINDOWS_END_TOGETHER",
  "evidence": {"kind": "exact_cuda_route_authority_conflict", "stage": "P0.4"}
}
```

The precedence graph is cycle-checked. Its edge, not route ID, name, creation date or input order, authorizes suppression. Route ID is only the deterministic tie-break for mutually disjoint generated nodes.

## J. Resume/end model

P0.5 supports only the policies required by evidence:

- bounded macros use `RETURN_TO_PARENT` when their inclusive window closes;
- preemption uses `BOTH_WINDOWS_END_TOGETHER` and requires identical windows.

Route213 and route218 both end at287. Route218 therefore clears the route213 latch when authorizing, owns the override through287, writes nothing from288, and requires no restoration stack. Parent routing resumes at288.

## K. Current composition declaration

| Specialist | Kind | Route | Authorization | Window | Authority |
|---|---|---:|---:|---|---|
| `fert_wool_expansion_v1` | embedded | 213 | 264 | 264..287 | strategic route override |
| `route435_pizza_glut_rebalance_v1` | bounded macro | 216 | 156 | 156..203 | strategic route override |
| `route410_wheat_liquidity_expansion_v1` | bounded macro | 217 | 120 | 120..137 | strategic route override |
| `route437_wheat_scarcity_sheep_shift_v1` | bounded macro | 218 | 264 | 264..287 | strategic route override |

The sole precedence edge is route218 over route213. Other accepted historical mechanics remain in the frozen parent unchanged.

## L. Generated output

The automatic result is:

- 398 nodes, 26 BB slots, 0 state slots;
- max depth 11, static max visits 312;
- controller semantic hash `4e611d80f0bbb38bd696aa3dbb8a5e9e903be556e4ceafa233b5cb5685597be6`;
- combined identity `357dbb9489b26a8fb424de592da92e45e2c0ac28d8a450a5e6e1b5791ade9d29`;
- route-pack hash `068dca3988994778dc2c3d96b36a30b078b85f6ac329b91dc1a6fbf9d2a323e9`;
- composition semantic ID `7d66b2f808437b24344d3c2132334540126c1fd34c84f3ed0e7abfe0650a823e`.

Only source metadata differs: automatic module name/manifest and generated identity provenance describe P0.5 instead of the historical P0.4 distiller. Executable artifacts do not differ.

## M. Structural equivalence

Automatic versus manual P0.4:

- blackboard declarations: exact;
- tree: exact;
- strategic route declarations/order: exact;
- runtime reconciler: exact;
- compiled `controller.bkgc`: byte-identical;
- compiled `routes.bkv6`: byte-identical;
- decompiled controller and controller manifest: exact;
- legacy route pack: exact;
- controller semantic hash and combined identity: exact.

No canonicalization exception was needed at the executable level.

## N. Game-level CUDA equivalence

The automatic controller was reevaluated on cert745 and archive766 with the exact P0.4 feed/strategic-slot settings:

- records: 1,511;
- CUDA games: 3,022, including the persisted baseline/forced pair per record;
- automatic result files versus manual P0.4 result files: byte-identical;
- terminal outcome mismatches: 0;
- margin mismatches: 0;
- activation/teacher mismatches: 0;
- route-decision mismatches in captured collision differential: 0;
- `bad`: 0.

Therefore the P0.4 result is reproduced exactly: 8 activations, 8 positive, 3 L->W, 0 harmful, 0 W->L, worst delta +1,129 and aggregate delta +31,923. All 969 protected wins remain unchanged.

## O. Inactive-parent equivalence

Of 1,511 records, 1,503 do not activate route218. Automatic output exactly matches the P0.4/manual parent reference on all 1,503. Mismatches: 0.

The remaining eight automatic cases exactly match the accepted forced route437 teacher. Thus the structural inactive-parent invariant remains intact.

## P. Route213/route218 collision

The synthetic frame makes both step264 gates true. Automatic composition:

1. evaluates the embedded route213 logic;
2. route218 authorization clears `fert_wool_expansion_v1_active`;
3. route218 sets its own latch and route218 authority;
4. route218 remains selected through287;
5. parent routing resumes at288.

The automatic and manual controllers produce identical decisions on every frame.

## Q. CPU/CUDA parity

The differential spans steps119..289 for both manual and automatic controllers:

- frames per controller: 171;
- total decision rows: 342;
- CPU versus CUDA mismatches: 0;
- manual versus automatic mismatches: 0;
- route217: active120..137, return138;
- route216: active156..203, return204;
- route213/218 collision: route218 owns264..287, return288.

## R. Determinism

Reversing the specialist input list produces the same:

- composition semantic ID;
- composition order;
- module object and serialized bytes;
- conflict report and serialized bytes.

Two consecutive CLI emissions are byte-identical. Filesystem iteration order is not used.

## S. Negative tests

Focused tests exercise unresolved overlap, reversed embedded precedence, self-preemption, cycles, missing targets, wrong BB owners, duplicate BB names, duplicate route IDs, duplicate semantic IDs, macro/window mismatch, preemption-window mismatch, unsupported resume policy, identity-like provenance, and gate-observation contradictions.

All fail closed with `ValueError`; the CLI exits non-zero.

## T. Tests

Commands and results:

```text
python3 -m unittest tools.bkg4_search.test_bonsai_composer -v
23/23 passed

python3 -m unittest \
  tools.bkg4_search.test_bounded_macro_cegis \
  tools.bkg4_search.test_synthesizer_repeatability \
  tools.bkg4_search.test_p0_4_third_specialist \
  tools.bkg4_search.test_bonsai_composer -v
52/52 passed

python3 -m unittest \
  tools.bkg4_search.test_regret_macro_search \
  tools.bkg4_search.edit_search.test_goals -v
13/13 passed
```

The CUDA fragment evaluator and information-state capture build targets are current.

## U. Composer performance

Current four-authority construction takes about 3 ms on this machine, excluding JSON serialization and BKG4 compilation. This is negligible compared with even the bounded CUDA equivalence run.

## V. Scaling interpretation

The data model and generator have no route-specific branches. BB/route/tree generation is linear in specialist count. The precedence graph uses adjacency sets. Pairwise conflict and gate analysis is O(n^2): trivial for 10 or 100 specialists and about 500K static pairs for 1,000 specialists, still reasonable for an offline construction gate.

The real scaling constraints are declaration/evidence quality, tree/compiler limits, pairwise conflict coverage and exact regression cost, not Python construction time. If thousands of specialists make pairwise analysis material, authority/window indexing can reduce candidate pairs without changing the schema.

## W. P0.6 readiness

**READY.** A future end-to-end synthesizer can provide a validated route document and `BoundedRouteMacro`, create a `SpecialistDeclaration`, add any evidence-backed preemption rules, and ask this composer to construct a fail-closed BKG4 candidate. It does not edit tree children, allocate BB slots, choose append positions or write latch-clear nodes.

The bounded contract treats the historical parent as already validated. When a new macro needs to arbitrate with a historical embedded authority, that authority must be exposed in the declaration just as route213 is now; absence of a required declaration is not permission to invent scheduling behavior.

## X. Remaining manual steps

P0.5 does not automate P0.6's upstream/downstream loop. An agent or future driver must still:

- discover and transfer-test a macro;
- synthesize/freeze its legal gate;
- emit the macro and route documents;
- declare evidence-backed conflicts with exposed parent authorities;
- run falsification, CUDA certification and promotion accounting;
- choose whether a validated candidate enters research lineage.

No manual controller surgery remains for the current bounded composition model. Complex resume stacks, multiple simultaneous suppression targets and resource-commitment scheduling remain deliberately unsupported and fail closed.

## Y. Next stage

Proceed to P0.6: the bounded end-to-end autonomous synthesis driver connecting the existing loss-frontier, regret-search, transfer, CEGIS, automatic-composer, falsification and certification APIs. Do not add a fourth specialist manually before that driver exists, and do not expand into PSRO or EditFormer.

## Z. Commit

All permanent P0.5 composer code, tests, declaration/report, generated module/assets, ledger evidence and this report are committed locally. The exact SHA and message are reported after the commit.

No push or Kaggle submission is part of P0.5.

## AA. Final git status

P0.5 CUDA outputs remain under ignored `scratch/p0_5_automatic_composer/`. The five unrelated pre-existing untracked files remain untouched and excluded. Exact final status is reported after the commit.

## AB. Final recommendation

Accept `bounded_automatic_composer_v1` as the authoritative reusable composition path. It reproduces P0.4 exactly, makes the route218-over-route213 relationship declarative, and rejects unproven authority interactions. Begin P0.6 against this API without widening P0.5 into a generic scheduler.
