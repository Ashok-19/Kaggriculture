# Kaggriculture RLRMB Phase 1 — Binary Specialist/PARENT Selective Routing Master Plan

**Team:** ReCurSiON  
**Repository:** `/home/nnmax/Desktop/kaggle/kaggriculture`  
**Branch:** `main`  
**Starting hardened HEAD:** `b8dc9c5`  
**Architecture:** Bonsai / BKG4  
**Phase objective:** Safely recover high-value specialists that are globally regressive by activating them only inside legal state regions where they outperform the current BKG4/PARENT.

---

# 1. Motivation

Current RLRMB promotion effectively favors specialists that are individually safe when evaluated broadly.

Phase-0.1 showed that this is unnecessarily restrictive.

Local evidence:

```text
560 archive entries
27 with TARGET L→W > 0
16 useful imperfect specialists analyzed

16/16 caused protected W→L when globally activated

15/16 were BINARY_PARENT_PLAUSIBLE
1/16 TRUE_FUTURE_DEPENDENT

0 candidates demonstrated that a multi-specialist portfolio
was actually required
```

Therefore the next architecture should not begin with a general waitlist or micro-portfolio system.

The immediate problem is:

> A specialist may be extremely strong in a narrow legal state regime and extremely harmful elsewhere.

For Bonsai, this is acceptable if routing can identify the useful regime.

Phase 1 changes the promotion unit from:

```text
globally activated specialist
```

to:

```text
specialist
+
legal selective activation gate
+
PARENT fallback
```

The final composed BKG4 must remain zero-regression.

---

# 2. Core architectural principle

For every useful specialist `S`, Phase 1 attempts to synthesize:

```text
if SAFE_USEFUL_REGION(state):
    route to S
else:
    continue with PARENT
```

The PARENT is always the current committed BKG4.

A specialist's protected W→L states therefore become:

```text
states where S must NOT activate
```

rather than automatically becoming:

```text
states another specialist must repair
```

This is the main architectural change.

---

# 3. Non-negotiable invariants

Phase 1 may relax **candidate-level** safety.

It must never relax **final composed-tree** safety.

The following remain mandatory:

```text
final protected W→L = 0
semantic mismatches = 0
CPU/CUDA/BKG4 semantics identical
SEALED never influences synthesis/search
PARENT fallback remains exact
production commits remain atomic
route IDs remain collision-free
restart/resume deterministic
SearchMemory context boundaries remain correct
```

Existing zero-W→L specialists retain their current fast path.

No micro-portfolio architecture is implemented in Phase 1.

---

# 4. Candidate classes

Phase 1 should use a small, explicit classification.

## 4.1 SAFE_FAST_PATH

```text
TARGET capability > 0
protected W→L = 0
```

Use the existing production path.

Do not make safe specialists pay the cost of binary-routing search unnecessarily.

---

## 4.2 BINARY_ROUTE_CANDIDATE

A specialist with:

```text
TARGET L→W > 0
protected W→L > 0
```

but with evidence that useful and protected-breaking regimes are legally separable.

This becomes the primary new Phase-1 candidate class.

---

## 4.3 GATE_LIMITED

Full legal state can distinguish useful from harmful cases, but the currently deployable gate grammar cannot express the separator.

Do not silently promote.

Persist as unresolved evidence.

Phase 1 may support only narrowly justified gate-grammar improvements if they are:

```text
already legal BKG4 features
deterministic
CPU/CUDA reproducible
bounded
auditable
```

Do not introduce learned or opaque routing.

---

## 4.4 DEFERRED_ROUTE_CANDIDATE

The necessary distinction becomes observable later, but before the specialist's harmful behavior must diverge.

Do not implement full deferred/reactive routing in the first Phase-1 slice.

Preserve this class for a later Phase-1 extension only if binary-at-initiation results justify it.

---

## 4.5 TRUE_FUTURE_DEPENDENT

Correct behavior must diverge before legal state can distinguish the cases.

Reject.

Additional specialists do not solve missing information.

---

# 5. Candidate admission

Do not send every candidate with one positive delta through expensive routing synthesis.

A Phase-1 binary candidate must have meaningful capability.

Initial admission should remain deterministic and vector-valued.

Useful signals:

```text
TARGET L→W count
TARGET positive support
TARGET worst delta
TARGET harm count
independent target support
both-seat support
structural/effect novelty
protected W→L count
protected harm severity
routing separability
routing complexity estimate
```

Do not collapse everything into one opaque score if existing lexicographic ranking works.

Suggested hard minimum:

```text
TARGET L→W > 0
semantic valid
effect evidence complete
no semantic mismatch
not TRUE_FUTURE_DEPENDENT
```

Then use bounded ranking to choose which imperfect candidates deserve expensive protected capture + CEGIS.

---

# 6. Evidence model

Each binary-route candidate needs a durable evidence package.

## TARGET positive evidence

Cases where the specialist creates capability:

```text
baseline loss -> specialist win
```

Also retain positive-margin improvements where useful.

## TARGET harmful evidence

Negative TARGET deltas remain useful diagnostics but are not equivalent to protected W→L.

## PROTECTED BREAK evidence

Hard negatives:

```text
baseline win
candidate loss
```

These are the most important binary-routing exclusion cases.

## PROTECTED HARM evidence

Negative margin delta without W→L.

Keep separate from BREAKS.

## Legal state evidence

At the routing decision boundary persist:

```text
full legal state/context
deployable gate context
case role
TARGET-positive vs PROTECTED-break label
first required action divergence
first legal divergence if known
```

This should reuse the Phase-0.1 diagnostic infrastructure rather than creating parallel representations.

---

# 7. Binary routing synthesis objective

The routing problem is:

```text
positive examples:
    TARGET capability cases where S should activate

hard negative examples:
    PROTECTED BREAK cases where S must not activate

default:
    PARENT
```

The synthesized gate should maximize preserved capability while satisfying hard safety constraints.

Conceptually:

```text
maximize:
    TARGET L→W preserved
    independent transfer support
    both-seat support
    margin improvement

subject to:
    zero observed protected BREAK activation
    legal bounded grammar
    route complexity limits
```

PARENT is the reject/default branch.

---

# 8. Router representation

Phase 1 should initially reuse the simplest representation compatible with the current BKG4 machinery.

Preferred initial form:

```text
bounded deterministic gate
→ SPECIALIST
else
→ PARENT
```

Do not introduce a general multiclass router yet.

The current bounded DNF / `ConditionGene` machinery is acceptable as the first synthesis grammar.

If current grammar proves insufficient in measured cases, extend it only with already-legal features supported identically in CPU/CUDA/BKG4.

Do not expand grammar speculatively.

---

# 9. CEGIS redesign for binary routing

Existing hardened CEGIS should remain the foundation.

The main change is the evidence supplied to it.

Current conceptual use:

```text
positive candidate cases
vs harmful candidate cases
```

Phase 1 should explicitly formulate:

```text
POSITIVE:
    TARGET L→W / required capability

NEGATIVE:
    protected W→L BREAK states

DEFAULT:
    PARENT
```

CEGIS loop:

```text
candidate gate
    ↓
compose S + PARENT
    ↓
evaluate actual routed behavior
    ↓
counterexample?
    ├─ yes → refine gate
    └─ no  → continue downstream
```

Critically:

> CEGIS must evaluate the actual routed composition, not merely classify captured states.

This avoids false confidence from static separation.

---

# 10. Prevent TARGET_CAPABILITY_LOST

The previous pipeline demonstrated that an overly narrow gate can become safe by deleting the specialist's useful behavior.

That is not success.

Binary synthesis must explicitly preserve capability.

At minimum require:

```text
required TARGET L→W >= configured threshold
```

Prefer preservation of all high-confidence target capabilities where possible.

Gate comparison should distinguish:

```text
SAFE BUT USELESS
```

from:

```text
SAFE AND CAPABILITY-PRESERVING
```

`TARGET_CAPABILITY_LOST` remains a valid rejection.

---

# 11. Actual routed evaluation

Once a gate is synthesized:

```text
materialize temporary routed specialist
→ actual CUDA evaluation
```

The evaluation must cover:

## TARGET

Confirm intended L→W survives.

## TRANSFER

Confirm specialist capability is not target-only overfit.

## VALIDATION / protected panel

Confirm actual routed composition causes:

```text
protected W→L = 0
```

and inspect harmful margin changes.

Do not infer this from isolated EffectVectors.

---

# 12. Re-capture states after routing

A routed specialist changes which trajectories reach later states.

Therefore, after a candidate gate appears viable:

```text
run actual routed composition
→ capture legal states again
```

Do not rely indefinitely on captures produced under globally activated specialist behavior.

This recapture is particularly important before final CEGIS convergence.

---

# 13. PARENT fallback correctness

Fallback behavior must be exact.

For every state where specialist gate is false:

```text
composed behavior == current committed BKG4/PARENT behavior
```

Add an explicit fallback audit.

No accidental route override.
No stale latch.
No partial specialist action.
No blackboard contamination.

This should be treated as a hard invariant.

---

# 14. Specialist active-window semantics

A specialist should not be activated merely because its gate was true once if doing so creates behavior outside its certified regime.

For Phase 1:

```text
decision point
+
bounded active window
```

must remain explicit.

Use existing `BoundedRouteMacro` semantics where compatible.

The router should select the specialist only at its certified initiation point/window.

Do not create unrestricted persistent policy switching.

---

# 15. Routing complexity limits

Avoid replay-memorizing gates.

Initial gate complexity should remain deliberately small.

Use existing bounds unless evidence requires changes:

```text
bounded number of clauses
bounded conditions per clause
legal current-state features only
```

Gate complexity should be included in candidate ranking.

Prefer:

```text
simple broad regime
```

over:

```text
many tiny exact-state exceptions
```

---

# 16. Overfitting defenses

Binary routing introduces a new overfitting risk.

Mandatory defenses:

```text
no replay identity
no opponent identity
no seed identity
no submission/team identity
no inaccessible future state
```

Use:

```text
cross-opponent support
both-seat support
transfer evidence
threshold perturbation
legal boundary perturbation
held-out validation
SEALED
```

One-case exact-equality gates should receive additional scrutiny unless the equality is structurally meaningful.

---

# 17. Gate-grammar limitation handling

Phase-0.1 established:

```text
full legal observability: 58 features
deployable ConditionGene grammar: 27
```

Phase 1 should preserve this distinction.

If:

```text
full legal state separates
but gate grammar does not
```

classify:

```text
GATE_LIMITED
```

Do not mislabel as FUTURE_DEPENDENT.

A later bounded grammar extension may use legal features such as worker/tile state only when actual candidate evidence demonstrates material value.

Every added feature must preserve CPU/CUDA exact parity.

---

# 18. SearchMemory

Extend existing memory minimally.

Persist binary-route candidate evidence under the correct context.

## Exact current-context

Bind to current BKG4 generation/evaluation context:

```text
candidate effect
TARGET positive cases
protected BREAK cases
routing captures
gate attempts
CEGIS counterexamples
actual routed evaluations
final binary-route disposition
```

## Structural

Reusable:

```text
program semantics
routing-feature usage
gate structure
candidate family
lane
checkpoint class
```

## Historical prior

Reusable only as non-authoritative guidance:

```text
this motif/family often binary-routes successfully
this legal feature often separates this routing regime
```

Never reuse an old gate as safety proof against a changed parent.

---

# 19. Adaptive allocator integration

Do not redesign the allocator immediately.

Initially add only transparent signals:

```text
binary-route candidate yield
binary-route success rate
TARGET capability preserved
protected BREAKs eliminated
CUDA cost per committed binary specialist
GATE_LIMITED rate
TRUE_FUTURE_DEPENDENT rate
```

Once enough history exists, the allocator can shift budget toward regions producing useful binary-route candidates.

Maintain a guaranteed safe-specialist discovery floor.

---

# 20. Unresolved-candidate archive

Phase 1 should preserve valuable failures without implementing the full micro-portfolio system.

A small durable unresolved archive can contain:

```text
candidate semantic ID
parent generation
TARGET capability
protected BREAK set
routing epoch
classification
reason binary routing failed
```

Possible dispositions:

```text
GATE_LIMITED
DEFERRED_ROUTE_CANDIDATE
PORTFOLIO_FUTURE_CANDIDATE
TRUE_FUTURE_DEPENDENT
```

Do not recursively search complement chains in Phase 1.

This archive becomes the evidence base for Phase 2.

---

# 21. Existing safe specialist path

Keep this path unchanged:

```text
candidate
→ zero protected regression
→ current routability
→ current CEGIS
→ transfer
→ validation
→ SEALED
→ materialize
→ compose
```

Binary routing is an additional path, not a replacement.

---

# 22. Multi-accept compatibility

The hardened multi-accept pipeline must remain authoritative.

A generation may contain:

```text
safe specialists
+
binary-routed specialists
```

If several candidates pass independently:

```text
compose deterministic compatible set
→ validate actual generation
→ atomic commit
```

Do not assume independent binary safety implies combined safety.

The final multi-specialist generation must be revalidated after composition.

---

# 23. Generation-level final validation

Before committing a generation containing any Phase-1 binary specialists:

```text
compose all intended accepted specialists
→ actual TARGET/TRANSFER/VALIDATION evaluation
→ protected W→L = 0
→ semantic mismatches = 0
→ compiler/resource checks
```

If composition introduces a regression:

```text
commit none of the conflicting proposed composition
```

or use the already hardened deterministic compatibility handling.

Never partially mutate committed state.

---

# 24. SEALED firewall

SEALED remains final and write-only.

It must not affect:

```text
candidate selection
gate synthesis
threshold selection
routing feature selection
repair search
allocator decisions
```

After SEALED failure:

```text
reject candidate/composition
```

Do not adapt to SEALED counterexamples.

---

# 25. CPU/CUDA/BKG4 parity

Every new router/gate feature must have exact semantics in:

```text
CUDA search engine
CPU C++ engine
materialized BKG4
```

Before any large campaign, focused parity tests must cover:

```text
gate true
gate false
boundary thresholds
indexed features
active-window start/end
PARENT fallback
latch reset
restart/resume
```

Zero semantic mismatch remains mandatory.

---

# 26. Implementation phases

## Phase 1.1 — Candidate admission path

Goal:

Allow useful protected-regressive candidates to survive long enough for binary-routing analysis.

Implement:

```text
binary-route candidate shortlist
protected evidence
classification
```

Do not synthesize new routing yet.

Success condition:

Useful imperfect candidate reaches Phase-1 path while current safe promotion remains unchanged.

---

## Phase 1.2 — Binary gate synthesis

Implement candidate-vs-PARENT gate synthesis using:

```text
TARGET positives
protected BREAK negatives
```

Reuse hardened CEGIS where possible.

Success condition:

Synthetic fixtures demonstrate:

```text
specialist globally unsafe
→ gate activates only useful regime
→ PARENT elsewhere
```

---

## Phase 1.3 — Actual routed CUDA evaluation

Materialize/evaluate temporary routed composition.

Verify:

```text
TARGET capability preserved
protected W→L eliminated
PARENT fallback exact
```

Success condition:

At least one Phase-0.1 representative candidate can be evaluated through the real binary route path.

---

## Phase 1.4 — Full downstream integration

Integrate:

```text
TRANSFER
VALIDATION
full routability
CEGIS
SEALED
materialization
composer
```

Success condition:

Binary-route candidates use the same final safety firewall as existing safe specialists.

---

## Phase 1.5 — Durable memory / restart

Persist:

```text
binary-route evidence
gate attempts
dispositions
accepted route
```

Verify restart/resume does not duplicate CUDA work or route IDs.

---

## Phase 1.6 — Multi-generation integration

Run a tiny bounded local multi-generation fixture.

Verify:

```text
safe + binary specialists coexist
route IDs remain unique
atomic multi-accept works
protected W→L remains zero
```

---

## Phase 1.7 — Local CUDA capability probe

Only after implementation passes focused validation.

Use a small CUDA search.

Measure:

```text
useful imperfect candidates
binary attempts
binary successful gates
TARGET capability preserved
protected W→L eliminated
committed binary specialists
```

This is the first major go/no-go gate.

---

## Phase 1.8 — Official Python canary before champion promotion

If a candidate generation is intended for actual champion promotion:

```text
64 or 128 games
official Python engine
```

Confirm C++/CUDA results remain true.

Do not perform this for every internal synthetic unit test.

---

# 27. Phase-1 success metrics

The main metric is not GPU utilization.

It is:

> additional safely committed strategic capability recovered from candidates that the old pipeline rejected.

Track:

```text
binary candidates attempted
binary gates synthesized
binary gates preserving capability
binary candidates reaching SEALED
binary specialists committed

new L→W from binary specialists
protected W→L after routing
protected W→L avoided versus global activation

safe specialists committed
binary specialists committed
CUDA games / committed specialist
nodes / new solved regime
```

A strong Phase-1 result would demonstrate that globally regressive specialists can regularly become zero-regression Bonsai subtrees through selective PARENT fallback.

---

# 28. Failure criteria

Phase 1 should be reconsidered if real evidence shows:

```text
most binary-plausible candidates lose all TARGET capability after gating
or
protected W→L persists after actual routed simulation
or
gates overfit validation/SEALED heavily
or
binary routing costs vastly more than specialist discovery
```

Do not force the architecture if the actual conversion rate is poor.

---

# 29. Phase-2 boundary

Micro-portfolios are explicitly deferred.

Phase 2 should only begin when Phase 1 has accumulated real evidence such as:

```text
A cannot safely route against PARENT
B cannot safely route against PARENT
but A and B have complementary routing regimes
and a shared routing epoch can plausibly choose:
PARENT / A / B
```

Phase 1 must therefore preserve enough unresolved evidence to quantify this later.

Do not implement:

```text
A/B/PARENT multiclass routing
portfolio beam search
regression-debt chains
compatibility hypergraphs
```

during Phase 1.

---

# 30. Long-term architecture

The intended progression remains:

```text
PHASE 1
Selective SPECIALIST/PARENT routing
    ↓
large increase in usable specialist supply

PHASE 2
Bounded routing-epoch micro-portfolios
only where binary routing is insufficient
    ↓
broader strategic coverage

PHASE 3
Whole-Bonsai adversarial refinement
routing repair
subtree split/merge
edge-case repair
counterexample-guided edits
L→W maximization
    ↓
ultimate BKG4
```

---

# Final Phase-1 architecture

```text
                     evaluated specialist
                            |
                  TARGET capability > 0?
                            |
                           yes
                            |
                  protected evaluation
                            |
              +-------------+-------------+
              |                           |
        protected W→L = 0           protected W→L > 0
              |                           |
       SAFE FAST PATH             binary-route analysis
                                          |
                          +---------------+----------------+
                          |                                |
                       routeable                     not routeable
                          |                                |
                specialist-vs-PARENT              classify/persist
                     gate CEGIS               GATE_LIMITED / DEFERRED /
                          |                     FUTURE_DEPENDENT
                actual routed CUDA
                          |
                 TARGET preserved?
                          |
                         yes
                          |
                 protected W→L = 0?
                          |
                         yes
                          |
              TRANSFER → VALIDATION
                          |
                   full routability
                          |
                    hardened CEGIS
                          |
                        SEALED
                          |
                  materialize route
                          |
                 compose generation
                          |
              generation-level parity
                          |
                protected W→L = 0
                          |
                     ATOMIC COMMIT
```

## Phase-1 completion criterion

Phase 1 is complete when the production RLRMB pipeline can repeatedly:

> discover a specialist that is unsafe when globally activated, synthesize a deterministic legal SPECIALIST/PARENT routing gate, preserve its useful L→W capability, eliminate all protected W→L in the actual composed BKG4, pass the existing downstream safety firewall, and commit it atomically without regression.

That is the next major Bonsai capability milestone.