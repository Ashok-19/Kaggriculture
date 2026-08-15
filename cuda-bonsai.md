Yes — **this is a genuinely promising direction for Kaggriculture**, but the reason is slightly different from “Bonsai replaces RL.”

I cloned and inspected the repository completely enough to understand its execution model, tests, examples, blackboard semantics, reactive nodes, and real-world references. The clone is at:

```text
/home/nnmax/Desktop/kaggle/kaggriculture/bonsai
```

Pinned upstream commit:

```text
8fe964fca85d584c7575fcba2754ff2e9f6fd031
```

The nested repo is clean. I also ran the Bonsai core test suite with two build workers: **55/55 tests passed**.

## What Bonsai actually is

Bonsai is a **behavior-tree engine** written in Rust, with Python bindings. Its own description is “behavior trees for deterministic AI.” ([GitHub][1])

It is not a neural network, not reinforcement learning, and not a planner that magically discovers intelligent behavior.

The core abstraction is:

```text
                     Behavior Tree
                          │
                    tick every turn
                          │
              ┌───────────┴───────────┐
              │                       │
          conditions                actions
              │                       │
      Success / Failure       Success / Failure /
                              Running
              │                       │
              └───────────┬───────────┘
                          │
                parent chooses what
                   to execute next
```

The useful nodes include:

```text
Sequence
    Do A, then B, then C.
    Fail immediately if one fails.

Select
    Try A.
    If A fails, try B.
    If B fails, try C.

If
    condition ? A : B

Invert
AlwaysSucceed

While / WhileAll

WhenAll
WhenAny
Race
After
```

And critically, the current commit you cloned added:

```text
Sequence(...).memory(false)
Select(...).memory(false)
```

These are **reactive / memoryless nodes**.

That addition is particularly relevant to us.

---

# Reactive selectors are the interesting part

Consider:

```text
MemorylessSelector
├── EmergencyRecovery
├── HighValueMarketOpportunity
├── ExploitOpponentState
├── ContinueCurrentLane
└── SafeFallback
```

Every Kaggriculture turn it starts again at the top.

Suppose `ContinueCurrentLane` has been running for 100 turns.

Then suddenly:

```text
WOOL price = 180
```

and `HighValueMarketOpportunity` becomes applicable.

On the **next turn**, the tree can immediately preempt the lower-priority continuation.

That is exactly what the repository's `memoryless_chase` example demonstrates: a lower-priority Chase action runs until the higher-priority Attack condition becomes true, at which point Attack preempts it. This behavior comes directly from the implementation we inspected.

That gives us:

```text
observe
  ↓
re-evaluate priorities
  ↓
highest valid strategy wins
  ↓
act
  ↓
next turn
  ↓
re-evaluate everything
```

rather than:

```text
if ...
elif ...
if some_global_flag ...
if step == ...
if another_global_flag ...
...
```

which is increasingly what v32 has become.

---

# Bonsai does NOT contain the intelligence

This distinction matters.

In the FPS example, Bonsai does not somehow learn how to aim, navigate or fight.

A leaf looks conceptually like:

```text
EnemyVisible?
EnemyInRange?
FindPath
MoveTowardEnemy
AttackEnemy
Evade
```

**The developer implements those functions.**

Bonsai decides how those components interact.

The FPS use is real in the sense that the upstream project explicitly points to `bp-ort`, a Northstar plugin implementing Titanfall 2 player bots. The bot project describes navigation, 3D pathfinding, combat, evasion and related systems. ([GitHub][2])

But I would not interpret the demo as:

> “Bonsai independently figured out how to beat FPS players.”

It is:

> “Complex authored game-playing capabilities were organized into a responsive autonomous agent using behavior-tree control.”

That is still significant for us.

---

# The NASA example is also real — with the same distinction

Bonsai's own examples documentation points to the University of Utah Lunabotics software as a real-world use of the library.

The Utah robot's AI layer maintains a blackboard including robot state, obstacle information and other observations, tracks what the robot is doing, and decides motor/actuator behavior from that information. ([GitHub][3])

NASA confirms that the University of Utah finished **third in the Caterpillar Autonomy Award** at the 2026 Lunabotics competition. The competition had 47 qualifying teams, 10 finalists, and 27 fully autonomous robots. ([NASA][4])

So this isn't just a toy library. NASA describes Lunabotics as a real prototype off-world construction-robot challenge with autonomy as part of the evaluation. ([NASA][5])

Again:

```text
Bonsai ≠ perception
Bonsai ≠ localization
Bonsai ≠ path planning
Bonsai ≠ motor control
```

It orchestrates those capabilities into a coherent autonomous controller.

I couldn't independently resolve the exact Reddit shortlink you supplied through the available browser tooling, so I won't claim details from that particular post that I couldn't inspect. The upstream Titanfall and NASA references themselves are independently verifiable.

---

# Now the important part: Kaggriculture

This is where I think your idea becomes considerably more interesting.

Our current v32 is **already effectively a manually flattened behavior tree**.

Look at what we've built:

```text
                         ROOT
                           │
              detect opening/public lane
                           │
        ┌──────────────────┼───────────────────┐
        │                  │                   │
      W14                 H4                 Q17
        │                  │                   │
    sublanes          donor route       capital sublanes
        │
 ┌──────┼────────┐
 │      │        │
Arda   SHM      normal
 │
market controllers
 │
wool debt
 │
milk debt
 │
strawberry reserve
 │
fallback
```

The current source instead expresses this through things such as:

```text
_USE_Q8
_W14T
_H4
_Q8_17
_W14_ARDA
_W14_SHM
_W14_CONT
_WOOL_DEBT
_MILK_DUE
_STRAW_DEBT
...
```

plus hundreds of lines of `if` logic.

That's behavior-tree logic without an explicit behavior tree.

---

# A Bonsai-style v34 could look like this

For example:

```text
MemorylessSelector ROOT
│
├── Sequence
│   ├── IsCatastrophicState?
│   └── EmergencyRecovery
│
├── Sequence
│   ├── IsPETLane?
│   └── PET_TREE
│
├── Sequence
│   ├── IsICELane?
│   └── ICE_TREE
│
├── Sequence
│   ├── IsFourHandLane?
│   └── FOUR_HAND_TREE
│
└── GeneralMetaFallback
```

Then PET:

```text
PET_TREE

MemorylessSelector
│
├── Sequence
│   ├── IsDangerousMarketState?
│   └── MarketRecovery
│
├── Sequence
│   ├── SecondShopIsPizza?
│   ├── StepAtLeast144?
│   └── PET_PIZZA_CONTINUATION
│
├── Sequence
│   ├── StepAtLeast120?
│   ├── IsKawContinuationSafe?
│   └── KAW_CONTINUATION
│
└── PET_BASE
```

ICE might have a completely different subtree.

This is already much cleaner than adding another 30 conditional patches to v32/v33.

---

# Blackboard is extremely useful for us

Bonsai includes a **blackboard**, shared working memory for the tree.

For Kaggriculture ours might contain:

```text
first_shop
second_shop

lane_commitment

deferred_melon
deferred_wool
deferred_milk

previous_public_market_state

market_regime

last_selected_expert

risk_state

history-derived public features
```

So instead of globals such as:

```text
_WOOL_DEBT[seat]
_MILK_DUE[seat]
_STRAW_DEBT[seat]
```

we'd have an explicit policy state:

```text
Blackboard
├── wool_debt
├── milk_debt
├── strawberry_debt
├── active_lane
├── lane_phase
└── ...
```

One implementation detail I found in the source is important:

**`reset_bt()` resets the behavior-tree execution state but deliberately does not reset its blackboard.**

So for Kaggriculture we'd explicitly reset the policy blackboard at `step == 0`.

---

# But this alone doesn't give us RL-level strength

If we manually create:

```text
if wool_price > 80:
    SellWool
else:
    ContinueKaw
```

then Bonsai has merely made the code cleaner.

That is useful, but not a jackpot.

The jackpot becomes possible when we combine Bonsai with what we just built:

# **Bonsai-style policy representation + CUDA policy search**

This is the part I think is potentially more appropriate for Kaggriculture than training an RL policy.

Think of Bonsai as our **policy language**.

Then use the exact CUDA simulator as the optimizer.

---

# Instead of training weights, search programs

Suppose we define legal primitives.

### Conditions

```text
StepGE(x)

FirstShopIs(PET)
SecondShopIs(PIZZA)

OwnMoneyGE(x)
OpponentMoneyGE(x)

PriceGE(WOOL, x)
PriceLE(EGG, x)

MarketInventoryGE(TOMATO, x)

OwnShedGE(MELON, x)

OwnWorkersGE(x)

ShopUnlocked(YARN)

MarketRegime(x)

BlackboardDebtGE(WOOL, x)
```

### Expert/action leaves

```text
V32_BASE

META_CONTINUATION

Q8_CONTINUATION

KAW_CONTINUATION

ANIK_CONTINUATION

OFA_CONTINUATION

JASON_CONTINUATION

MARKET_SELL_RANK

DEFER_WOOL

RELEASE_WOOL

BUY_MELON

SAFE_RECOVERY
```

### Structure

```text
Sequence
Selector
MemorylessSelector
If
Invert
AlwaysSucceed
```

Now a candidate policy itself becomes data.

Something like:

```text
SELECT
    SEQUENCE
        PriceGE(WOOL, 83)
        ReleaseWool

    SEQUENCE
        StepGE(144)
        SecondShopIs(PIZZA)
        BaseContinuation

    SEQUENCE
        StepGE(120)
        KawContinuation

    PetBase
```

---

# CUDA can search those trees

This is where everything connects.

Instead of PPO:

```text
network
→ gradient
→ rollout
→ gradient
→ rollout
→ ...
```

we do:

```text
             population of behavior trees
                         │
                         ▼
                 exact CUDA engine
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
     thousands        thousands         both seats
      of seeds        opponents
        │                │                 │
        └────────────────┼─────────────────┘
                         ▼
                     fitness
                         │
                         ▼
            mutate / splice / select
                         │
                         └──── repeat
```

That is **program synthesis / genetic programming / policy search**, rather than RL.

And this is where I think your intuition could turn into something very strong.

---

# Example

Start with:

```text
PET TREE #0001

Selector
├── KawAfter120
└── Base
```

Generate mutations automatically:

```text
#0002
Kaw after 116

#0003
Kaw after 120

#0004
Kaw after 124
```

CUDA evaluates all.

Then winners mutate:

```text
#0137

Selector
├── Sequence
│   ├── Step>=120
│   ├── WoolPrice>=83
│   └── Kaw
│
└── Base
```

Later:

```text
#2189

Selector
├── Sequence
│   ├── EmergencyInventoryPressure
│   └── RecoveryExpert
│
├── Sequence
│   ├── Step>=120
│   ├── SecondShop=PIZZA
│   └── BasePlusMelonRestore
│
├── Sequence
│   ├── Step>=120
│   └── Kaw
│
└── Base
```

Then:

```text
#173,842
```

could be a tree no human would realistically have assembled manually.

But unlike a neural network, **we can inspect exactly why it makes every decision**.

---

# We can search more than thresholds

Mutation operators could include:

```text
threshold mutation
    80 → 84

step mutation
    120 → 116

expert replacement
    Kaw → Jason

condition addition

condition deletion

subtree insertion

subtree deletion

priority reorder

subtree crossover

market-controller parameter mutation

reserve-size mutation

debt-window mutation

lane split

lane merge
```

This is a large policy space.

Our CUDA engine makes evaluating it feasible.

---

# And this addresses the exact problem we have with v33

Recall our PET result:

```text
simple switch120
recorded: excellent

holdout:
36/50
```

Then hierarchy:

```text
40/50
```

The problem isn't that we have no good continuations.

We have many.

The problem is:

> **Which continuation should be active under which observable state?**

That's almost the canonical behavior-tree problem.

Instead of manually deciding:

```text
if second shop PIZZA:
    ...
elif this price:
    ...
elif this inventory:
    ...
```

we can **search the routing tree itself**.

---

# Bonsai may therefore be more useful than RL for our immediate problem

RL is powerful when the system must learn features and smooth decision boundaries from huge state spaces.

But Kaggriculture has several properties favoring structured policy search:

```text
only 719 turns

discrete actions

strong economic structure

a large amount of deterministic behavior

public shop milestones

many highly informative low-dimensional variables

strong existing expert agents

elite replay continuations

known strategic phases

an exact simulator

cheap snapshot cloning

and now very fast CUDA evaluation
```

Most importantly, **we are not starting from zero**.

We already have:

```text
v32
META
Q8
OFA
Kaw
Anik
Jason
28 donor continuations
live-loss replays
PET/ICE discoveries
market controllers
```

An end-to-end RL policy would have to rediscover a large amount of this.

A tree-search system can immediately use all of them as **expert leaves**.

That changes the problem from:

> learn Kaggriculture

to:

> learn how to combine our already-good Kaggriculture strategies.

The second problem is dramatically easier.

---

# This is very similar to a Mixture-of-Experts controller

Conceptually:

```text
                    OBSERVATION
                         │
                         ▼
                  Behavior Tree
                         │
        ┌────────────────┼────────────────┐
        │                │                │
      Expert A         Expert B         Expert C
       Kaw              Anik             META
        │                │                │
        └────────────────┼────────────────┘
                         │
                     ACTION
```

Except the router isn't a neural gating network.

It is a **learned/search-optimized deterministic program**.

That gives us some excellent properties:

```text
interpretable
deterministic
easy to distill
easy to debug
public-state legality auditable
tiny inference cost
no numerical NN dependencies
no distributional neural surprises
easy Kaggle submission
```

---

# It can also do something our current router struggles with: preemption

Suppose the active plan is:

```text
KawContinuation
```

but suddenly our inventory fills dangerously.

A reactive tree:

```text
MemorylessSelector
├── EmergencyShedPressure
├── HighPriceLiquidation
├── CurrentExpert
└── SafeFallback
```

will reconsider from the top every turn.

So:

```text
turn 400:
CurrentExpert

turn 401:
CurrentExpert

turn 402:
shed pressure becomes dangerous

turn 403:
EmergencyShedPressure preempts CurrentExpert
```

When the emergency resolves:

```text
turn 406:
back to CurrentExpert
```

This is substantially more natural than the persistent flags we currently accumulate.

---

# We can still have commitments

Not every decision should be reactive.

Sometimes we specifically **do not want** the tree changing strategy every turn.

Bonsai supports ordinary stateful `Sequence`/`Select` too.

So we can represent:

```text
hard strategic commitment
```

with memoryful nodes, and:

```text
market opportunity / safety overrides
```

with memoryless nodes.

That distinction is very powerful.

For example:

```text
PET_PHASE

Sequence                   ← stateful strategic plan
│
├── Opening
│
├── EstablishProduction
│
└── MemorylessSelector     ← reactive tactical layer
    ├── Emergency
    ├── MarketOpportunity
    ├── OpponentExploit
    └── ContinuePlan
```

That's a much better architecture than everything being either permanent or fully reactive.

---

# Where CUDA fits

I would **not** try to run the Rust Bonsai runtime inside CUDA.

Instead:

```text
Sollimann/bonsai
       │
       │ semantics/reference
       ▼
Our compact BT IR
       │
       ├── CPU/Rust visualizer/debugger
       │
       ├── CUDA executor
       │
       └── Python compiler
```

The CUDA representation could be tiny:

```cpp
struct Node {
    uint8_t type;
    uint16_t child_begin;
    uint16_t child_count;
    uint16_t primitive;
    int32_t param0;
    int32_t param1;
};
```

A whole policy could be perhaps tens or hundreds of nodes.

Then each environment executes:

```text
ObservationView
      ↓
BT interpreter
      ↓
condition/action leaves
      ↓
Game Action
      ↓
Sim::step()
```

all on the GPU.

No Python.

---

# Then search thousands of policies simultaneously

Our natural CUDA tensor becomes:

```text
snapshot
×
behavior-tree candidate
×
opponent
×
seat
```

For example:

```text
128 states
× 512 trees
× 8 opponent families
× 2 seats

= 1,048,576 continuation evaluations
```

Obviously we'd tile this for 4 GB VRAM, but architecturally that is the problem.

The exact simulator then gives us the fitness.

---

# Fitness should not be simple average reward

This is another advantage over blindly training standard RL.

We can explicitly optimize what matters for Elo robustness:

```text
fitness =
    win rate
  + mean margin contribution
  + lower-tail performance
  + opponent diversity
  + seat robustness
  - catastrophic loss penalty
  - unnecessary tree complexity
```

Potentially something like:

```text
maximize:
    50% win rate
    20% CVaR10
    15% minimum-cohort robustness
    10% mean margin
     5% simplicity
```

Exact coefficients would be researched, not guessed.

A simplicity penalty is valuable because otherwise genetic programming produces enormous brittle trees.

---

# We can also adversarially evolve opponents

This gets very interesting.

Have two populations:

```text
MAIN POLICY POPULATION

EXPLOITER POPULATION
```

The exploiter's fitness:

```text
beat current main policy
```

Then:

```text
main improves
     ↓
exploiter finds weakness
     ↓
main evolves around weakness
     ↓
new exploiter
     ↓
...
```

This gives us some of the strongest benefits of adversarial RL/self-play **without training a neural policy**.

It's still optimization, but not RL.

---

# Important terminology

Technically, I wouldn't say:

> “no training.”

There is still an optimization process.

But it's:

**program search**

rather than:

**gradient-based reinforcement learning**.

No backpropagation.

No PPO instability.

No reward-network interaction.

No neural architecture problem.

No model checkpoint inference dependency.

No worry about the NN inventing hidden correlations that are difficult to audit.

---

# Where RL could still win

There is one important case.

Imagine the optimal decision boundary depends on a combination like:

```text
27 continuous state features
× complex nonlinear interactions
```

A neural network can naturally represent:

```text
f(x1, x2, ... x27)
```

A behavior tree may need hundreds/thousands of branches.

So if Kaggriculture ultimately demands very smooth high-dimensional generalization, RL/value networks may outperform a compact tree.

But based on everything we've learned so far, our failures tend to look like:

```text
wrong continuation in recognizable cohort
wrong switch timing
wrong market regime response
wrong shop-dependent continuation
wrong inventory/capital state
```

Those are **extremely BT-friendly problems**.

---

# And our current v32 is evidence of that

v32's strength comes from exactly these kinds of decisions.

It isn't randomly succeeding despite being heuristic.

It is succeeding because the game has exploitable structure.

The problem is that our implementation has become difficult to scale.

Bonsai gives us the formal language for scaling it.

CUDA gives us the mechanism to automatically optimize it.

Those two together are much more interesting than Bonsai alone.

---

# My assessment

I would separate the propositions.

### “Take our current agent and rewrite it using Bonsai”

Useful, but not transformative.

It gives us:

```text
cleaner architecture
better debugging
proper memory semantics
reactive preemption
modular experts
visualization
```

Worth doing, but it doesn't magically improve Elo.

### “Use Bonsai-style trees as the search space and let the exact CUDA engine evolve/synthesize them”

**This could be a major breakthrough.**

That gives us:

```text
expert knowledge
+
automatic policy discovery
+
massive exact simulation
+
reactivity
+
interpretability
+
legal auditing
+
very cheap deployment
```

This is the version of your idea I think could plausibly compete with a hybrid RL system.

---

# One especially attractive architecture

I'd build three layers:

```text
┌──────────────────────────────────────────────┐
│ LAYER 1 — SAFETY / INVARIANTS                │
│ deterministic hard rules                     │
│ legality, catastrophic prevention, etc.      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ LAYER 2 — EVOLVED BEHAVIOR TREE              │
│ CUDA-optimized routing / conditions / memory │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ LAYER 3 — EXPERT LIBRARY                     │
│ v32 / META / Q8 / Kaw / Anik / Jason / ...  │
│ + specialized market/worker controllers      │
└──────────────────────────────────────────────┘
```

So the evolved tree doesn't need to understand every low-level action.

It can say:

```text
UseExpert(KAW)
```

or:

```text
UseExpert(ANIK)
```

or:

```text
OverrideMarket(WOOL_HOLD)
```

That makes the search dramatically more sample-efficient.

---

# Eventually the tree can become finer-grained

Once routing is working, let evolution select smaller modules:

```text
worker expert = Kaw
market expert = OFA
crop expert = META
risk controller = custom
```

Then potentially:

```text
actor controller
+
market controller
```

can be independently selected.

That starts producing genuinely novel hybrid agents rather than merely selecting entire donor policies.

---

# Final deployment is trivial compared with RL

Once we have the winning tree:

```text
CUDA/Rust research representation
        ↓
tree compiler
        ↓
plain Python
```

For example an optimized tree could compile into:

```python
if emergency(obs, bb):
    return emergency_action(obs, bb)

if pet_lane(obs, bb):
    if wool_opportunity(obs, bb):
        return wool_controller(obs, bb)
    if pizza_branch(obs, bb):
        return pet_pizza(obs, bb)
    return kaw(obs, bb)

return meta(obs, bb)
```

No Bonsai package needs to be present on Kaggle.

No Rust binary.

No GPU.

No PyTorch.

Just the final discovered deterministic policy.

That is extremely attractive.

---

## So: is this a jackpot?

**Bonsai by itself: no.**

**Bonsai's behavior-tree model + our elite-policy library + the exact CUDA engine + automated tree/program search: potentially yes.**

I think this is actually a **better next research direction than immediately committing to full RL**.

The timing is also unusually good because the two pieces we would have needed before attempting this are exactly what we now possess:

```text
1. strong existing heuristic/expert policies
2. extremely fast exact CUDA evaluation
```

Bonsai supplies the missing third piece:

```text
3. a principled hierarchical/reactive policy representation
```

My next move would therefore be **not to rewrite v32 in the Rust crate and stop there**. I would use the cloned Bonsai implementation as the semantic reference, build a compact **Bonsai-compatible behavior-tree IR/executor directly into `cuda-engine`**, translate v32/v33 into its initial tree, and then build an evolutionary/program-synthesis search around that representation.

That has a credible path from our present heuristic agent to a **search-discovered hybrid agent** without neural RL. ([GitHub][1])

[1]: https://github.com/Sollimann/bonsai "https://github.com/Sollimann/bonsai"
[2]: https://github.com/catornot/bp-ort "https://github.com/catornot/bp-ort"
[3]: https://github.com/utahrobotics/utah-lunabotics-2026 "https://github.com/utahrobotics/utah-lunabotics-2026"
[4]: https://www.nasa.gov/centers-and-facilities/kennedy/nasas-2026-lunabotics-winning-student-teams-engineering-lunar-future/ "https://www.nasa.gov/centers-and-facilities/kennedy/nasas-2026-lunabotics-winning-student-teams-engineering-lunar-future/"
[5]: https://www.nasa.gov/learning-resources/lunabotics-challenge/ "https://www.nasa.gov/learning-resources/lunabotics-challenge/"

