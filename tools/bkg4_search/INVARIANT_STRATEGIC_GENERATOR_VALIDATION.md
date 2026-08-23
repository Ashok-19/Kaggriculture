# Invariant Strategic Generator Validation

All game labels in this report were produced by the exact resident CUDA evaluators.
Certification was evaluated only after the discovery candidate was frozen; ranker validation was not read by search.

## Representation Spike

- Candidates: 84
- Candidate-case evaluations (including shared baseline): 2125
- Active / behavior-changing: 76 / 73
- Production-qualified: 0
- Unsafe: 65
- Structural semantic niches: 14

## Five Tape-Disjoint Splits

| Fold | Budget | Train cases | Held-out cases | Held-out useful groups | Held-out regressions | Production |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 64 | 20 | 5 | 0 | 0 | False |
| 1 | 64 | 21 | 4 | 1 | 0 | False |
| 2 | 64 | 19 | 6 | 1 | 0 | False |
| 3 | 64 | 19 | 6 | 0 | 0 | False |
| 4 | 64 | 21 | 4 | 0 | 0 | False |

## Equal-Budget Benchmark

Each arm evaluated 64 children on 20 discovery cases, then evaluated its frozen best child on 5 tape-disjoint discovery cases.

```json
{
  "fixed_systematic_legacy": {
    "behavior_changing_rate": 0.671875,
    "best_target_improvement": 11497,
    "best_train_semantic_id": "74635a2dd1fc928f4c21ce9767402284cf8d5a037dfb6a823796543817b2a31b",
    "best_worst_margin_improvement": 0,
    "duplicate_rate": 0.0,
    "elite_per_1000": 15.625,
    "emitter_families": [
      "systematic"
    ],
    "evaluated_candidates": 64,
    "exact_candidate_case_evaluations": 1280,
    "heldout": {
      "action_change_total": 58,
      "activation_total": 240,
      "behavior_changing": true,
      "exact_noop": false,
      "fragile_wins_after": 0,
      "fragile_wins_before": 0,
      "l2w_count": 0,
      "lex_fragile_wins_after": 0,
      "lex_losses_after": 1,
      "lex_mean_margin_delta": -3411.8,
      "lex_new_losses": 0,
      "lex_w2l": 0,
      "lex_worst_margin": -3204,
      "losses_after": 1,
      "losses_before": 1,
      "mean_margin_delta": -3411.8,
      "new_loss_count": 0,
      "severe_regression_count": 1,
      "strict_safe": false,
      "w2l_count": 0,
      "worst_candidate_margin": -3204,
      "worst_protected_win_delta": -17155
    },
    "l2w_per_1000": 250.0,
    "max_generation_depth": 1,
    "new_loss_candidates": 38,
    "new_loss_rate": 0.59375,
    "no_op_rate": 0.328125,
    "operation_coverage": 3,
    "operation_families": [
      "ADD_SPECIALIST",
      "CHANGE_WINDOW",
      "INSERT_RECOVERY"
    ],
    "parent_rule_count_coverage": 1,
    "parent_rule_count_lower_bounds": [
      0
    ],
    "qd_niches": 24,
    "requested_candidates": 64,
    "root_feasible_rate": 0.40625,
    "safe_useful_per_1000": 15.625,
    "severe_regression_rate": 0.59375,
    "source_parent_count": 1,
    "strict_safe_l2w_per_1000": 0.0,
    "target_depth_coverage": 1,
    "target_kinds": [
      "root"
    ],
    "target_loss_improving_per_1000": 390.625,
    "unique_valid_proposals": 64,
    "useful_diversity": 1,
    "w2l_rate": 0.0
  },
  "invariant_strategic_b4im": {
    "activation_count": 107567,
    "behavior_changing_rate": 0.875,
    "best_train_semantic_id": "ea23f9ce1b465adb459b76fb7af7ed155bab5eb43212e15de60d7a97232e705b",
    "evaluated_candidates": 64,
    "exact_candidate_case_evaluations": 1280,
    "heldout": {
      "action_change_count": 9,
      "activation_count": 9,
      "case_rows": [
        {
          "action_changed": true,
          "activation_count": 1,
          "baseline_margin": 23545,
          "candidate_margin": 22949,
          "case_id": "mt-041",
          "margin_delta": -596,
          "new_loss": false,
          "trace_hash": 7832542820661769978,
          "w2l": false
        },
        {
          "action_changed": true,
          "activation_count": 1,
          "baseline_margin": -3204,
          "candidate_margin": -3543,
          "case_id": "mt-016",
          "margin_delta": -339,
          "new_loss": false,
          "trace_hash": 6960897705336187060,
          "w2l": false
        },
        {
          "action_changed": true,
          "activation_count": 3,
          "baseline_margin": 5491,
          "candidate_margin": -2115,
          "case_id": "mt-035",
          "margin_delta": -7606,
          "new_loss": true,
          "trace_hash": 77576159661175205,
          "w2l": true
        },
        {
          "action_changed": true,
          "activation_count": 3,
          "baseline_margin": 5491,
          "candidate_margin": -2115,
          "case_id": "mt-036",
          "margin_delta": -7606,
          "new_loss": true,
          "trace_hash": 77576159661175205,
          "w2l": true
        },
        {
          "action_changed": true,
          "activation_count": 1,
          "baseline_margin": 59504,
          "candidate_margin": 59183,
          "case_id": "mt-056",
          "margin_delta": -321,
          "new_loss": false,
          "trace_hash": 5955072595298745264,
          "w2l": false
        }
      ],
      "cases": 5,
      "descriptor": {
        "actions": [
          "ROUTE_OVERRIDE"
        ],
        "activation_topology": "broad",
        "controls": [
          "INSTANT"
        ],
        "domains": [
          "GLOBAL"
        ],
        "effects": [
          "AVOID_OVERBUY"
        ],
        "guard_families": [
          [
            "own_resource"
          ]
        ]
      },
      "fitness": {
        "behavior_groups": 4,
        "groups_with_new_loss": 1,
        "groups_with_severe_regression": 1,
        "l2w_groups": 0,
        "lower_quartile_delta": -7606.0,
        "mean_group_delta": -2215.5,
        "median_group_delta": -467.5,
        "safe_groups": 3,
        "unsafe_groups": 1,
        "useful_groups": 0,
        "worst_group_delta": -7606.0
      },
      "group_outcomes": [
        {
          "behavior_changed": true,
          "group_id": "/home/nnmax/Desktop/kaggle/kaggriculture/scratch/bkg4_search/fresh_promotion24/tapes/Ryo_Hasegawa_s0_ep94545578.json",
          "l2w": 0,
          "margin_delta": -7606.0,
          "new_loss": true,
          "severe_regression": true,
          "useful": false
        },
        {
          "behavior_changed": true,
          "group_id": "/home/nnmax/Desktop/kaggle/kaggriculture/scratch/bkg4_search/fresh_promotion24/tapes/Ryo_Hasegawa_s1_ep94460852.json",
          "l2w": 0,
          "margin_delta": -339.0,
          "new_loss": false,
          "severe_regression": false,
          "useful": false
        },
        {
          "behavior_changed": true,
          "group_id": "/home/nnmax/Desktop/kaggle/kaggriculture/scratch/bkg4_search/optimization_oracle/fresh_tapes/Arman_Tuganbaev_s1_ep94523511.json",
          "l2w": 0,
          "margin_delta": -321.0,
          "new_loss": false,
          "severe_regression": false,
          "useful": false
        },
        {
          "behavior_changed": true,
          "group_id": "/home/nnmax/Desktop/kaggle/kaggriculture/scratch/kaggriculture_meta_search_bundle/populations/dev_tapes/0058_0009_Efe_Can_Celiksoy_s0_ep94119472.json",
          "l2w": 0,
          "margin_delta": -596.0,
          "new_loss": false,
          "severe_regression": false,
          "useful": false
        }
      ],
      "groups": 4,
      "qualification": {
        "behavior_groups": 4,
        "l2w_groups": 0,
        "production": false,
        "reason": "unsafe discovery group",
        "safe_groups": 3,
        "total_groups": 4,
        "useful_groups": 0
      },
      "semantic_id": "ea23f9ce1b465adb459b76fb7af7ed155bab5eb43212e15de60d7a97232e705b"
    },
    "new_loss_candidates": 47,
    "production_qualified_elites": 2,
    "requested_candidates": 64,
    "semantic_niches": 13,
    "unique_valid_proposals": 64
  },
  "permanent_module_genome": {
    "behavior_changing_rate": 0.625,
    "best_target_improvement": 11188,
    "best_train_semantic_id": "ce8178c69d1abea2ed2ed2429cd276b0f22f52184c263987f55c9ea5f4ca11fa",
    "best_worst_margin_improvement": 0,
    "duplicate_rate": 0.24175824175824176,
    "elite_per_1000": 0.0,
    "emitter_families": [
      "contrastive",
      "evolvability_mcts",
      "lns",
      "local_refine",
      "motif",
      "random_typed",
      "recombine",
      "structural",
      "systematic"
    ],
    "evaluated_candidates": 64,
    "exact_candidate_case_evaluations": 1280,
    "heldout": {
      "action_change_total": 485,
      "activation_total": 533,
      "behavior_changing": true,
      "exact_noop": false,
      "fragile_wins_after": 0,
      "fragile_wins_before": 0,
      "l2w_count": 0,
      "lex_fragile_wins_after": 0,
      "lex_losses_after": 1,
      "lex_mean_margin_delta": -3163.2,
      "lex_new_losses": 0,
      "lex_w2l": 0,
      "lex_worst_margin": -3204,
      "losses_after": 1,
      "losses_before": 1,
      "mean_margin_delta": -3163.2,
      "new_loss_count": 0,
      "severe_regression_count": 1,
      "strict_safe": false,
      "w2l_count": 0,
      "worst_candidate_margin": -3204,
      "worst_protected_win_delta": -15816
    },
    "l2w_per_1000": 109.375,
    "max_generation_depth": 2,
    "new_loss_candidates": 22,
    "new_loss_rate": 0.34375,
    "no_op_rate": 0.375,
    "operation_coverage": 7,
    "operation_families": [
      "ADD_CONDITION",
      "ADD_SPECIALIST",
      "CHANGE_ROUTE",
      "CHANGE_THRESHOLD",
      "CHANGE_WINDOW",
      "COMPOSE_PREDICATES",
      "INSERT_RECOVERY"
    ],
    "parent_rule_count_coverage": 2,
    "parent_rule_count_lower_bounds": [
      0,
      1
    ],
    "qd_niches": 35,
    "requested_candidates": 64,
    "root_feasible_rate": 0.65625,
    "safe_useful_per_1000": 93.75,
    "severe_regression_rate": 0.34375,
    "source_parent_count": 9,
    "strict_safe_l2w_per_1000": 0.0,
    "target_depth_coverage": 3,
    "target_kinds": [
      "condition",
      "root",
      "rule"
    ],
    "target_loss_improving_per_1000": 468.75,
    "unique_valid_proposals": 64,
    "useful_diversity": 3,
    "w2l_rate": 0.0
  },
  "random_typed_legacy": {
    "behavior_changing_rate": 0.4375,
    "best_target_improvement": 8533,
    "best_train_semantic_id": "455f46b34d5d8aa74b0fd8ca9dd8951aab9c00b5883e688a588b00dc0b68feb7",
    "best_worst_margin_improvement": 0,
    "duplicate_rate": 0.0,
    "elite_per_1000": 15.625,
    "emitter_families": [
      "random_typed"
    ],
    "evaluated_candidates": 64,
    "exact_candidate_case_evaluations": 1280,
    "heldout": {
      "action_change_total": 0,
      "activation_total": 0,
      "behavior_changing": false,
      "exact_noop": true,
      "fragile_wins_after": 0,
      "fragile_wins_before": 0,
      "l2w_count": 0,
      "lex_fragile_wins_after": 0,
      "lex_losses_after": 1,
      "lex_mean_margin_delta": 0.0,
      "lex_new_losses": 0,
      "lex_w2l": 0,
      "lex_worst_margin": -3204,
      "losses_after": 1,
      "losses_before": 1,
      "mean_margin_delta": 0.0,
      "new_loss_count": 0,
      "severe_regression_count": 0,
      "strict_safe": true,
      "w2l_count": 0,
      "worst_candidate_margin": -3204,
      "worst_protected_win_delta": 0
    },
    "l2w_per_1000": 93.75,
    "max_generation_depth": 1,
    "new_loss_candidates": 19,
    "new_loss_rate": 0.296875,
    "no_op_rate": 0.5625,
    "operation_coverage": 3,
    "operation_families": [
      "ADD_SPECIALIST",
      "CHANGE_WINDOW",
      "INSERT_RECOVERY"
    ],
    "parent_rule_count_coverage": 1,
    "parent_rule_count_lower_bounds": [
      0
    ],
    "qd_niches": 31,
    "requested_candidates": 64,
    "root_feasible_rate": 0.671875,
    "safe_useful_per_1000": 15.625,
    "severe_regression_rate": 0.3125,
    "source_parent_count": 1,
    "strict_safe_l2w_per_1000": 15.625,
    "target_depth_coverage": 1,
    "target_kinds": [
      "root"
    ],
    "target_loss_improving_per_1000": 187.5,
    "unique_valid_proposals": 64,
    "useful_diversity": 1,
    "w2l_rate": 0.0
  }
}
```

## Exact Parity And Runtime Integration

- Native architecture: `cuobjdump --list-elf` reports only `sm_86` cubins for both resident evaluators.
- Empty overlay: 4/4 B4IM rows matched B4EC byte-for-byte after the 20-byte ABI header; 0 activations, 0 action changes, 0 bad rows. B4IM kernel time was 366.848 ms.
- Active mini-IR: 16/16 CPU C++ rows matched CUDA exactly for terminal money, action trace hash, and activation count; 5,947 action changes, 11,504 activations, 0 bad rows. CUDA kernel time was 490.777 ms.
- Distilled ordinary BKG4: representative `ROUTE_OVERRIDE`, `COMMITTED_ROUTE`, `UNIT_REWRITE`, and `MARKET_INTENT` modules produced 16/16 exact CPU C++/CUDA rows; 5,947 action changes, 11,504 activations, 0 bad rows. CUDA kernel time was 499.023 ms.
- Python ordinary runtime tests execute all four action families through real composed current-parent assets while preserving feed-deferral slots 5..17 and route-value slot 19.

## Regression Matrix

- `python3 -m unittest discover -s tools/bkg4_search/edit_search -p 'test_*.py' -v`: 182 passed.
- `python3 -m unittest discover -s tools/edit_corpus -p 'test_*.py' -v`: 59 passed.
- Modified Python `compileall`: passed.
- `bkg4_invariant_module_semantics.cpp`: compiled and passed.
- `bkg4_invariant_overlay_cpp.cpp` and `bkg4_invariant_distilled_cpp.cpp`: compiled; exact parity checks passed.
- Fresh corpus validation: 84 rows, 84 canonical semantic IDs, 0 duplicates, 84/84 exact `B4IC1/v1` discovery labels with parent/evaluator provenance.

No production candidate transferred across enough independent discovery groups or passed the two-useful-group certification threshold, so no champion was promoted and no official-engine 64-game promotion campaign was run. This is an intentional label-only rejection, not a parity or generator failure.

## Certification And Corpus

- Frozen certification semantic ID: `d7ca7700ea5b3822e870c98b329ee3e1aede039ed435140f596afe1bd5ad5bda`
- Certification production qualification: False
- Candidate promoted: False
- Fresh corpus rows / unique semantics: 84 / 84
- Ranker trained: False

## Throughput

```json
{
  "cuda_invocations": 23,
  "effective_games_s": 1324.0157411712526,
  "games": 13905,
  "gpu_ms": 10502.141,
  "max_invocation_games_s": 2832.8,
  "min_invocation_games_s": 16.4989
}
```
