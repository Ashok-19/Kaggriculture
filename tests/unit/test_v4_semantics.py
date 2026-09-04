from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from kaggriculture_bonsai.runtime import bonsai_kg as b3
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as b


def P(tree, bb=None, enums=None, **extra):
    out = {"schema_version": 4, "tree": tree}
    if bb is not None: out["blackboard"] = bb
    if enums is not None: out["enums"] = enums
    out.update(extra)
    return out


def obs(step=0, units=1, extra=None):
    vals = {
        (b.FEATURES["STEP"], 0): step,
        (b.FEATURES["OWN_UNITS"], 0): units,
        (b.FEATURES["OWN_WORKER_X"], 0): 4,
        (b.FEATURES["OWN_WORKER_Y"], 0): 4,
    }
    if extra: vals.update(extra)
    return b.SyntheticObservation(vals)


class BKG4Semantics(unittest.TestCase):
    def vm(self, tree, bb=None, enums=None):
        return b.VM(b.compile_policy(P(tree, bb, enums)))

    def test_01_abi_sizes_and_ceiling(self):
        self.assertEqual((b.HEADER.size, b.NODE_STRUCT.size, b.INTENT_STRUCT.size), (64, 32, 12))
        self.assertEqual((b.MAX_NODES, b.MAX_CHILD_REFS, b.NONE_U16), (65534, 65533, 65535))

    def test_02_sync_sequence_state_elided(self):
        cp = b.compile_policy(P({"type":"Sequence","children":[{"type":"Success"},{"type":"Failure"}]}))
        self.assertEqual(cp.state_count, 0)
        self.assertTrue(all(n.state_slot == b.NONE_U16 for n in cp.nodes))
        self.assertEqual(b.VM(cp).tick(obs()).status, b.STATUS_FAILURE)

    def test_03_running_sequence_allocates_only_required_state(self):
        cp = b.compile_policy(P({"type":"Sequence","children":[{"type":"Success"},{"type":"WaitTurns","turns":2}]}))
        # Sequence cursor + WaitTurns timer. Success is stateless.
        self.assertEqual(cp.state_count, 2)
        self.assertNotEqual(cp.nodes[0].state_slot, b.NONE_U16)
        self.assertEqual(cp.nodes[1].state_slot, b.NONE_U16)
        self.assertNotEqual(cp.nodes[2].state_slot, b.NONE_U16)

    def test_04_synchronous_lifecycle_does_not_force_state(self):
        tree={"type":"Success","lifecycle":{"on_success":[{"op":"set","slot":"x","value":1}]}}
        cp=b.compile_policy(P(tree,[{"name":"x","type":"I32","initial":0}]))
        self.assertEqual(cp.state_count,0)
        self.assertEqual(b.VM(cp).tick(obs()).blackboard,(1,))

    def test_05_running_lifecycle_gets_active_state(self):
        tree={"type":"MoveUnitTo","unit":0,"target":{"const":0,"type":"TILE"},"lifecycle":{"on_halted":[{"op":"set","slot":"x","value":1}]}}
        cp=b.compile_policy(P(tree,[{"name":"x","type":"I32","initial":0}]))
        self.assertEqual(cp.state_count,1)

    def test_06_named_enum_switch_sync_has_no_state(self):
        enums=[{"name":"StrategicMode","values":["OPENING","NORMAL","RECOVERY"]}]
        bb=[{"name":"mode","type":"StrategicMode","initial":"NORMAL"}]
        tree={"type":"SwitchEnum","slot":"mode","cases":{"OPENING":{"type":"Failure"},"NORMAL":{"type":"Success"},"RECOVERY":{"type":"Failure"}},"default":{"type":"Failure"}}
        cp=b.compile_policy(P(tree,bb,enums))
        self.assertEqual(cp.state_count,0)
        self.assertEqual(b.VM(cp).tick(obs()).status,b.STATUS_SUCCESS)
        self.assertEqual(cp.analysis[0].max_visits,2)

    def test_07_switch_running_branch_preemption_halts_old_child(self):
        enums=[{"name":"Mode","values":["A","B"]}]
        bb=[{"name":"mode","type":"Mode","initial":"A"},{"name":"halted","type":"I32","initial":0}]
        old={"type":"WaitTurns","turns":9,"lifecycle":{"on_halted":[{"op":"set","slot":"halted","value":1}]}}
        tree={"type":"SwitchEnum","slot":"mode","cases":{"A":old,"B":{"type":"Success"}},"default":{"type":"Failure"}}
        vm=self.vm(tree,bb,enums)
        self.assertEqual(vm.tick(obs(0)).status,b.STATUS_RUNNING)
        vm.state.blackboard[0]=1
        r=vm.tick(obs(1),trace=True)
        self.assertEqual((r.status,r.blackboard),(b.STATUS_SUCCESS,(1,1)))
        self.assertTrue(any(e[2]=="halt" for e in r.trace))

    def test_08_reactive_if_rechecks_and_halts(self):
        bb=[{"name":"halted","type":"I32","initial":0}]
        then={"type":"WaitTurns","turns":9,"lifecycle":{"on_halted":[{"op":"set","slot":"halted","value":1}]}}
        tree={"type":"ReactiveIf","condition":{"type":"Condition","lhs":{"feature":"STEP"},"op":"LT","rhs":1},"then":then,"else":{"type":"Success"}}
        vm=self.vm(tree,bb)
        self.assertEqual(vm.tick(obs(0)).status,b.STATUS_RUNNING)
        r=vm.tick(obs(1),trace=True)
        self.assertEqual((r.status,r.blackboard),(b.STATUS_SUCCESS,(1,)))
        self.assertTrue(any(e[2]=="halt" for e in r.trace))

    def test_09_reactive_if_condition_must_be_synchronous(self):
        tree={"type":"ReactiveIf","condition":{"type":"WaitTurns","turns":1},"then":{"type":"Success"},"else":{"type":"Failure"}}
        with self.assertRaisesRegex(b.PolicyError,"condition must be synchronous"):
            b.compile_policy(P(tree))

    def test_10_if_remains_latched(self):
        bb=[{"name":"else_hit","type":"I32","initial":0}]
        tree={"type":"If","condition":{"type":"Condition","lhs":{"feature":"STEP"},"op":"LT","rhs":1},"then":{"type":"WaitTurns","turns":2},"else":{"type":"SetBlackboard","slot":"else_hit","value":1}}
        vm=self.vm(tree,bb)
        self.assertEqual(vm.tick(obs(0)).status,b.STATUS_RUNNING)
        self.assertEqual(vm.tick(obs(1)).blackboard,(0,))

    def test_11_named_enum_type_safety(self):
        enums=[{"name":"A","values":["X","Y"]},{"name":"B","values":["X","Y"]}]
        bb=[{"name":"a","type":"A","initial":"X"},{"name":"b","type":"B","initial":"X"}]
        with self.assertRaisesRegex(b.PolicyError,"enum"):
            b.compile_policy(P({"type":"Condition","lhs":{"blackboard":"a"},"op":"EQ","rhs":{"blackboard":"b"}},bb,enums))
        with self.assertRaisesRegex(b.PolicyError,"EQ/NE"):
            b.compile_policy(P({"type":"Condition","lhs":{"blackboard":"a"},"op":"LT","rhs":"Y"},bb,enums))

    def test_12_authority_dominates_local_priority(self):
        tree={"type":"Parallel","success_threshold":2,"failure_threshold":1,"children":[
            {"type":"EmitUnitIntent","unit":0,"action":"NORTH","authority":"TACTICAL","priority":255},
            {"type":"EmitUnitIntent","unit":0,"action":"SOUTH","authority":"INVARIANT","priority":0},
        ]}
        r=self.vm(tree).tick(obs())
        self.assertEqual(r.action.units[0][0],b.UNIT_OPS["SOUTH"])

    def test_13_metadata_is_interned(self):
        c={"type":"Condition","lhs":{"feature":"STEP"},"op":"GE","rhs":72}
        cp=b.compile_policy(P({"type":"Sequence","children":[copy.deepcopy(c),copy.deepcopy(c),copy.deepcopy(c)]}))
        # STEP and constant 72 are shared ValueRefs rather than triplicated.
        self.assertEqual(len(cp.values),2)
        self.assertEqual(len(cp.predicates),0)  # inline Conditions do not allocate lifecycle predicates.

    def test_14_lifecycle_predicate_and_effect_sequence_interning(self):
        lc={"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":4},"on_halted":[{"op":"clear","slot":"x"}]}
        tree={"type":"Parallel","success_threshold":2,"failure_threshold":2,"children":[
            {"type":"WaitTurns","turns":5,"lifecycle":copy.deepcopy(lc)},
            {"type":"WaitTurns","turns":6,"lifecycle":copy.deepcopy(lc)},
        ]}
        cp=b.compile_policy(P(tree,[{"name":"x","type":"I32","initial":0}]))
        self.assertEqual((len(cp.lifecycles),len(cp.predicates),len(cp.effects)),(1,1,1))

    def test_15_live_intent_static_bound_respects_selector_rollback(self):
        branches=[]
        for i in range(40):
            branches.append({"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE"},{"type":"Failure"}]})
        branches.append({"type":"EmitMarketIntent","action":"BUY_LAND"})
        cp=b.compile_policy(P({"type":"Selector","children":branches}))
        self.assertEqual(cp.max_live_intents,1)
        self.assertEqual(cp.analysis[0].max_emissions,41)

    def test_16_emission_order_can_exceed_255_while_live_journal_stays_small(self):
        branches=[{"type":"Sequence","children":[{"type":"EmitMarketIntent","action":"HIRE","priority":1},{"type":"Failure"}]} for _ in range(300)]
        branches.append({"type":"EmitMarketIntent","action":"BUY_LAND","priority":1})
        cp=b.compile_policy(P({"type":"Selector","children":branches}))
        self.assertGreater(cp.analysis[0].max_emissions,255)
        self.assertEqual(cp.max_live_intents,1)
        r=b.VM(cp).tick(obs(),profile=True)
        self.assertEqual(r.action.market,[(b.MARKET_OPS["BUY_LAND"],0,1)])
        self.assertEqual(r.profile.emitted_intents,301)

    def test_17_binary_roundtrip(self):
        src=P({"type":"Sequence","children":[{"type":"Success"},{"type":"WaitTurns","turns":2}]})
        cp=b.compile_policy(src)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/"p.bkgc";b.save_compiled(cp,f);loaded=b.load_compiled(f)
            self.assertEqual(cp.semantic_hash,loaded.semantic_hash)
            self.assertEqual(b.semantic_payload_bytes(cp),b.semantic_payload_bytes(loaded))

    def test_18_bkg3_conversion_preserves_actions_and_status(self):
        src3={"schema_version":3,"tree":{"type":"Sequence","children":[
            {"type":"EmitMarketIntent","action":"HIRE","priority":12},
            {"type":"WaitTurns","turns":2},
        ]}}
        c3=b3.compile_policy(src3);c4=b.compile_bkg3_to_v4(c3);v3=b3.VM(c3);v4=b.VM(c4)
        for step in range(4):
            r3=v3.tick(obs(step));r4=v4.tick(obs(step))
            self.assertEqual((r3.status,r3.action.compact(),r3.blackboard),(r4.status,r4.action.compact(),r4.blackboard))

    def test_19_illegal_features_remain_absent(self):
        for name in ("RNG_SEED","OPP_SHED","REPLAY_ID","BOARD_HASH"):
            with self.assertRaises(b.PolicyError):
                b.compile_policy(P({"type":"Condition","lhs":{"feature":name},"op":"EQ","rhs":0}))

    def test_20_switch_enum_rejects_raw_i32_and_raw_enum8(self):
        for typ in ("I32","ENUM8"):
            bb=[{"name":"mode","type":typ,"initial":0}]
            with self.assertRaisesRegex(b.PolicyError,"named enum"):
                b.compile_policy(P({"type":"SwitchEnum","slot":"mode","cases":{},"default":{"type":"Success"}},bb))

    def test_21_header_proof_tamper_rejected(self):
        cp=b.compile_policy(P({"type":"WaitTurns","turns":2}))
        bad=copy.deepcopy(cp);bad.max_depth+=1
        with self.assertRaisesRegex(b.PolicyError,"max_depth"):
            b.validate_compiled(bad)
        bad=copy.deepcopy(cp);bad.max_live_intents=1
        with self.assertRaisesRegex(b.PolicyError,"max_live_intents"):
            b.validate_compiled(bad)

    def test_22_linter_flags_large_linear_dispatch(self):
        children=[{"type":"Failure"} for _ in range(24)]
        cp=b.compile_policy(P({"type":"Selector","children":children}))
        self.assertTrue(any(w["code"]=="LARGE_LINEAR_DISPATCH" for w in cp.warnings))

    def test_23_specificity_linter_flags_exact_high_cardinality(self):
        cp=b.compile_policy(P({"type":"Condition","lhs":{"feature":"STEP"},"op":"EQ","rhs":197}))
        self.assertTrue(any(w["code"]=="SPECIFIC_EXACT_EQUALITY" for w in cp.warnings))

    def test_24_actual_runtime_allocates_actual_counts(self):
        cp=b.compile_policy(P({"type":"WaitTurns","turns":2},[{"name":"x","type":"I32","initial":3}]))
        vm=b.VM(cp)
        self.assertEqual((len(vm.state.blackboard),len(vm.state.node_state)),(1,1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
