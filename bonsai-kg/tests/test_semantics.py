from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg


def policy(tree, bb=None):
    return {"schema_version": 2, "blackboard": bb or [], "tree": tree}


def obs(step=0, extra=None):
    values = {
        (bkg.FEATURES["STEP"], 0): step,
        (bkg.FEATURES["DAY"], 0): step // 24,
        (bkg.FEATURES["HOUR"], 0): step % 24,
        (bkg.FEATURES["OWN_UNITS"], 0): 1,
        (bkg.FEATURES["OPP_UNITS"], 0): 1,
        (bkg.FEATURES["OWN_WORKER_X"], 0): 4,
        (bkg.FEATURES["OWN_WORKER_Y"], 0): 4,
    }
    if extra:
        values.update(extra)
    return bkg.SyntheticObservation(values)


class BonsaiKGSemantics(unittest.TestCase):
    def vm(self, tree, bb=None):
        return bkg.VM(bkg.compile_policy(policy(tree, bb)))

    def test_01_success_failure_running_propagation(self):
        v = self.vm({"type":"Sequence","children":[{"type":"Success"},{"type":"Failure"}]})
        self.assertEqual(v.tick(obs()).status, bkg.STATUS_FAILURE)
        v = self.vm({"type":"Selector","children":[{"type":"Failure"},{"type":"Success"}]})
        self.assertEqual(v.tick(obs()).status, bkg.STATUS_SUCCESS)
        v = self.vm({"type":"WaitTurns","turns":2})
        self.assertEqual(v.tick(obs(0)).status, bkg.STATUS_RUNNING)

    def test_02_stateful_sequence_resumes_running_child(self):
        bb=[{"name":"count","type":"I32","initial":0}]
        tree={"type":"Sequence","children":[{"type":"AddBlackboard","slot":"count","value":1},{"type":"WaitTurns","turns":2}]}
        v=self.vm(tree,bb)
        self.assertEqual(v.tick(obs(0)).blackboard,(1,))
        self.assertEqual(v.tick(obs(1)).blackboard,(1,))
        r=v.tick(obs(2)); self.assertEqual(r.status,bkg.STATUS_SUCCESS); self.assertEqual(r.blackboard,(1,))

    def test_03_reactive_sequence_rechecks_and_halts(self):
        bb=[{"name":"reservation","type":"I32","initial":0}]
        plan={"type":"Sequence","lifecycle":{"on_halted":[{"op":"clear","slot":"reservation"}]},"children":[{"type":"SetBlackboard","slot":"reservation","value":99},{"type":"WaitTurns","turns":10}]}
        tree={"type":"ReactiveSequence","children":[{"type":"Condition","lhs":{"feature":"STEP"},"op":"LT","rhs":2},plan]}
        v=self.vm(tree,bb)
        self.assertEqual(v.tick(obs(0)).blackboard,(99,))
        self.assertEqual(v.tick(obs(1)).blackboard,(99,))
        r=v.tick(obs(2),trace=True); self.assertEqual(r.status,bkg.STATUS_FAILURE); self.assertEqual(r.blackboard,(0,)); self.assertTrue(any(e[2]=="halt" for e in r.trace))

    def test_04_stateful_selector_resumes_running_child(self):
        bb=[{"name":"count","type":"I32","initial":0}]
        first={"type":"Sequence","children":[{"type":"AddBlackboard","slot":"count","value":1},{"type":"Failure"}]}
        v=self.vm({"type":"Selector","children":[first,{"type":"WaitTurns","turns":2}]},bb)
        self.assertEqual(v.tick(obs(0)).blackboard,(1,))
        self.assertEqual(v.tick(obs(1)).blackboard,(1,))

    def test_05_reactive_selector_priority_preempts(self):
        bb=[{"name":"reservation","type":"I32","initial":0}]
        emergency={"type":"Sequence","children":[{"type":"Condition","lhs":{"feature":"STEP"},"op":"GE","rhs":2},{"type":"Success"}]}
        plan={"type":"Sequence","lifecycle":{"on_halted":[{"op":"clear","slot":"reservation"}]},"children":[{"type":"SetBlackboard","slot":"reservation","value":7},{"type":"WaitTurns","turns":20}]}
        v=self.vm({"type":"ReactiveSelector","children":[emergency,plan]},bb)
        self.assertEqual(v.tick(obs(0)).blackboard,(7,)); self.assertEqual(v.tick(obs(1)).blackboard,(7,))
        r=v.tick(obs(2)); self.assertEqual(r.status,bkg.STATUS_SUCCESS); self.assertEqual(r.blackboard,(0,))

    def test_06_if_latches_selected_branch_while_running(self):
        bb=[{"name":"else_hit","type":"I32","initial":0}]
        tree={"type":"If","condition":{"type":"Condition","lhs":{"feature":"STEP"},"op":"LT","rhs":1},"then":{"type":"WaitTurns","turns":2},"else":{"type":"SetBlackboard","slot":"else_hit","value":1}}
        v=self.vm(tree,bb); self.assertEqual(v.tick(obs(0)).status,bkg.STATUS_RUNNING)
        r=v.tick(obs(1)); self.assertEqual(r.status,bkg.STATUS_RUNNING); self.assertEqual(r.blackboard,(0,))
        self.assertEqual(v.tick(obs(2)).status,bkg.STATUS_SUCCESS)

    def test_07_invert(self):
        self.assertEqual(self.vm({"type":"Invert","child":{"type":"Success"}}).tick(obs()).status,bkg.STATUS_FAILURE)
        self.assertEqual(self.vm({"type":"Invert","child":{"type":"Failure"}}).tick(obs()).status,bkg.STATUS_SUCCESS)

    def test_08_force_success_failure(self):
        self.assertEqual(self.vm({"type":"ForceSuccess","child":{"type":"Failure"}}).tick(obs()).status,bkg.STATUS_SUCCESS)
        self.assertEqual(self.vm({"type":"ForceFailure","child":{"type":"Success"}}).tick(obs()).status,bkg.STATUS_FAILURE)
        self.assertEqual(self.vm({"type":"ForceSuccess","child":{"type":"WaitTurns","turns":3}}).tick(obs()).status,bkg.STATUS_RUNNING)

    def test_09_turn_timeout_uses_game_steps(self):
        v=self.vm({"type":"TurnTimeout","turns":2,"child":{"type":"WaitTurns","turns":20}})
        self.assertEqual(v.tick(obs(5)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(6)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(7)).status,bkg.STATUS_FAILURE)

    def test_10_retry_boundary_and_success(self):
        cond={"type":"Condition","lhs":{"feature":"STEP"},"op":"GE","rhs":2}
        v=self.vm({"type":"Retry","n":3,"child":cond})
        self.assertEqual(v.tick(obs(0)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(1)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(2)).status,bkg.STATUS_SUCCESS)
        v=self.vm({"type":"Retry","n":2,"child":{"type":"Failure"}})
        self.assertEqual(v.tick(obs(0)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(1)).status,bkg.STATUS_FAILURE)

    def test_11_repeat_boundary(self):
        v=self.vm({"type":"Repeat","n":3,"child":{"type":"Success"}})
        self.assertEqual([v.tick(obs(i)).status for i in range(3)],[bkg.STATUS_RUNNING,bkg.STATUS_RUNNING,bkg.STATUS_SUCCESS])

    def test_12_keep_running_until_failure(self):
        v=self.vm({"type":"KeepRunningUntilFailure","child":{"type":"Success"}})
        self.assertEqual(v.tick(obs()).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(1)).status,bkg.STATUS_RUNNING)
        self.assertEqual(self.vm({"type":"KeepRunningUntilFailure","child":{"type":"Failure"}}).tick(obs()).status,bkg.STATUS_SUCCESS)

    def test_13_typed_blackboard_set_add_read(self):
        bb=[{"name":"x","type":"I32","initial":2}]
        tree={"type":"Sequence","children":[{"type":"AddBlackboard","slot":"x","value":5},{"type":"Condition","lhs":{"blackboard":"x"},"op":"EQ","rhs":7}]}
        r=self.vm(tree,bb).tick(obs()); self.assertEqual(r.status,bkg.STATUS_SUCCESS); self.assertEqual(r.blackboard,(7,))

    def test_14_invalid_blackboard_types_rejected(self):
        with self.assertRaises(bkg.PolicyError): self.vm({"type":"SetBlackboard","slot":"b","value":2},[{"name":"b","type":"BOOL","initial":0}])
        with self.assertRaises(bkg.PolicyError): self.vm({"type":"AddBlackboard","slot":"b","value":1},[{"name":"b","type":"BOOL","initial":0}])

    def test_15_precondition_failure_success_skip(self):
        for key,expected in (("failure_if",bkg.STATUS_FAILURE),("success_if",bkg.STATUS_SUCCESS),("skip_if",bkg.STATUS_SUCCESS)):
            tree={"type":"Failure","lifecycle":{key:{"lhs":{"feature":"ALWAYS"},"op":"EQ","rhs":1}}}
            self.assertEqual(self.vm(tree).tick(obs()).status,expected)

    def test_16_while_precondition_initial_skip_and_running_halt(self):
        bb=[{"name":"x","type":"I32","initial":0}]
        tree={"type":"WaitTurns","turns":10,"lifecycle":{"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":2},"on_halted":[{"op":"set","slot":"x","value":9}]}}
        v=self.vm(tree,bb); self.assertEqual(v.tick(obs(0)).status,bkg.STATUS_RUNNING); self.assertEqual(v.tick(obs(1)).status,bkg.STATUS_RUNNING)
        r=v.tick(obs(2)); self.assertEqual(r.status,bkg.STATUS_SUCCESS); self.assertEqual(r.blackboard,(9,))
        tree2={"type":"Failure","lifecycle":{"while":{"lhs":{"feature":"ALWAYS"},"op":"EQ","rhs":0}}}
        self.assertEqual(self.vm(tree2).tick(obs()).status,bkg.STATUS_SUCCESS)

    def test_17_postconditions_success_failure_post(self):
        bb=[{"name":"x","type":"I32","initial":0}]
        s={"type":"Success","lifecycle":{"on_success":[{"op":"set","slot":"x","value":2}],"post":[{"op":"add","slot":"x","value":3}]}}
        f={"type":"Failure","lifecycle":{"on_failure":[{"op":"set","slot":"x","value":4}],"post":[{"op":"add","slot":"x","value":5}]}}
        self.assertEqual(self.vm(s,bb).tick(obs()).blackboard,(5,)); self.assertEqual(self.vm(f,bb).tick(obs()).blackboard,(9,))

    def test_18_episode_reset_clears_execution_and_blackboard(self):
        bb=[{"name":"x","type":"I32","initial":3}]
        v=self.vm({"type":"Sequence","children":[{"type":"SetBlackboard","slot":"x","value":8},{"type":"WaitTurns","turns":5}]},bb)
        self.assertEqual(v.tick(obs()).blackboard,(8,)); v.reset(); self.assertEqual(tuple(v.state.blackboard),(3,)); self.assertTrue(all(x.a==0 and x.b==0 for x in v.state.node_state))

    def test_19_unit_intent_priority_and_dfs_tie(self):
        tree={"type":"Parallel","success_threshold":2,"failure_threshold":1,"children":[{"type":"EmitUnitIntent","unit":0,"action":"NORTH","priority":10},{"type":"EmitUnitIntent","unit":0,"action":"SOUTH","priority":20}]}
        self.assertEqual(self.vm(tree).tick(obs()).action.units[0][0],bkg.UNIT_OPS["SOUTH"])
        tree["children"][1]["priority"]=10
        self.assertEqual(self.vm(tree).tick(obs()).action.units[0][0],bkg.UNIT_OPS["NORTH"])

    def test_20_market_order_priority_ordering(self):
        children=[{"type":"EmitMarketIntent","action":"HIRE","priority":10},{"type":"EmitMarketIntent","action":"BUY_LAND","priority":30},{"type":"EmitMarketIntent","action":"HIRE","priority":20}]
        r=self.vm({"type":"Parallel","success_threshold":3,"failure_threshold":1,"children":children}).tick(obs())
        self.assertEqual([x[0] for x in r.action.market],[bkg.MARKET_OPS["BUY_LAND"],bkg.MARKET_OPS["HIRE"],bkg.MARKET_OPS["HIRE"]])

    def test_21_market_order_cap(self):
        children=[{"type":"EmitMarketIntent","action":"HIRE","priority":i} for i in range(5)]
        r=self.vm({"type":"Parallel","success_threshold":5,"failure_threshold":1,"children":children}).tick(obs(),max_market_orders=2)
        self.assertEqual(len(r.action.market),2); self.assertEqual(r.action.market[0][0],bkg.MARKET_OPS["HIRE"])

    def test_22_parallel_threshold_and_running_sibling_halt(self):
        bb=[{"name":"halted","type":"I32","initial":0}]
        wait={"type":"WaitTurns","turns":20,"lifecycle":{"on_halted":[{"op":"set","slot":"halted","value":1}]}}
        tree={"type":"Parallel","success_threshold":1,"failure_threshold":2,"children":[wait,{"type":"Success"}]}
        r=self.vm(tree,bb).tick(obs()); self.assertEqual(r.status,bkg.STATUS_SUCCESS); self.assertEqual(r.blackboard,(1,))
        tree={"type":"Parallel","success_threshold":2,"failure_threshold":1,"children":[{"type":"Failure"},{"type":"WaitTurns","turns":2}]}
        self.assertEqual(self.vm(tree).tick(obs()).status,bkg.STATUS_FAILURE)

    def test_23_nested_decorators(self):
        tree={"type":"Invert","child":{"type":"ForceFailure","child":{"type":"Success"}}}
        self.assertEqual(self.vm(tree).tick(obs()).status,bkg.STATUS_SUCCESS)

    def test_24_bounded_loop_one_body_per_tick(self):
        v=self.vm({"type":"Repeat","n":255,"child":{"type":"Success"}}); r=v.tick(obs()); self.assertEqual(r.status,bkg.STATUS_RUNNING); self.assertLessEqual(v.state.visits,2)
        v=self.vm({"type":"KeepRunningUntilFailure","child":{"type":"Success"}}); r=v.tick(obs()); self.assertEqual(r.status,bkg.STATUS_RUNNING); self.assertLessEqual(v.state.visits,2)

    def test_25_serialization_roundtrip_preserves_runtime(self):
        src=policy({"type":"Sequence","children":[{"type":"Success"},{"type":"WaitTurns","turns":2}]})
        cp=bkg.compile_policy(src)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"p.bkgc"; bkg.save_compiled(cp,path); cp2=bkg.load_compiled(path)
            self.assertEqual(cp.canonical_hash,cp2.canonical_hash); self.assertEqual(cp.compiled_bytes,cp2.compiled_bytes)
            a=bkg.VM(cp).tick(obs()).status; c=bkg.VM(cp2).tick(obs()).status; self.assertEqual(a,c)

    def test_26_canonical_policy_hash_stable(self):
        src=policy({"type":"Success"}); h=bkg.policy_hash(src); canonical=bkg.canonical_policy(src)
        self.assertEqual(canonical["semantic_hash"],h); self.assertEqual(bkg.policy_hash(canonical),h)
        reordered=json.loads(json.dumps(src,sort_keys=True)); self.assertEqual(bkg.policy_hash(reordered),h)

    def test_27_hidden_or_unknown_features_rejected(self):
        bad={"type":"Condition","lhs":{"feature":"OPP_SHED","arg":0},"op":"GT","rhs":0}
        with self.assertRaises(bkg.PolicyError): self.vm(bad)
        bad={"type":"Condition","lhs":{"feature":"RNG_SEED"},"op":"GT","rhs":0}
        with self.assertRaises(bkg.PolicyError): self.vm(bad)

    def test_28_compiled_validation_rejects_cycle(self):
        cp=bkg.compile_policy(policy({"type":"Sequence","children":[{"type":"Success"}]}))
        cp.child_refs[0]=0
        with self.assertRaises(bkg.PolicyError): bkg.validate_compiled(cp)

    def test_29_trace_contains_node_results(self):
        r=self.vm({"type":"Sequence","children":[{"type":"Success"},{"type":"Failure"}]}).tick(obs(4),trace=True)
        self.assertGreaterEqual(len(r.trace),3); self.assertEqual(r.trace[-1][0],4); self.assertEqual(r.trace[-1][3],"FAILURE")

    def test_30_move_unit_to_is_multiturn_running_task(self):
        v=self.vm({"type":"MoveUnitTo","unit":0,"x":0,"y":0,"priority":100})
        r=v.tick(obs(0)); self.assertEqual(r.status,bkg.STATUS_RUNNING); self.assertEqual(r.action.units[0][0],bkg.UNIT_OPS["WEST"])
        r=v.tick(obs(1,{(bkg.FEATURES["OWN_WORKER_X"],0):0,(bkg.FEATURES["OWN_WORKER_Y"],0):0})); self.assertEqual(r.status,bkg.STATUS_SUCCESS)

    def test_31_wait_turns_exact_boundary(self):
        v=self.vm({"type":"WaitTurns","turns":3}); self.assertEqual([v.tick(obs(i)).status for i in range(4)],[2,2,2,0])

    def test_32_market_intent_capacity_keeps_best_16(self):
        children=[{"type":"EmitMarketIntent","action":"SELL","item":"WHEAT","quantity":i+1,"priority":i} for i in range(20)]
        v=self.vm({"type":"Sequence","children":children})
        r=v.tick(obs(),max_market_orders=16)
        self.assertEqual(len(r.action.market),16)
        self.assertEqual([x[2] for x in r.action.market],list(range(20,4,-1)))
        self.assertEqual([c.priority for c in v.state.market_claims],list(range(19,3,-1)))

    def test_33_emit_unit_intent_missing_unit_fails(self):
        v=self.vm({"type":"EmitUnitIntent","unit":1,"action":"NORTH","priority":1})
        r=v.tick(obs())
        self.assertEqual(r.status,bkg.STATUS_FAILURE)
        self.assertEqual(r.action.n_units,1)
        self.assertEqual(r.action.units[0][0],bkg.UNIT_OPS["PASS"])

    def test_34_decompile_recompile_preserves_compiled_semantics(self):
        src=policy({"type":"ReactiveSelector","children":[
            {"type":"Condition","lhs":{"feature":"STEP"},"op":"GE","rhs":4},
            {"type":"WaitTurns","turns":3,"lifecycle":{"while":{"lhs":{"feature":"STEP"},"op":"LT","rhs":7},"on_halted":[{"op":"clear","slot":"x"}]}}
        ]},[{"name":"x","type":"I32","initial":2}])
        cp=bkg.compile_policy(src)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"p.bkgc";bkg.save_compiled(cp,path);loaded=bkg.load_compiled(path)
            rebuilt=bkg.compile_policy(bkg.decompile_policy(loaded))
            self.assertEqual(bkg.semantic_payload_bytes(loaded),bkg.semantic_payload_bytes(rebuilt))

    def test_35_official_agent_wrapper_resets_on_step_rollback(self):
        src=policy({"type":"WaitTurns","turns":5})
        agent=bkg.KaggricultureAgent(src)
        tiles=[[None for _ in range(10)] for _ in range(10)]
        base={"player":0,"farms":[{"tiles":tiles},{"tiles":tiles}]}
        agent({**base,"step":0});agent({**base,"step":1})
        self.assertEqual(agent.vm.state.ticks,2)
        agent({**base,"step":0})
        self.assertEqual(agent.vm.state.ticks,1)

    def test_36_blackboard_add_overflow_faults_closed(self):
        bb=[{"name":"x","type":"I32","initial":2147483647}]; v=self.vm({"type":"AddBlackboard","slot":"x","value":1},bb)
        r=v.tick(obs()); self.assertEqual(r.status,bkg.STATUS_FAILURE); self.assertEqual(r.fault,2); self.assertEqual(r.action.market,[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
