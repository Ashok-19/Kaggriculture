from __future__ import annotations

import copy
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg as bkg
OUT=ROOT/"scratch/bonsai_kg/invalid_validation"
NATIVE=ROOT/"scratch/bonsai_kg/validate_compiled"

def base_policy():
    return {
        "schema_version":3,
        "blackboard":[{"name":"x","type":"I32","initial":0},{"name":"target","type":"TILE","initial":44}],
        "tree":{"type":"Sequence","lifecycle":{"on_success":[{"op":"set","slot":"x","value":1}]},"children":[
            {"type":"Condition","lhs":{"feature":"OWN_TILE_WATERED","arg":{"blackboard":"target"}},"op":"GE","rhs":0},
            {"type":"WaitTurns","turns":2},
            {"type":"EmitMarketIntent","action":"SELL","item":"WHEAT","quantity":1,"priority":5}
        ]}
    }

def save(cp,name):
    path=OUT/name;bkg.save_compiled(cp,path);return path

def native_ok(path):return subprocess.run([str(NATIVE),str(path)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
def python_ok(path):
    try:bkg.load_compiled(path);return True
    except bkg.PolicyError:return False

def main():
    if OUT.exists():shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    valid=bkg.compile_policy(base_policy());good=save(valid,"valid.bkgc");cases=[]
    cp=copy.deepcopy(valid);cp.values[0].id=999;cases.append(("illegal_feature_id.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[0].debug_id=99;cases.append(("bad_debug_id.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[0].child_count=0;cases.append(("bad_cardinality.bkgc",cp))
    cp=copy.deepcopy(valid);cp.child_refs[cp.nodes[0].first_child]=0;cases.append(("cycle_root_parent.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[2].state_slot=cp.nodes[0].state_slot;cases.append(("aliased_state.bkgc",cp))
    cp=copy.deepcopy(valid);cp.state_count+=1;cases.append(("unused_state.bkgc",cp))
    cp=copy.deepcopy(valid);cp.bb[0].type=99;cases.append(("bad_bb_type.bkgc",cp))
    cp=copy.deepcopy(valid);cp.effects[0].slot=31;cases.append(("bad_effect_slot.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[3].p0=99;cases.append(("bad_market_op.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[2].p0=0;cases.append(("bad_wait_bound.bkgc",cp))
    cp=copy.deepcopy(valid);cp.child_refs.append(0);cases.append(("unused_child_ref.bkgc",cp))
    cp=copy.deepcopy(valid);cp.lifecycles.append(bkg.Lifecycle());cases.append(("unused_lifecycle.bkgc",cp))
    cp=copy.deepcopy(valid);cp.values.append(bkg.ValueRef(bkg.VALUE_KINDS["CONST"],bkg.BB_TYPES["I32"],0,0,7));cases.append(("unused_value_ref.bkgc",cp))
    cp=copy.deepcopy(valid);cp.feature_args.append(bkg.FeatureArgRef(bkg.FEATURE_ARG_KINDS["CONST"],bkg.BB_TYPES["TILE"],0,43));cases.append(("unused_feature_arg.bkgc",cp))
    cp=copy.deepcopy(valid);cp.feature_args[0].kind=99;cases.append(("bad_feature_arg_kind.bkgc",cp))
    cp=copy.deepcopy(valid);cp.feature_args[0].type=bkg.BB_TYPES["UNIT"];cases.append(("bad_feature_arg_type.bkgc",cp))
    cp=copy.deepcopy(valid);cp.nodes[0].subtree_end-=1;cases.append(("bad_subtree_end.bkgc",cp))
    cp=copy.deepcopy(valid);cp.values[cp.nodes[3].p1].type=bkg.BB_TYPES["I32"];cases.append(("bad_item_value_type.bkgc",cp))

    # Compiler/decompiler canonicality: these mutations remain structurally typed and
    # receive a fresh semantic hash on save(), but cannot be produced faithfully by
    # the authoring compiler for the mutated opcode/static domain.
    dyn_bb=[{"name":"item","type":"ITEM","initial":-1},{"name":"qty","type":"I32","initial":1}]
    cp=bkg.compile_policy({"schema_version":3,"blackboard":dyn_bb,"tree":{"type":"EmitMarketIntent","action":"SELL","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"}}});cp.nodes[0].p0=bkg.MARKET_OPS["HIRE"];cases.append(("hire_hidden_dynamic_operands.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"blackboard":dyn_bb,"tree":{"type":"EmitMarketIntent","action":"SELL","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"}}});cp.nodes[0].p0=bkg.MARKET_OPS["BUY_LAND"];cases.append(("buy_land_hidden_dynamic_operands.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitMarketIntent","action":"HIRE"}});cp.values[cp.nodes[0].p2].value=2;cases.append(("hire_hidden_quantity_two.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitMarketIntent","action":"BUY_LAND"}});cp.values[cp.nodes[0].p1].value=bkg.ITEM_ID["WHEAT"];cases.append(("buy_land_hidden_item_wheat.bkgc",cp))

    unit_bb=[{"name":"unit","type":"UNIT","initial":0},{"name":"item","type":"ITEM","initial":0},{"name":"qty","type":"I32","initial":1}]
    cp=bkg.compile_policy({"schema_version":3,"blackboard":unit_bb,"tree":{"type":"EmitUnitIntent","unit":{"blackboard":"unit"},"action":"PICKUP","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"}}});cp.nodes[0].p1=bkg.UNIT_OPS["WATER"];cases.append(("ordinary_unit_hidden_dynamic_operands.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"blackboard":unit_bb,"tree":{"type":"EmitUnitIntent","unit":{"blackboard":"unit"},"action":"PICKUP","item":{"blackboard":"item"},"quantity":{"blackboard":"qty"}}});cp.nodes[0].p1=bkg.UNIT_OPS["PLANT"];cases.append(("plant_dynamic_quantity.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitUnitIntent","unit":0,"action":"PLANT","item":"WHEAT"}});cp.values[cp.nodes[0].p2].value=bkg.ITEM_ID["GOOSE"];cases.append(("plant_const_non_crop.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitUnitIntent","unit":0,"action":"PICKUP","item":"WHEAT","quantity":1}});cp.values[cp.nodes[0].p2].value=-1;cases.append(("pickup_const_item_none.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitUnitIntent","unit":0,"action":"PLACE","item":"WHEAT","quantity":1}});cp.values[cp.nodes[0].p3].value=0;cases.append(("place_const_quantity_zero.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitUnitIntent","unit":0,"action":"WATER"}});cp.values[cp.nodes[0].p2].value=bkg.ITEM_ID["WHEAT"];cases.append(("ordinary_unit_hidden_item_wheat.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitUnitIntent","unit":0,"action":"WATER"}});cp.values[cp.nodes[0].p3].value=2;cases.append(("ordinary_unit_hidden_quantity_two.bkgc",cp))

    def mutated_market(action,item,bad_item=None,bad_qty=None):
        cp=bkg.compile_policy({"schema_version":3,"tree":{"type":"EmitMarketIntent","action":action,"item":item,"quantity":1}})
        if bad_item is not None:cp.values[cp.nodes[0].p1].value=bad_item
        if bad_qty is not None:cp.values[cp.nodes[0].p2].value=bad_qty
        return cp
    cases.append(("buy_seed_const_non_crop.bkgc",mutated_market("BUY_SEED","WHEAT",bkg.ITEM_ID["GOOSE"])))
    cases.append(("buy_product_const_carrot.bkgc",mutated_market("BUY_PRODUCT","WHEAT",bkg.ITEM_ID["CARROT"])))
    cases.append(("buy_animal_const_wheat.bkgc",mutated_market("BUY_ANIMAL","GOOSE",bkg.ITEM_ID["WHEAT"])))
    cases.append(("sell_const_animal.bkgc",mutated_market("SELL","WHEAT",bkg.ITEM_ID["GOOSE"])))
    cases.append(("sell_const_quantity_zero.bkgc",mutated_market("SELL","WHEAT",bad_qty=0)))

    cp=bkg.compile_policy({"schema_version":3,"blackboard":[{"name":"flag","type":"BOOL","initial":0}],"tree":{"type":"SetBlackboard","slot":"flag","value":1}});cp.values[cp.nodes[0].p1].value=2;cases.append(("set_bool_const_two.bkgc",cp))
    cp=bkg.compile_policy({"schema_version":3,"blackboard":[{"name":"flag","type":"BOOL","initial":0}],"tree":{"type":"Success","lifecycle":{"on_success":[{"op":"set","slot":"flag","value":1}]}}});cp.values[cp.effects[0].value].value=2;cases.append(("effect_set_bool_const_two.bkgc",cp))

    invalid=[save(cp,name) for name,cp in cases]

    raw=bytearray(good.read_bytes())
    trailing=OUT/"trailing_bytes.bkgc";trailing.write_bytes(raw+b"X");invalid.append(trailing)
    bad_board=OUT/"bad_board_size.bkgc";x=bytearray(raw);struct.pack_into("<H",x,bkg.HEADER.size-2,11);bad_board.write_bytes(x);invalid.append(bad_board)
    bb_off=bkg.HEADER.size+len(valid.nodes)*bkg.NODE_STRUCT.size+len(valid.child_refs)*2
    bb_padding=OUT/"bb_padding.bkgc";x=bytearray(raw);x[bb_off+1]=1;bb_padding.write_bytes(x);invalid.append(bb_padding)
    pred_off=bb_off+len(valid.bb)*bkg.BB_STRUCT.size+len(valid.values)*bkg.VALUE_STRUCT.size+len(valid.feature_args)*bkg.FEATURE_ARG_STRUCT.size
    effect_off=pred_off+len(valid.predicates)*bkg.PRED_STRUCT.size+len(valid.lifecycles)*bkg.LIFECYCLE_STRUCT.size
    effect_reserved=OUT/"effect_reserved.bkgc";x=bytearray(raw);x[effect_off+2]=1;effect_reserved.write_bytes(x);invalid.append(effect_reserved)
    semantic_tamper=OUT/"semantic_tamper.bkgc";x=bytearray(raw);wait_p0=bkg.HEADER.size+2*bkg.NODE_STRUCT.size+12;struct.pack_into("<i",x,wait_p0,3);semantic_tamper.write_bytes(x);invalid.append(semantic_tamper)

    failures=[]
    if not python_ok(good):failures.append("python rejected valid")
    if not native_ok(good):failures.append("native rejected valid")
    for path in invalid:
        if python_ok(path):failures.append(f"python accepted {path.name}")
        if native_ok(path):failures.append(f"native accepted {path.name}")
    print(f"valid_cases=1 invalid_cases={len(invalid)} loader_checks={2*(1+len(invalid))} failures={len(failures)}")
    for x in failures:print("FAIL",x)
    return 1 if failures else 0

if __name__=="__main__":raise SystemExit(main())
