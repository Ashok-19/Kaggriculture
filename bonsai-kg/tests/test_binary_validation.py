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
        "schema_version":2,
        "blackboard":[{"name":"x","type":"I32","initial":0}],
        "tree":{
            "type":"Sequence",
            "lifecycle":{"on_success":[{"op":"set","slot":"x","value":1}]},
            "children":[
                {"type":"Condition","lhs":{"feature":"STEP"},"op":"GE","rhs":0},
                {"type":"WaitTurns","turns":2},
                {"type":"EmitMarketIntent","action":"SELL","item":"WHEAT","quantity":1,"priority":5}
            ]
        }
    }


def save(cp,name):
    path=OUT/name
    bkg.save_compiled(cp,path)
    return path


def native_ok(path):
    return subprocess.run([str(NATIVE),str(path)],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def python_ok(path):
    try:bkg.load_compiled(path);return True
    except bkg.PolicyError:return False


def main():
    if OUT.exists():shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    valid=bkg.compile_policy(base_policy())
    good=save(valid,"valid.bkgc")
    cases=[]

    cp=copy.deepcopy(valid);cp.nodes[1].p1=999;cases.append(("illegal_feature.bkgc",cp))
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

    invalid=[]
    for name,cp in cases:invalid.append(save(cp,name))
    trailing=OUT/"trailing_bytes.bkgc";trailing.write_bytes(good.read_bytes()+b"X");invalid.append(trailing)
    raw=bytearray(good.read_bytes())
    node_reserved=OUT/"node_reserved.bkgc"; x=bytearray(raw); x[bkg.HEADER.size+10]=1; node_reserved.write_bytes(x); invalid.append(node_reserved)
    bad_max_intents=OUT/"bad_max_intents.bkgc"; x=bytearray(raw); current=struct.unpack_from("<H",x,bkg.HEADER.size-2)[0]; struct.pack_into("<H",x,bkg.HEADER.size-2,current+1); bad_max_intents.write_bytes(x); invalid.append(bad_max_intents)
    bb_off=bkg.HEADER.size+len(valid.nodes)*bkg.NODE_STRUCT.size+len(valid.child_refs)*2
    bb_padding=OUT/"bb_padding.bkgc"; x=bytearray(raw); x[bb_off+1]=1; bb_padding.write_bytes(x); invalid.append(bb_padding)
    pred_off=bb_off+len(valid.bb)*bkg.BB_STRUCT.size+len(valid.values)*bkg.VALUE_STRUCT.size
    pred_padding=OUT/"predicate_padding.bkgc"; x=bytearray(raw); x[pred_off+5]=1; pred_padding.write_bytes(x); invalid.append(pred_padding)
    effect_off=pred_off+len(valid.predicates)*bkg.PRED_STRUCT.size+len(valid.lifecycles)*bkg.LIFECYCLE_STRUCT.size
    effect_reserved=OUT/"effect_reserved.bkgc"; x=bytearray(raw); x[effect_off+2]=1; effect_reserved.write_bytes(x); invalid.append(effect_reserved)

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
