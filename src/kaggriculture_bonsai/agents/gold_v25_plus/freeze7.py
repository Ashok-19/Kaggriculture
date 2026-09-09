"""Serialize the already-frozen round-7 composition, without reselection."""
import json
import argparse
from .benchmark import OUT
from .repair_controller import controller
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.agents.gold_v25.extract import sha


def main(round_number=7):
    out=OUT/f'round{round_number}'
    assert json.loads((out/'internal_dev.json').read_text())['summary']['improved_wtl']=={'W':128,'T':0,'L':0}
    candidate=json.loads((out/'frozen_candidate.json').read_text())['candidate']
    gate=controller(seed_reserve=round_number==8);bkg.save_compiled(gate,out/'specialist.bkgc')
    identity={'identity':f'BKG4_GOLD_V25_PLUS_RECOVERY_V{round_number}','parent':'BKG4_GOLD_V25','candidate':candidate,
        'status':'FROZEN_AFTER_PASSING_INTERNAL_DEV','gate_sha256':sha((out/'specialist.bkgc').read_bytes()),
        'gate_semantic_hash':gate.semantic_hash,'gate_nodes':len(gate.nodes),'gate_bb':len(gate.bb),
        'gate_node_state_slots':gate.state_count,'specialists':6 if round_number==8 else 5,'route_pack_changes':0,
        'repair_state':'Per player: last step, 100 missed-construction entries, at most five farmer actions. Reset on zero or repeated/backward step.',
        'wrapper_abi_marker':-800 if round_number==8 else -700}
    (out/'composition.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,choices=(7,8),default=7);args=parser.parse_args();main(args.round)
