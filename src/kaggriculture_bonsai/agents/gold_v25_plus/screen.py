"""Screen the BUILD-selected frozen candidate; never tune from DEV results."""
import json
from .benchmark import OUT, panels
from .search import run
from .agent import controller
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.agents.gold_v25.extract import sha


def summarize(parent, improved):
    assert [(r['seed'],r['seat']) for r in parent]==[(r['seed'],r['seat']) for r in improved]
    transitions={a+b:sum(p['outcome']==a and r['outcome']==b for p,r in zip(parent,improved)) for a in 'WTL' for b in 'WTL'}
    return {'games':len(improved),'parent_wtl':{o:sum(r['outcome']==o for r in parent) for o in 'WTL'},
            'improved_wtl':{o:sum(r['outcome']==o for r in improved) for o in 'WTL'},
            'transitions':transitions,'protected_regressions':sum(transitions[t] for t in ('WL','WT','TL')),
            'minimum_margin':min(r['margin'] for r in improved)}


def main():
    search=json.loads((OUT/'search.json').read_text())
    eligible=[s for s in search['summary'] if s['wtl']=={'W':32,'T':0,'L':0}]
    chosen=max(eligible,key=lambda s:(s['min_margin'],-s['candidate']['id']))['candidate']
    gate=controller(chosen['start'],chosen['end'],chosen['mode'])
    bkg.save_compiled(gate,OUT/'specialist.bkgc')
    identity={'identity':'BKG4_GOLD_V25_PLUS_ENDGAME_V1','parent':'BKG4_GOLD_V25','candidate':chosen,
              'selection':'BUILD only: all wins, greatest minimum margin, lowest id tie-break',
              'gate_semantic_hash':gate.semantic_hash,'gate_nodes':len(gate.nodes),'gate_bb':len(gate.bb),
              'gate_node_state_slots':gate.state_count,'gate_sha256':sha((OUT/'specialist.bkgc').read_bytes()),
              'status':'FROZEN_BEFORE_INTERNAL_DEV','specialists':1,'route_pack_changes':0}
    (OUT/'composition.json').write_text(json.dumps(identity,indent=2)+'\n')
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],panels()['INTERNAL_DEV'],'dev_parent')
    improved=run('/tmp/gold_plus_cuda',[chosen],panels()['INTERNAL_DEV'],'dev_candidate')
    cpu=run('/tmp/gold_plus_cpu',[chosen],panels()['INTERNAL_DEV'][:2],'dev_parity')
    assert cpu==improved[:4]
    summary=summarize(parent,improved)
    (OUT/'internal_dev.json').write_text(json.dumps({'summary':summary,'parent':parent,'improved':improved,'cpu_cuda_trace_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
