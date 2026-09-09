"""Bounded fresh-BUILD sale-timing experiment; held-out cases are not inputs."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize
from .agent import controller
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.agents.gold_v25.extract import sha


def main(out=OUT/'round3', candidates=None, diagnosis_path=None, rationale=None, identity_name='BKG4_GOLD_V25_PLUS_SALES_V3'):
    manifest=json.loads((out/'panels.json').read_text())
    diagnosis=json.loads((diagnosis_path or out/'diagnostic.json').read_text())
    seeds=manifest['BUILD']
    weak=list(dict.fromkeys(r['seed'] for r in sorted(diagnosis['candidate'],key=lambda r:r['margin'])))[:4]
    if candidates is None:
        candidates=[{'id':200+i,'start':s,'end':718,'mode':17} for i,s in enumerate((0,144,288,360,432,504,576,648))]
    (out/'question.json').write_text(json.dumps({'candidates':candidates,'screen_seeds':weak,
        'rationale':rationale or 'The weakest BUILD case loses thousands before late routing, while saleable products wait in the shed. Sell non-input products on all-SELL/no-op turns, preserving wheat, fertilizer, purchases, unit actions and parent state.'},indent=2)+'\n')
    screen=run('/tmp/gold_plus_cuda',candidates,weak,'round3_screen')
    ranked=sorted(candidates,key=lambda c:(-sum(r['outcome']=='W' for r in screen if r['candidate']==c['id']),-min(r['margin'] for r in screen if r['candidate']==c['id']),c['id']))
    build=run('/tmp/gold_plus_cuda',ranked[:4],seeds,'round3_build')
    summaries=[{'candidate':c,**summarize(diagnosis['parent'],[r for r in build if r['candidate']==c['id']])} for c in ranked[:4]]
    (out/'build.json').write_text(json.dumps({'screen':screen,'build':build,'summary':summaries},separators=(',',':'))+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
    eligible=[s for s in summaries if s['improved_wtl']=={'W':128,'T':0,'L':0} and not s['protected_regressions']]
    if not eligible:return
    chosen=max(eligible,key=lambda s:(s['minimum_margin'],-s['candidate']['id']))['candidate']
    gate=controller(chosen['start'],chosen['end'],chosen['mode']);bkg.save_compiled(gate,out/'specialist.bkgc')
    identity={'identity':identity_name,'parent':'BKG4_GOLD_V25','candidate':chosen,
        'status':'FROZEN_BEFORE_INTERNAL_DEV','selection':'BUILD only: all wins, greatest minimum margin',
        'gate_semantic_hash':gate.semantic_hash,'gate_sha256':sha((out/'specialist.bkgc').read_bytes()),
        'gate_nodes':len(gate.nodes),'gate_bb':len(gate.bb),'gate_node_state_slots':gate.state_count,'specialists':2,'route_pack_changes':0}
    (out/'composition.json').write_text(json.dumps(identity,indent=2)+'\n')
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],manifest['INTERNAL_DEV'],'round3_dev_parent')
    improved=run('/tmp/gold_plus_cuda',[chosen],manifest['INTERNAL_DEV'],'round3_dev')
    summary=summarize(parent,improved)
    cpu=run('/tmp/gold_plus_cpu',[chosen],manifest['INTERNAL_DEV'][:2],'round3_cpu');assert cpu==improved[:4]
    (out/'internal_dev.json').write_text(json.dumps({'summary':summary,'parent':parent,'improved':improved,'cpu_cuda_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
