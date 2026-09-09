"""Fresh BUILD question after v1's independent canary rejected universal success."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize
from .agent import controller
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
from kaggriculture_bonsai.agents.gold_v25.extract import sha


def main():
    out=OUT/'round2';out.mkdir(exist_ok=True)
    seeds=list(range(950000,950032));dev=list(range(960000,960032))
    candidates=[{'id':100+i,'start':s,'end':718,'mode':9} for i,s in enumerate((0,72,144,216,288,360,432,504,576,648,672,696))]
    manifest={'BUILD':seeds,'INTERNAL_DEV':dev,'official_canary':list(range(980000,980032)),'seats':[0,1],
              'question':'Stable projected-value ordering for all-SELL/no-op market sequences before the fixed step708 endgame liquidation. Mixed purchase sequences remain untouched.',
              'candidates':candidates,'canary_v1_used_only_as_rejection':True}
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'round2_parent')
    screen=run('/tmp/gold_plus_cuda',candidates,seeds[:4],'round2_screen')
    ranked=sorted(candidates,key=lambda c:(-sum(r['outcome']=='W' for r in screen if r['candidate']==c['id']),-min(r['margin'] for r in screen if r['candidate']==c['id']),c['id']))
    build=run('/tmp/gold_plus_cuda',ranked[:4],seeds,'round2_build')
    summaries=[{'candidate':c,**summarize(parent,[r for r in build if r['candidate']==c['id']])} for c in ranked[:4]]
    (out/'build.json').write_text(json.dumps({'parent':parent,'screen':screen,'build':build,'summary':summaries},separators=(',',':'))+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
    eligible=[s for s in summaries if s['improved_wtl']=={'W':64,'T':0,'L':0} and s['protected_regressions']==0]
    if not eligible:return
    chosen=max(eligible,key=lambda s:(s['minimum_margin'],-s['candidate']['id']))['candidate']
    gate=controller(chosen['start'],chosen['end'],chosen['mode']);bkg.save_compiled(gate,out/'specialist.bkgc')
    identity={'identity':'BKG4_GOLD_V25_PLUS_MARKET_V2','parent':'BKG4_GOLD_V25','candidate':chosen,
              'status':'FROZEN_BEFORE_INTERNAL_DEV','selection':'BUILD only: all wins, greatest minimum margin',
              'gate_semantic_hash':gate.semantic_hash,'gate_sha256':sha((out/'specialist.bkgc').read_bytes()),
              'gate_nodes':len(gate.nodes),'gate_bb':len(gate.bb),'gate_node_state_slots':gate.state_count,
              'specialists':2,'route_pack_changes':0}
    (out/'composition.json').write_text(json.dumps(identity,indent=2)+'\n')
    p=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],dev,'round2_dev_parent')
    improved=run('/tmp/gold_plus_cuda',[chosen],dev,'round2_dev')
    summary=summarize(p,improved)
    cpu=run('/tmp/gold_plus_cpu',[chosen],dev[:2],'round2_cpu');assert cpu==improved[:4]
    (out/'internal_dev.json').write_text(json.dumps({'summary':summary,'parent':p,'improved':improved,'cpu_cuda_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
