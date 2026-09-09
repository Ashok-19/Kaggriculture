"""CUDA BUILD screening of bounded own-pickup wheat coverage."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out=OUT/'round5';out.mkdir(exist_ok=True)
    seeds=list(range(1020000,1020064))
    manifest={'BUILD':seeds,'diagnostic_BUILD':list(range(990000,990064)),
              'INTERNAL_DEV':list(range(1030000,1030064)),
              'official_canary':list(range(1040000,1040032)),'seats':[0,1],
              'question':'Cover next own-route wheat pickup shortage with a bounded market purchase; keep unit actions and v2 market fallback unchanged.'}
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    candidates=[{'id':500+i,'start':start,'end':end,'mode':129}
        for i,(start,end) in enumerate((s,e) for s in (144,216,288) for e in (360,528,647))]
    screen_seeds=[990057,990042,990017,990031]+seeds[:4]
    screen=run('/tmp/gold_plus_cuda',candidates,screen_seeds,'feed_screen')
    ranked=sorted(candidates,key=lambda c:(-sum(r['outcome']=='W' for r in screen if r['candidate']==c['id']),-min(r['margin'] for r in screen if r['candidate']==c['id']),c['id']))
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'feed_parent')
    rows=run('/tmp/gold_plus_cuda',ranked[:3],seeds,'feed_build')
    summaries=[{'candidate':c,**summarize(parent,[r for r in rows if r['candidate']==c['id']])} for c in ranked[:3]]
    cpu=run('/tmp/gold_plus_cpu',ranked[:1],[990057,990031],'feed_cpu')
    gpu=run('/tmp/gold_plus_cuda',ranked[:1],[990057,990031],'feed_gpu')
    assert cpu==gpu
    (out/'build.json').write_text(json.dumps({'candidates':candidates,'screen':screen,'parent':parent,'build':rows,'summary':summaries,'cpu_cuda_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
