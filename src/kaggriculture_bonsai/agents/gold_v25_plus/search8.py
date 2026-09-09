"""Frozen seed-reserve composition; BUILD precedes fresh DEV."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out=OUT/'round8';manifest=json.loads((out/'panels.json').read_text())
    candidate={'id':800,'start':144,'end':647,'mode':1025}
    (out/'frozen_candidate.json').write_text(json.dumps({'candidate':candidate,'status':'FROZEN_BEFORE_INTERNAL_DEV',
        'question':'Reduce only seed purchases exceeding the worst remaining own-route inventory requirement, accounting for both late branches.'},indent=2)+'\n')
    panels={}
    for label,seeds in [('BUILD',manifest['BUILD']),('prior_BUILD',list(range(990000,990064))+list(range(1020000,1020064))+list(range(1050000,1050128)))]:
        parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'seed_parent_'+label)
        rows=run('/tmp/gold_plus_cuda',[candidate],seeds,'seed_'+label)
        panels[label]={'parent':parent,'improved':rows,'summary':summarize(parent,rows)}
    cpu=run('/tmp/gold_plus_cpu',[candidate],[990057,1020047,1050019,1080046],'seed_cpu')
    gpu=run('/tmp/gold_plus_cuda',[candidate],[990057,1020047,1050019,1080046],'seed_gpu');assert cpu==gpu
    (out/'build.json').write_text(json.dumps({'panels':panels,'cpu_cuda_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps({k:v['summary'] for k,v in panels.items()},indent=2),flush=True)
    if not all(all(r['outcome']=='W' for r in v['improved']) for v in panels.values()):return
    seeds=manifest['INTERNAL_DEV']
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'seed_dev_parent')
    rows=run('/tmp/gold_plus_cuda',[candidate],seeds,'seed_dev')
    summary=summarize(parent,rows)
    (out/'internal_dev.json').write_text(json.dumps({'parent':parent,'improved':rows,'summary':summary},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
