"""Freeze and screen the BUILD-derived wheat/structure composition."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out=OUT/'round6';out.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'round5/panels.json').read_text())
    manifest['candidate']={'id':600,'start':144,'end':647,'mode':257}
    manifest['status']='FROZEN_BEFORE_INTERNAL_DEV'
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    candidate=manifest['candidate']
    panels={}
    for label,seeds in [('BUILD',manifest['BUILD']),('diagnostic_BUILD',manifest['diagnostic_BUILD'])]:
        parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'repair_parent_'+label)
        rows=run('/tmp/gold_plus_cuda',[candidate],seeds,'repair_'+label)
        panels[label]={'parent':parent,'improved':rows,'summary':summarize(parent,rows)}
    cpu=run('/tmp/gold_plus_cpu',[candidate],[990057,1020047],'repair_cpu')
    gpu=run('/tmp/gold_plus_cuda',[candidate],[990057,1020047],'repair_gpu')
    assert cpu==gpu
    (out/'build.json').write_text(json.dumps({'panels':panels,'cpu_cuda_mismatches':0},separators=(',',':'))+'\n')
    print(json.dumps({k:v['summary'] for k,v in panels.items()},indent=2),flush=True)
    if not all(all(r['outcome']=='W' for r in v['improved']) for v in panels.values()):return
    seeds=manifest['INTERNAL_DEV']
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'repair_dev_parent')
    rows=run('/tmp/gold_plus_cuda',[candidate],seeds,'repair_dev')
    summary=summarize(parent,rows)
    (out/'internal_dev.json').write_text(json.dumps({'parent':parent,'improved':rows,'summary':summary},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
