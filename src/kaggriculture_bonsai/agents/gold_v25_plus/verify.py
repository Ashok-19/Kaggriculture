"""Official engine and standalone native candidate action/state verification."""
import argparse
import json
from pathlib import Path
from .benchmark import OUT, panels
from .agent import GoldV25Plus
from .feed_probe import DispatchProbe,SeedProbe
from .package import PACKAGE
from .search import run
from .screen import summarize
from kaggriculture_bonsai.agents.gold_v25.extract import reference, sha
from kaggriculture_bonsai.agents.gold_v25.native import NativeAgent, state_summary
from kaggriculture_bonsai.agents.gold_v25.validate import engine
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


class Trace:
    def __init__(self):self.value=14695981039346656037
    def word(self,n):
        for x in (int(n)&0xffffffff).to_bytes(4,'little'):self.value=((self.value^x)*1099511628211)&0xffffffffffffffff
    def action(self,a):
        for u in [a['farmer']]+a['hands']:
            self.word(bkg.UNIT_OPS[u[0]]);self.word(bkg.ITEM_ID[u[1]] if len(u)>1 else 0);self.word(u[2] if len(u)>2 else 1)
        self.word(len(a['market']))
        for o in a['market']:
            self.word(bkg.MARKET_OPS[o[0]] if o else 0);self.word(bkg.ITEM_ID[o[1]] if len(o)>1 else 0);self.word(o[2] if len(o)>2 else 1 if o else 0)


def canary(evidence=OUT):
    make=engine();identity=json.loads((evidence/'composition.json').read_text());c=identity['candidate'];rows=[]
    seeds=json.loads((evidence/'panels.json').read_text()).get('official_canary',list(range(940000,940032)))
    for seed in seeds:
        for seat in (0,1):
            parent=reference();python=SeedProbe() if c['mode']==1025 else DispatchProbe() if c['mode']==513 else GoldV25Plus(c['start'],c['end'],c['mode'])
            native=NativeAgent(PACKAGE/'gold_cpu.so',assets=PACKAGE/'assets');trace=Trace();count=0
            def own(obs,cfg=None):
                nonlocal count
                expected=python(obs,cfg);actual=native(obs,cfg)
                assert actual==expected,(seed,seat,obs['step'],expected,actual)
                assert native.state==state_summary(python.parent.chassis.players[seat])
                trace.action(actual);count+=1;return actual
            def other(obs,cfg=None):
                a=parent['agent'](obs,cfg);trace.action(a);return a
            agents=[other,other];agents[seat]=own
            try:
                env=make('kaggriculture',configuration={'episodeSteps':720,'seed':seed},debug=True)
                env.run(agents);assert count==719
                assert [s.status for s in env.state]==['DONE','DONE']
                money=[s.reward for s in env.state];margin=money[seat]-money[1-seat]
                rows.append({'seed':seed,'seat':seat,'money':money,'trace':str(trace.value),'margin':margin,'outcome':'W' if margin>0 else 'T' if margin==0 else 'L'})
                print('official',len(rows),'/64',seed,seat,margin,flush=True)
            finally:native.close()
    (evidence/'official_canary.json').write_text(json.dumps({'games':64,'engine':'1.32.7','action_state_mismatches':0,'faults':0,'cases':rows},separators=(',',':'))+'\n')


def final(evidence=OUT):
    identity=json.loads((evidence/'composition.json').read_text());candidate=identity['candidate']
    dev=json.loads((evidence/'internal_dev.json').read_text());assert dev['summary']['protected_regressions']==0
    assert dev['summary']['improved_wtl']=={'W':len(dev['improved']),'T':0,'L':0}
    # Verify the final compiled BKG4 gate on existing BUILD before opening FINAL_TARGET.
    local=json.loads((evidence/'panels.json').read_text());gate=evidence/'specialist.bkgc'
    build=run('/tmp/gold_plus_cuda',[candidate],local['BUILD'],'frozen_build',gate=gate)
    saved=json.loads((evidence/('search.json' if evidence==OUT else 'build.json')).read_text())
    old=saved['panels']['BUILD']['improved'] if candidate['mode'] in (513,1025) else [r for r in saved['build'] if r['candidate']==candidate['id']]
    assert build==old
    cpu=run('/tmp/gold_plus_cpu',[candidate],local['BUILD'][:2],'frozen_cpu',gate=gate)
    assert cpu==build[:4]
    canary=json.loads((evidence/'official_canary.json').read_text())
    cuda=run('/tmp/gold_plus_cuda',[candidate],local.get('official_canary',list(range(940000,940032))),'official_native',gate=gate)
    assert [(r['money'],r['trace']) for r in cuda]==[(r['money'],r['trace']) for r in canary['cases']]
    assert all(r['outcome']=='W' for r in canary['cases'])
    manifest=json.loads((OUT/'panels.json').read_text());assert manifest['final_target_status']=='UNOPENED'
    # Old BUILD remains protected after changing the discovery question.
    old_parent=json.loads((OUT/'parent_build.json').read_text())['cases']
    old_result=run('/tmp/gold_plus_cuda',[candidate],panels()['BUILD'],'old_build_protection',gate=gate)
    protection=summarize(old_parent,old_result);assert protection['protected_regressions']==0
    protected={'original_BUILD':protection}
    for label,path,key in (
        ('original_DEV',OUT/'internal_dev.json','parent'),
        ('round2_BUILD',OUT/'round2/build.json','parent'),
        ('round2_DEV',OUT/'round2/internal_dev.json','parent'),
        ('round6_DEV',OUT/'round6/internal_dev.json','parent'),
        ('round7_DEV',OUT/'round7/internal_dev.json','parent'),
    ):
        if not path.exists():continue
        baseline=json.loads(path.read_text())[key]
        seeds=list(dict.fromkeys(r['seed'] for r in baseline))
        result=run('/tmp/gold_plus_cuda',[candidate],seeds,label+'_protection',gate=gate)
        protected[label]=summarize(baseline,result)
    (evidence/'protected.json').write_text(json.dumps(protected,indent=2)+'\n')
    assert all(s['protected_regressions']==0 for s in protected.values())
    freeze={'composition_sha256':sha((evidence/'composition.json').read_bytes()),'gate_sha256':sha(gate.read_bytes()),
            'source_sha256':{p.name:sha(p.read_bytes()) for p in Path(__file__).parent.iterdir() if p.is_file()},
            'official_cpu_cuda_action_trace_mismatches':0,'official_canary_wtl':{'W':64,'T':0,'L':0},'protected_panels':protected,'status':'FROZEN_BEFORE_FINAL_TARGET'}
    (evidence/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    manifest['final_target_status']='OPENED_AFTER_FREEZE'
    (OUT/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    baseline=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],manifest['FINAL_TARGET'],'final_parent')
    improved=run('/tmp/gold_plus_cuda',[candidate],manifest['FINAL_TARGET'],'final_candidate',gate=gate)
    summary=summarize(baseline,improved)
    (evidence/'final_target.json').write_text(json.dumps({'summary':summary,'parent':baseline,'improved':improved},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('canary','final'));parser.add_argument('--evidence',type=Path,default=OUT);args=parser.parse_args()
    canary(args.evidence) if args.mode=='canary' else final(args.evidence)
