"""Close the bounded unsuccessful search without opening FINAL_TARGET."""
import json
import subprocess
from .benchmark import OUT
from .search import run
from .screen import summarize
from kaggriculture_bonsai.agents.gold_v25.extract import ROOT,sha


def main():
    identity=json.loads((OUT/'round2/composition.json').read_text())
    candidate=identity['candidate'];gate=OUT/'round2/specialist.bkgc'
    evidence={};all_parent=[];all_improved=[]
    for label,path,key in (
        ('BUILD_v1',OUT/'parent_build.json','cases'),
        ('DEV_v1',OUT/'internal_dev.json','parent'),
        ('BUILD_v2',OUT/'round2/build.json','parent'),
        ('DEV_v2',OUT/'round2/internal_dev.json','parent'),
        ('BUILD_v3',OUT/'round3/diagnostic.json','parent'),
    ):
        parent=json.loads(path.read_text())[key]
        seeds=list(dict.fromkeys(r['seed'] for r in parent))
        result=run('/tmp/gold_plus_cuda',[candidate],seeds,'audit_'+label,gate=gate)
        evidence[label]=summarize(parent,result)
        all_parent.extend(parent);all_improved.extend(result)
    probes=[candidate,{'id':300,'start':0,'end':718,'mode':33},
            {'id':301,'start':0,'end':718,'mode':65},
            {'id':200,'start':0,'end':718,'mode':17}]
    cpu=run('/tmp/gold_plus_cpu',probes,[990042,990057],'audit_cpu')
    cuda=run('/tmp/gold_plus_cuda',probes,[990042,990057],'audit_cuda')
    assert cpu==cuda
    protected_paths=['src/kaggriculture_bonsai/agents/gold_v25','assets/champion','assets/baseline',
                     'workspace/gold_v25_port','workspace/gold_v25_submission']
    subprocess.run(['git','diff','--exit-code','47b304c','--',*protected_paths],cwd=ROOT,check=True)
    assert json.loads((OUT/'panels.json').read_text())['final_target_status']=='UNOPENED'
    summary={'status':'NOT_COMPLETE_NO_PROMOTION','starting_head':'47b304c8954216b99705df92c510612d7068f6eb',
        'best_frozen_candidate':identity,'panels':evidence,'combined':summarize(all_parent,all_improved),
        'cpu_cuda_probe_games':len(cpu),'cpu_cuda_mismatches':0,
        'tests':{'parent_focused_groups':3,'v4_semantic_tests':40,'evaluator_tests_including_cuda':19,'plus_focused_groups':2},
        'FINAL_TARGET_games':0,'FINAL_TARGET_status':'UNOPENED',
        'official_canary_v1':json.loads((OUT/'v1_rejection.json').read_text()),
        'official_canary_v2_games':0,'original_parent_champion_baselines_unchanged':True,
        'controller_nodes_total':23+identity['gate_nodes'],'controller_blackboard_total':3+identity['gate_bb'],
        'controller_node_state_slots':0,'route_pack_changes':0,'generic_runtime_changes':0,
        'stop_reason':'The best frozen composition still loses. Two distinct additional market representations yielded no better candidate; no proven production-preserving weed repair was found. No extra downstream evaluation or promotion is justified.',
        'source_sha256':{p.name:sha(p.read_bytes()) for p in (ROOT/'src/kaggriculture_bonsai/agents/gold_v25_plus').iterdir() if p.is_file()}}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','panels')},indent=2))


if __name__=='__main__':main()
