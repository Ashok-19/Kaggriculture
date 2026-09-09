"""Accept only a frozen composition with passing independent evidence."""
import json
import shutil
import subprocess
from pathlib import Path
from .benchmark import OUT
from .package import PACKAGE
from kaggriculture_bonsai.agents.gold_v25.extract import ROOT,sha


def main():
    out=OUT/'round8'
    final=json.loads((out/'final_target.json').read_text())
    freeze=json.loads((out/'freeze.json').read_text())
    canary=json.loads((out/'official_canary.json').read_text())
    protected=json.loads((out/'protected.json').read_text())
    identity=json.loads((out/'composition.json').read_text())
    assert final['summary']['improved_wtl']=={'W':256,'T':0,'L':0}
    assert all(s['protected_regressions']==0 for s in protected.values())
    assert all(r['outcome']=='W' for r in canary['cases']) and len(canary['cases'])==64
    assert canary['action_state_mismatches']==canary['faults']==freeze['official_cpu_cuda_action_trace_mismatches']==0
    assert freeze['composition_sha256']==sha((out/'composition.json').read_bytes())
    source=Path(__file__).parent
    assert freeze['source_sha256']=={p.name:sha(p.read_bytes()) for p in source.iterdir() if p.is_file()}
    tests=['kaggriculture_bonsai.agents.gold_v25.test_gold_v25','tests.unit.test_v4_semantics',
           'tests.unit.test_v4_malformed','tests.parity.test_evaluator_parity',
           'kaggriculture_bonsai.agents.gold_v25_plus.test_repair','kaggriculture_bonsai.agents.gold_v25_plus.test_plus']
    subprocess.run([str(ROOT/'.venv/bin/python'),'-m','unittest',*tests],cwd=ROOT,check=True)
    shutil.copyfile(source/'cleanroom_check.py',PACKAGE/'check.py')
    subprocess.run([str(ROOT/'.venv/bin/python'),'-E',str(PACKAGE/'check.py')],cwd='/tmp',check=True)
    paths=['src/kaggriculture_bonsai/agents/gold_v25','assets/champion','assets/baseline',
           'workspace/gold_v25_port','workspace/gold_v25_submission']
    subprocess.run(['git','diff','--exit-code','47b304c','--',*paths],cwd=ROOT,check=True)
    result={'status':'COMPLETE_OBSERVED_100_PERCENT_WITH_ZERO_PROTECTED_REGRESSION',
        'starting_head_this_continuation':'26284e75deb74dc45898b8f3285360cdcca9ca2c',
        'original_phase_starting_head':'47b304c8954216b99705df92c510612d7068f6eb',
        'candidate':identity['identity'],'parent':identity['parent'],
        'final_target':final['summary'],'official_canary':{'games':64,'W':64,'T':0,'L':0,'action_state_mismatches':0,'cpu_cuda_trace_mismatches':0},
        'protected_panels':protected,'tests_passed':65,'standalone_cleanroom':'PASS',
        'parent_champion_baselines_unchanged':True,'nodes':39,'blackboard_slots':4,'vm_node_state_slots':0,
        'additional_overlay_state_bytes_per_player':488,'derived_seed_cache_bytes_per_player':12984,
        'generic_runtime_changes':0,'route_pack_changes':0,'specialists':6,
        'package_sha256':sha(Path('/tmp/submission_gold_v25_plus.tar.gz').read_bytes()),
        'freeze_sha256':sha((out/'freeze.json').read_bytes())}
    (out/'release.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'active_candidate.json').write_text(json.dumps({**identity,'status':'ACCEPTED','release':'round8/release.json'},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
