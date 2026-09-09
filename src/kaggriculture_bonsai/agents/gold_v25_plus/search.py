"""Small deterministic endgame continuation question, evaluated in CUDA batches."""
import json
import subprocess
from pathlib import Path
from .benchmark import OUT, panels
from kaggriculture_bonsai.agents.gold_v25.extract import ASSETS


def run(binary, candidates, seeds, label, gate=False):
    directory = Path('/tmp/gold_v25_plus')
    directory.mkdir(exist_ok=True)
    cp, sp = directory / (label + '.candidates'), directory / (label + '.seeds')
    cp.write_text(''.join(f"{c['id']} {c['start']} {c['end']} {c['mode']}\n" for c in candidates))
    sp.write_text(''.join(f'{s}\n' for s in seeds))
    command = [binary, str(ASSETS/'controller.bkgc'), str(ASSETS/'routes.bkv6'), str(ASSETS/'arity.bin'), str(cp), str(sp)]
    if gate:
        command.append(str(OUT/'specialist.bkgc') if gate is True else str(gate))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        cid, seed, seat, m0, m1, trace, fault, changes = line.split()
        money, seat = [float(m0), float(m1)], int(seat)
        margin = money[seat] - money[1-seat]
        rows.append({'candidate': int(cid), 'seed': int(seed), 'seat': seat, 'money': money, 'margin': margin,
                     'outcome': 'W' if margin > 0 else 'L' if margin < 0 else 'T', 'trace': trace, 'fault': int(fault), 'changes': int(changes)})
    assert not any(r['fault'] for r in rows)
    return rows


def main():
    seeds = panels()['BUILD']
    parent = [{'id': 0, 'start': 0, 'end': 0, 'mode': 0}]
    gpu_parent = run('/tmp/gold_plus_cuda', parent, seeds, 'parent')
    cpu_parent = run('/tmp/gold_plus_cpu', parent, seeds[:2], 'parity')
    assert cpu_parent == gpu_parent[:4], (cpu_parent, gpu_parent[:4])
    old = json.loads((OUT/'parent_build.json').read_text())['cases']
    assert [r['money'] for r in gpu_parent] == [r['rewards'] for r in old]
    candidates = [{'id': i+1, 'start': start, 'end': 718, 'mode': mode}
                  for i, (start, mode) in enumerate((s,m) for s in (696,704,708,712,714,716,717,718) for m in (1,2,3,5,6,7))]
    (OUT/'candidate_question.json').write_text(json.dumps({'question': 'Can bounded late cargo consolidation / market priority turn symmetric outcomes into wins?', 'candidates': candidates}, indent=2)+'\n')
    screen = run('/tmp/gold_plus_cuda', candidates, seeds[:4], 'screen')
    ranking = []
    for c in candidates:
        rr = [r for r in screen if r['candidate'] == c['id']]
        ranking.append((sum(r['outcome']=='W' for r in rr), min(r['margin'] for r in rr), c))
    ranking.sort(key=lambda x:(-x[0],-x[1],x[2]['id']))
    survivors = [r[2] for r in ranking[:6]]
    build = run('/tmp/gold_plus_cuda', survivors, seeds, 'build')
    summary = []
    for c in survivors:
        rr = [r for r in build if r['candidate']==c['id']]
        transitions = {a+b:sum(p['outcome']==a and r['outcome']==b for p,r in zip(gpu_parent,rr)) for a,b in (('L','W'),('T','W'),('W','T'),('W','L'),('T','L'))}
        summary.append({'candidate':c,'wtl':{o:sum(r['outcome']==o for r in rr) for o in 'WTL'}, 'min_margin':min(r['margin'] for r in rr),'transitions':transitions})
    result={'parent_cpu_cuda_mismatches':0,'parent_python_native_reward_mismatches':0,'screen':screen,'build':build,'summary':summary}
    (OUT/'search.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
