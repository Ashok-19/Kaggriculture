from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from tools.bkg4_search.campaign import CudaEvaluator, fitness, mutate, with_complexity
from tools.bkg4_search.evolvability import direct_scalar
from tools.bkg4_search.module_codec import load_genome

ROOT=Path(__file__).resolve().parents[2]


def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--genome',type=Path,required=True);ap.add_argument('--neighbors',type=int,default=128);ap.add_argument('--seed',type=int,default=99172);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--exe',type=Path,default=Path('scratch/bkg4_search/bkg4_snapshot_module_eval'));ap.add_argument('--cases-tsv',type=Path,default=Path('scratch/bkg4_search/cuda_cases16.tsv'));ap.add_argument('--case-json',type=Path,default=Path('scratch/bkg4_search/base0_discovery16_p72/cases.jsonl'));ap.add_argument('--routes',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle/actuators/v6_routes.txt'));a=ap.parse_args();g=load_genome(a.genome);cases=[json.loads(x) for x in (ROOT/a.case_json).read_text().splitlines() if x.strip()];baseline=[int(round(c['base_margin'])) for c in cases];idx=list(range(len(cases)));base_f=with_complexity(fitness(baseline,idx,cases,baseline),type(g)(g.prefix_step,()));r=random.Random(a.seed);neighbors=[];seen={g.semantic_id()}
 while len(neighbors)<a.neighbors:
  x=mutate(g,r);sid=x.semantic_id()
  if sid in seen:continue
  seen.add(sid);neighbors.append(x)
 out=(ROOT/a.out).resolve() if not a.out.is_absolute() else a.out;ev=CudaEvaluator((ROOT/a.exe).resolve(),(ROOT/a.cases_tsv).resolve(),(ROOT/a.routes).resolve(),out,a.gpu);rows=ev.evaluate([g,*neighbors],'probe');vs=[];fits=[]
 for gg,m in zip([g,*neighbors],rows):
  f=with_complexity(fitness(m,idx,cases,baseline),gg);fits.append(f);vs.append(direct_scalar(f,base_f))
 candidate_f=fits[0];candidate_v=vs[0];nv=vs[1:];nf=fits[1:];safe=[f.catastrophic_regressions==0 and f.protected_regressions==0 and f.own_history_regressions==0 for f in nf];elite=[s and f.wins>=candidate_f.wins-1 for s,f in zip(safe,nf)];promotion=[s and f.wins>=base_f.wins for s,f in zip(safe,nf)];ordered=sorted(nv);q25=ordered[max(0,int(.25*(len(ordered)-1)))] if ordered else 0;med=statistics.median(nv) if nv else 0;q=(candidate_v+sum(nv))/(1+len(nv));res={'genome_id':g.semantic_id(),'neighbors':len(neighbors),'candidate_fitness':asdict(candidate_f),'candidate_v':candidate_v,'neighbor_mean_v':statistics.mean(nv) if nv else 0,'neighbor_median_v':med,'neighbor_q25_v':q25,'safe_fraction':sum(safe)/max(1,len(safe)),'promotion_floor_fraction':sum(promotion)/max(1,len(promotion)),'near_elite_fraction':sum(elite)/max(1,len(elite)),'probe_Q':q,'probe_sigma':candidate_v-q,'gpu_games':ev.games,'gpu_seconds':ev.seconds,'gpu_games_s_median':statistics.median(ev.games_s) if ev.games_s else 0};out.mkdir(parents=True,exist_ok=True);(out/'probe_result.json').write_text(json.dumps(res,sort_keys=True,indent=2)+'\n');print(json.dumps(res,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
