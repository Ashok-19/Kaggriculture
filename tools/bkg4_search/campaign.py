from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tools.bkg4_search.cuda_io import load_results, pack_genomes
from tools.bkg4_search.evolvability import EvolvabilityTree, FitnessVector, direct_scalar
from tools.bkg4_search.make_genomes import ROUTES
from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene, save_genome

ROOT = Path(__file__).resolve().parents[2]

# First-divergence grammar. Every feature is BKG4 registry-v1 legal and semantically
# relevant to an opening/lane commitment. No identity/seed/replay/future-state inputs.
MONEY = (250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 15000)
UNITS = (2, 3, 4, 5, 6, 7, 8, 10, 12)
LAND = (1, 2, 3, 4)
MARKET_PRICE = (30, 50, 70, 81, 100, 120, 140, 160)
MARKET_INV = (0, 5, 10, 20, 40, 80)
HIRES = (0, 1, 2, 3, 4, 5, 6, 8)


def random_condition(r: random.Random) -> ConditionGene:
    x = r.random()
    if x < 0.44:
        if r.random() < 0.72:
            return ConditionGene("SHOP_AT", "EQ", 0, r.randrange(8))
        shop = r.randrange(8)
        return ConditionGene("SHOP_SEEN", "GE", shop, 1)
    if x < 0.66:
        f = r.choice(("OWN_MONEY", "OPP_MONEY"))
        return ConditionGene(f, r.choice(("LE", "GE")), 0, r.choice(MONEY))
    if x < 0.78:
        f = r.choice(("OWN_UNITS", "OPP_UNITS"))
        return ConditionGene(f, r.choice(("LE", "GE")), 0, r.choice(UNITS))
    if x < 0.86:
        f = r.choice(("OWN_LAND", "OPP_LAND"))
        return ConditionGene(f, r.choice(("LE", "GE")), 0, r.choice(LAND))
    if x < 0.94:
        item = r.randrange(9)
        return ConditionGene("MARKET_PRICE", r.choice(("LE", "GE")), item, r.choice(MARKET_PRICE))
    item = r.randrange(9)
    return ConditionGene("MARKET_INVENTORY", r.choice(("LE", "GE")), item, r.choice(MARKET_INV))


def perturb_condition(c: ConditionGene, r: random.Random) -> ConditionGene:
    if r.random() < 0.22:
        return random_condition(r)
    if c.feature == "SHOP_AT":
        return ConditionGene(c.feature, c.cmp, c.index, (c.value + r.choice((-1, 1))) % 8)
    if c.feature == "SHOP_SEEN":
        return ConditionGene(c.feature, c.cmp, (c.index + r.choice((-1, 1))) % 8, 1)
    if c.feature in ("OWN_MONEY", "OPP_MONEY"):
        delta = r.choice((-500, -250, -100, 100, 250, 500))
        return ConditionGene(c.feature, c.cmp if r.random() < .8 else ("GE" if c.cmp == "LE" else "LE"), 0, max(0, c.value + delta))
    if c.feature in ("OWN_UNITS", "OPP_UNITS"):
        return ConditionGene(c.feature, c.cmp if r.random() < .8 else ("GE" if c.cmp == "LE" else "LE"), 0, max(1, min(40, c.value + r.choice((-1, 1)))))
    if c.feature in ("OWN_LAND", "OPP_LAND"):
        return ConditionGene(c.feature, c.cmp if r.random() < .8 else ("GE" if c.cmp == "LE" else "LE"), 0, max(1, min(4, c.value + r.choice((-1, 1)))))
    if c.feature == "MARKET_PRICE":
        return ConditionGene(c.feature, c.cmp if r.random() < .85 else ("GE" if c.cmp == "LE" else "LE"), c.index, max(1, c.value + r.choice((-20, -10, 10, 20))))
    if c.feature == "MARKET_INVENTORY":
        return ConditionGene(c.feature, c.cmp if r.random() < .85 else ("GE" if c.cmp == "LE" else "LE"), c.index, max(0, c.value + r.choice((-10, -5, 5, 10))))
    return random_condition(r)


def mutate(g: ModuleGenome, r: random.Random) -> ModuleGenome:
    rules = list(g.rules)
    if not rules:
        return ModuleGenome(g.prefix_step, (RuleGene((random_condition(r),), r.choice(ROUTES)),))
    op = r.random()
    if op < .18 and len(rules) < 8:
        pos = r.randrange(len(rules) + 1)
        rules.insert(pos, RuleGene((random_condition(r),), r.choice(ROUTES)))
    elif op < .27 and len(rules) > 1:
        del rules[r.randrange(len(rules))]
    else:
        i = r.randrange(len(rules)); rule = rules[i]; conds = list(rule.conditions)
        q = r.random()
        if q < .32:
            route = r.choice([x for x in ROUTES if x != rule.route])
            rules[i] = RuleGene(tuple(conds), route)
        elif q < .60:
            j = r.randrange(len(conds)); conds[j] = perturb_condition(conds[j], r); rules[i] = RuleGene(tuple(conds), rule.route)
        elif q < .78 and len(conds) < 4:
            nc = random_condition(r)
            if nc not in conds: conds.append(nc)
            rules[i] = RuleGene(tuple(conds), rule.route)
        elif len(conds) > 1:
            del conds[r.randrange(len(conds))]; rules[i] = RuleGene(tuple(conds), rule.route)
        else:
            conds[0] = perturb_condition(conds[0], r); rules[i] = RuleGene(tuple(conds), rule.route)
    return ModuleGenome(g.prefix_step, tuple(rules))


def shop_seed_layer(prefix: int) -> List[ModuleGenome]:
    out=[]
    for shop in range(8):
        for route in ROUTES:
            out.append(ModuleGenome(prefix,(RuleGene((ConditionGene("SHOP_AT","EQ",0,shop),),route),)))
    return out


def guarded_seed_layer(prefix:int)->List[ModuleGenome]:
    """Causal repair grid for activation-regime brittleness.

    It adds only broad public liquidity guards to PET/PIZZA first-shop decisions; no
    case identity or exact replay state enters the grammar.
    """
    out=shop_seed_layer(prefix)
    for shop in (4,5):
        for route in ROUTES:
            for feature in ("OWN_MONEY","OPP_MONEY"):
                for cmp in ("LE","GE"):
                    for threshold in MONEY:
                        out.append(ModuleGenome(prefix,(RuleGene((ConditionGene("SHOP_AT","EQ",0,shop),ConditionGene(feature,cmp,0,threshold)),route),)))
    return out


def broad_guard_seed_layer(prefix:int)->List[ModuleGenome]:
    """Broad but still domain-restricted public-state guard grid for first-shop lanes.

    The grid is deliberately limited to causal opening-economy quantities available in
    BKG4 registry-v1. It never includes identity, replay/seed, whole-state hashes, worker
    coordinates, or tile fingerprints. Only PET/PIZZA receive guarded variants because
    the first activation campaign established those as the brittle regimes.
    """
    guards:List[ConditionGene]=[]
    for feature,values in (
        ("OWN_MONEY",MONEY),("OPP_MONEY",MONEY),
        ("OWN_UNITS",UNITS),("OPP_UNITS",UNITS),
        ("OWN_LAND",LAND),("OPP_LAND",LAND),
        ("OWN_HIRES_TODAY",HIRES),("OPP_HIRES_TODAY",HIRES),
    ):
        for cmp in ("LE","GE"):
            for threshold in values:guards.append(ConditionGene(feature,cmp,0,threshold))
    for item in range(9):
        for cmp in ("LE","GE"):
            for threshold in MARKET_PRICE:guards.append(ConditionGene("MARKET_PRICE",cmp,item,threshold))
            for threshold in MARKET_INV:guards.append(ConditionGene("MARKET_INVENTORY",cmp,item,threshold))
    out=shop_seed_layer(prefix)
    for shop in (4,5):
        base=ConditionGene("SHOP_AT","EQ",0,shop)
        for route in ROUTES:
            for guard in guards:
                out.append(ModuleGenome(prefix,(RuleGene((base,guard),route),)))
    return out


def evaluate_chunked(evaluator:"CudaEvaluator",genomes:Sequence[ModuleGenome],label:str,chunk:int=512)->List[List[int]]:
    out:List[List[int]]=[]
    for i in range(0,len(genomes),chunk):
        out.extend(evaluator.evaluate(genomes[i:i+chunk],f"{label}_{i//chunk:03d}"))
    return out


class CudaEvaluator:
    def __init__(self, exe: Path, cases: Path, routes: Path, out_dir: Path, gpu: int = 0):
        self.exe=exe;self.cases=cases;self.routes=routes;self.out_dir=out_dir;self.gpu=gpu;self.batch=0;self.games=0;self.seconds=0.0;self.games_s=[]
        out_dir.mkdir(parents=True,exist_ok=True)
    def evaluate(self, genomes: Sequence[ModuleGenome], label: str) -> List[List[int]]:
        if not genomes:return []
        stem=self.out_dir/f"batch_{self.batch:04d}_{label}";self.batch+=1;gf=stem.with_suffix('.b4gm');rf=stem.with_suffix('.b4rs');pack_genomes(genomes,gf)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(self.gpu)
        p=subprocess.run([str(self.exe),'--cases',str(self.cases),'--routes',str(self.routes),'--genomes',str(gf),'--out',str(rf),'--tpb','64'],cwd=ROOT,env=env,text=True,capture_output=True,check=False)
        if p.returncode:raise RuntimeError(p.stderr[-6000:])
        results=load_results(rf);margins=[]
        for row in results:
            ms=[]
            for own,opp,route,flags in row:
                if flags:raise RuntimeError(f'CUDA evaluator flags={flags}')
                ms.append(own-opp)
            margins.append(ms)
        toks={}
        for tok in p.stdout.strip().split():
            if '=' in tok:
                k,v=tok.split('=',1);toks[k]=v
        self.games+=int(toks.get('games',len(genomes)*len(results[0])));self.seconds+=float(toks.get('gpu_ms',0))/1000.0
        if 'games_s' in toks:self.games_s.append(float(toks['games_s']))
        return margins


def case_utility(m: int) -> float:
    outcome=1.0 if m>0 else .5 if m==0 else 0.0
    return outcome*100000.0+max(-20000,min(20000,m))


def screen_accept(parent: Sequence[int], child: Sequence[int], batch: Sequence[int], baseline: Sequence[int]) -> bool:
    # Same minibatch comparison principle as the paper, with an extra hard safety gate.
    if any(baseline[i]>0 and child[i]<=0 for i in batch):return False
    return sum(case_utility(child[i]) for i in batch) > sum(case_utility(parent[i]) for i in batch)


def fitness(margins: Sequence[int], indices: Sequence[int], cases: Sequence[dict], baseline: Sequence[int]) -> FitnessVector:
    idx=list(indices);wins=sum(margins[i]>0 for i in idx);ties=sum(margins[i]==0 for i in idx);losses=len(idx)-wins-ties
    catastrophic=sum((baseline[i]>0 and margins[i]<=0) or margins[i]<baseline[i]-15000 for i in idx)
    protected=sum(bool(cases[i].get('protected')) and margins[i]<baseline[i]-3000 for i in idx)
    own=sum(bool(cases[i].get('own_history')) and margins[i]<baseline[i] for i in idx)
    fam:Dict[str,List[int]]={}
    for i in idx:fam.setdefault(cases[i]['team'],[]).append(i)
    rates=[];losing=0
    for xs in fam.values():
        fw=sum(margins[i]>0 for i in xs);ft=sum(margins[i]==0 for i in xs);fl=len(xs)-fw-ft
        if fl:losing+=1
        rates.append(int(round(1000*(fw+.5*ft)/len(xs))))
    complexity=0
    return FitnessVector(catastrophic,protected,own,losing,losses,ties,wins,min(rates,default=0),float(min((margins[i] for i in idx),default=0)),sum(margins[i] for i in idx)/max(1,len(idx)),0,complexity)


def with_complexity(f:FitnessVector,g:ModuleGenome)->FitnessVector:
    c=len(g.rules)+sum(len(r.conditions) for r in g.rules)
    visits=2+sum(2+len(r.conditions) for r in g.rules)
    return FitnessVector(f.catastrophic_regressions,f.protected_regressions,f.own_history_regressions,f.losing_families,f.losses,f.ties,f.wins,f.worst_family_milli,f.floor_margin,f.mean_margin,visits,c)


def search_indices(cases: Sequence[dict]):
    search=[i for i,c in enumerate(cases) if c.get('search_eligible') and c['team'] not in ('カワシギ','Arman Tuganbaev')]
    dev=[i for i,c in enumerate(cases) if c.get('search_eligible') and c['team'] in ('カワシギ','Arman Tuganbaev')]
    own=[i for i,c in enumerate(cases) if c.get('own_history')]
    return search,dev,own


def run_search(mode: str, budget: int, batch_size: int, seed: int, evaluator: CudaEvaluator, cases: Sequence[dict], baseline: Sequence[int], prefix: int, out_dir: Path, guarded_seeds: bool=False, broad_seeds: bool=False):
    lam=1.0 if mode=='evo' else 0.0;r=random.Random(seed);search_idx,dev_idx,own_idx=search_indices(cases);root_g=ModuleGenome(prefix,());root_f=with_complexity(fitness(baseline,search_idx,cases,baseline),root_g);tree=EvolvabilityTree(root_f,root_f);node_g=[root_g];node_m=[list(baseline)];depth=[0];seen={root_g.semantic_id()};proposals=0;rejected=0
    # Shared pre-search grid: cheaply rank all causal first-shop/route choices, then
    # seed exactly four root children as in the paper. The grid cost is identical in
    # lambda=0 and lambda=1 arms and is reported separately from tree proposals.
    seeds=broad_guard_seed_layer(prefix) if broad_seeds else guarded_seed_layer(prefix) if guarded_seeds else shop_seed_layer(prefix);seed_m=evaluate_chunked(evaluator,seeds,'seed_grid');grid=[]
    for g,m in zip(seeds,seed_m):
        seen.add(g.semantic_id());f=with_complexity(fitness(m,search_idx,cases,baseline),g)
        if screen_accept(baseline,m,search_idx,baseline) and f.catastrophic_regressions==0 and f.protected_regressions==0:
            grid.append((f.lexicographic_key(),g,m,f))
    chosen=[];used_shops=set()
    for _,g,m,f in sorted(grid,key=lambda x:x[0]):
        shop=g.rules[0].conditions[0].value
        if shop in used_shops:continue
        used_shops.add(shop);chosen.append((g,m,f))
        if len(chosen)==4:break
    if not chosen:raise RuntimeError('no safe first-layer seed survived grid')
    for g,m,f in chosen:
        ni=tree.add_accepted(0,f);assert ni==len(node_g);node_g.append(g);node_m.append(m);depth.append(1)
    grid_evals=len(seeds)
    while proposals < budget:
        parents=[];children=[];screen_batches=[];tries=0
        while len(children)<batch_size and proposals+len(children)<budget and tries<batch_size*100:
            tries+=1;p=tree.select(0,.4,lam,1.0,.5);g=mutate(node_g[p],r);sid=g.semantic_id()
            if sid in seen:continue
            seen.add(sid);parents.append(p);children.append(g)
            # Deterministic rotating minibatch of five discovery cases.
            order=list(search_idx);rr=random.Random(seed+proposals+len(children)*1009);rr.shuffle(order);screen_batches.append(order[:min(5,len(order))])
        if not children:break
        cm=evaluator.evaluate(children,mode);proposals+=len(children)
        for p,g,m,bidx in zip(parents,children,cm,screen_batches):
            f=with_complexity(fitness(m,search_idx,cases,baseline),g)
            pf=tree.nodes[p].fitness
            if screen_accept(node_m[p],m,bidx,baseline) and f.catastrophic_regressions<=pf.catastrophic_regressions:
                ni=tree.add_accepted(p,f);assert ni==len(node_g);node_g.append(g);node_m.append(m);depth.append(depth[p]+1)
            else:tree.add_rejected_visit(p);rejected+=1
        if proposals%256<batch_size:
            ck={'mode':mode,'proposals':proposals,'accepted':len(node_g)-1,'rejected':rejected,'best':tree.best_direct(),'nodes':len(node_g)};(out_dir/'checkpoint.json').write_text(json.dumps(ck,sort_keys=True,indent=2)+'\n')
    # Promotion candidate is selected among search-strong nodes by the separate full regression vector.
    ranked=sorted(range(len(node_g)),key=lambda i:tree.nodes[i].fitness.lexicographic_key())[:max(32,min(128,len(node_g)))]
    full_idx=list(range(len(cases)));base_full=with_complexity(fitness(baseline,full_idx,cases,baseline),root_g);candidates=[]
    for i in ranked:
        ff=with_complexity(fitness(node_m[i],full_idx,cases,baseline),node_g[i]);df=with_complexity(fitness(node_m[i],dev_idx,cases,baseline),node_g[i]) if dev_idx else ff
        candidates.append((ff.lexicographic_key(),i,ff,df))
    _,best,full_f,dev_f=min(candidates,key=lambda x:x[0]);n=tree.nodes[best];save_genome(node_g[best],out_dir/'best_genome.json')
    data={'mode':mode,'lambda':lam,'budget':budget,'grid_evals':grid_evals,'proposals':proposals,'accepted':len(node_g)-1,'rejected':rejected,'best_node':best,'best_genome':node_g[best].canonical(),'best_genome_id':node_g[best].semantic_id(),'search_fitness':asdict(n.fitness),'full_fitness':asdict(full_f),'dev_fitness':asdict(dev_f),'v':n.v,'Q':n.q,'sigma':n.brittleness,'visits':n.visits,'value_samples':n.value_samples,'depth':depth[best],'tree_nodes':len(node_g),'gpu_games':evaluator.games,'gpu_seconds':evaluator.seconds,'gpu_games_s_median':statistics.median(evaluator.games_s) if evaluator.games_s else 0,'margins':node_m[best],'baseline':list(baseline)}
    (out_dir/'search_result.json').write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n');return data,node_g[best],node_m[best]


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('evo','greedy'),required=True);ap.add_argument('--budget',type=int,default=1024);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--seed',type=int,default=720004);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--prefix',type=int,default=72);ap.add_argument('--exe',type=Path,default=Path('scratch/bkg4_search/bkg4_snapshot_module_eval'));ap.add_argument('--cases-tsv',type=Path,default=Path('scratch/bkg4_search/cuda_cases16.tsv'));ap.add_argument('--case-json',type=Path,default=Path('scratch/bkg4_search/base0_discovery16_p72/cases.jsonl'));ap.add_argument('--routes',type=Path,default=Path('scratch/kaggriculture_meta_search_bundle/actuators/v6_routes.txt'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--guarded-seeds',action='store_true');ap.add_argument('--broad-seeds',action='store_true');a=ap.parse_args();out=(ROOT/a.out).resolve() if not a.out.is_absolute() else a.out;cases=[json.loads(x) for x in (ROOT/a.case_json).read_text().splitlines() if x.strip()];baseline=[int(round(c['base_margin'])) for c in cases];ev=CudaEvaluator((ROOT/a.exe).resolve(),(ROOT/a.cases_tsv).resolve(),(ROOT/a.routes).resolve(),out,a.gpu);data,_,_=run_search(a.mode,a.budget,a.batch,a.seed,ev,cases,baseline,a.prefix,out,a.guarded_seeds,a.broad_seeds);print(json.dumps(data,ensure_ascii=False,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
