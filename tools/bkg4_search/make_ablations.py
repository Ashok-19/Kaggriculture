from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.bkg4_search.module_codec import ModuleGenome,load_genome,save_genome

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--genome',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();g=load_genome(a.genome);a.out.mkdir(parents=True,exist_ok=True);paths=[]
 for i in range(len(g.rules)):
  x=ModuleGenome(g.prefix_step,tuple(r for j,r in enumerate(g.rules) if j!=i));p=a.out/f'drop_rule_{i}.json';save_genome(x,p);paths.append(str(p))
 # Rule-order swap is a semantic equivalence check when predicates are mutually exclusive.
 if len(g.rules)==2:
  x=ModuleGenome(g.prefix_step,(g.rules[1],g.rules[0]));p=a.out/'swap_rules.json';save_genome(x,p);paths.append(str(p))
 print(json.dumps({'ablations':paths},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
