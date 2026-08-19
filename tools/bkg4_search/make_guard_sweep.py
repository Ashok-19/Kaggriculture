from __future__ import annotations
import argparse,json
from pathlib import Path
from tools.bkg4_search.cuda_io import pack_genomes
from tools.bkg4_search.module_codec import ConditionGene,ModuleGenome,RuleGene

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',type=int,default=72);ap.add_argument('--shop',type=int,default=4);ap.add_argument('--route',type=int,default=29);ap.add_argument('--feature',default='OPP_MONEY');ap.add_argument('--cmp',default='GE');ap.add_argument('--thresholds',type=int,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();g=[ModuleGenome(a.prefix,())]
 for t in a.thresholds:g.append(ModuleGenome(a.prefix,(RuleGene((ConditionGene('SHOP_AT','EQ',0,a.shop),ConditionGene(a.feature,a.cmp,0,t)),a.route),)))
 pack_genomes(g,a.out);print(json.dumps({'genomes':len(g),'thresholds':a.thresholds},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
