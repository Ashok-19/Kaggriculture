#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'python'))
import bonsai_kg_v4 as b

def seq(n,leaf=None):return {'type':'Sequence','children':[leaf or {'type':'Success'} for _ in range(n)]}
def cold_policy(target):
 vals=[f'V{i}' for i in range(254)];cases={vals[0]:{'type':'Success'}}
 remaining=target-1-255 # root + 255 switch children baseline
 if remaining<0:
  # small exact tree: active first child is Success, cold second branch carries remaining.
  vals=['HOT','COLD'];base=1+3;extra=target-base
  cases={'HOT':{'type':'Success'},'COLD':seq(max(1,extra)) if extra else {'type':'Success'}}
  # Sequence adds one node plus children, so adjust exact target below.
  p={'schema_version':4,'enums':[{'name':'Mode','values':vals}],'blackboard':[{'name':'mode','type':'Mode','initial':'HOT'}],'tree':{'type':'SwitchEnum','slot':'mode','cases':cases,'default':{'type':'Success'}}}
 else:
  per=remaining//253;rem=remaining%253
  for i in range(1,254):
   extra=per+(1 if i-1<rem else 0)
   cases[vals[i]]=seq(extra) if extra else {'type':'Success'}
  p={'schema_version':4,'enums':[{'name':'Mode','values':vals}],'blackboard':[{'name':'mode','type':'Mode','initial':'V0'}],'tree':{'type':'SwitchEnum','slot':'mode','cases':cases,'default':{'type':'Success'}}}
 cp=b.compile_policy(p)
 if len(cp.nodes)!=target:
  # deterministic exact fallback for small targets: Selector hot first succeeds and trailing cold nodes are legal but incurs same 2-visit runtime.
  children=[{'type':'Success'}]+[{'type':'Success'} for _ in range(target-2)]
  cp=b.compile_policy({'schema_version':4,'tree':{'type':'Selector','children':children}})
 return cp

def hot_policy(visits,total=4096):
 # Active branch is a synchronous Sequence with visits-2 children because root Switch + Sequence are also visited.
 vals=['HOT','COLD'];hotn=max(1,visits-2);hot=seq(hotn);base=1+hotn+1+2 # switch + hot seq+children + cold success + default success
 filler=max(1,total-base-1);cold=seq(filler)
 p={'schema_version':4,'enums':[{'name':'Mode','values':vals}],'blackboard':[{'name':'mode','type':'Mode','initial':'HOT'}],'tree':{'type':'SwitchEnum','slot':'mode','cases':{'HOT':hot,'COLD':cold},'default':{'type':'Success'}}}
 return b.compile_policy(p)

def state_policy(n):
 if n==0:return b.compile_policy({'schema_version':4,'tree':{'type':'Success'}})
 # root Sequence requires a slot because its first WaitTurns may run; total slots = 1 + waits.
 waits=max(1,n-1);p={'schema_version':4,'tree':{'type':'Sequence','children':[{'type':'WaitTurns','turns':720} for _ in range(waits)]}}
 cp=b.compile_policy(p);assert cp.state_count==n;return cp

def intent_policy(n):
 if n==0:return b.compile_policy({'schema_version':4,'tree':{'type':'Success'}})
 p={'schema_version':4,'tree':{'type':'Sequence','children':[{'type':'EmitMarketIntent','action':'HIRE'} for _ in range(n)]}}
 cp=b.compile_policy(p);assert cp.max_live_intents==n;return cp

def depth_policy(n):
 x={'type':'Success'}
 for _ in range(n-1):x={'type':'Invert','child':x}
 cp=b.compile_policy({'schema_version':4,'tree':x});assert cp.max_depth==n;return cp

def save(cp,path):b.save_compiled(cp,path);return {'file':str(path),'nodes':len(cp.nodes),'state':cp.state_count,'depth':cp.max_depth,'live':cp.max_live_intents,'max_visits':cp.analysis[0].max_visits}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='scratch/bonsai_kg/v4_bench_policies');a=ap.parse_args();o=Path(a.out);o=o if o.is_absolute() else ROOT/o;o.mkdir(parents=True,exist_ok=True);rows=[]
 for n in (128,1024,4096,16384,32768,65534):rows.append(('cold',n,save(cold_policy(n),o/f'cold_{n}.bkgc')))
 for n in (16,32,64,128,256,512):rows.append(('hot',n,save(hot_policy(n),o/f'hot_{n}.bkgc')))
 for n in (32,64,128,256,512):rows.append(('state',n,save(state_policy(n),o/f'state_{n}.bkgc')))
 for n in (0,8,16,32,64,128):rows.append(('intent',n,save(intent_policy(n),o/f'intent_{n}.bkgc')))
 for n in (16,24,32,48):rows.append(('depth',n,save(depth_policy(n),o/f'depth_{n}.bkgc')))
 tiny={'schema_version':4,'tree':{'type':'ReactiveSelector','children':[{'type':'Sequence','children':[{'type':'Condition','lhs':{'feature':'STEP'},'op':'GE','rhs':1000000000},{'type':'Success'}]},{'type':'MoveUnitTo','unit':0,'target':{'const':0,'type':'TILE'},'priority':100}]}}
 rows.append(('comparison',5,save(b.compile_policy(tiny),o/'tiny_equiv.bkgc')))
 for r in rows:print(r)
 return 0
if __name__=='__main__':raise SystemExit(main())
