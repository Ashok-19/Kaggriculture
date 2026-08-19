#!/usr/bin/env python3
from __future__ import annotations
import argparse,random,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'python'))
import bonsai_kg as b3
import bonsai_kg_v4 as b4

def make_obs(r,step):
 d={(b3.FEATURES['STEP'],0):step,(b3.FEATURES['DAY'],0):step//24,(b3.FEATURES['HOUR'],0):step%24,(b3.FEATURES['OWN_MONEY'],0):r.randrange(0,3000),(b3.FEATURES['OPP_MONEY'],0):r.randrange(0,3000),(b3.FEATURES['OWN_UNITS'],0):r.randrange(1,12),(b3.FEATURES['OPP_UNITS'],0):r.randrange(1,12),(b3.FEATURES['OWN_LAND'],0):r.randrange(1,5),(b3.FEATURES['OPP_LAND'],0):r.randrange(1,5),(b3.FEATURES['OWN_HIRES_TODAY'],0):r.randrange(0,8),(b3.FEATURES['OPP_HIRES_TODAY'],0):r.randrange(0,8),(b3.FEATURES['SHOP_COUNT'],0):r.randrange(0,5)}
 for i in range(9):d[(b3.FEATURES['MARKET_PRICE'],i)]=r.randrange(20,180);d[(b3.FEATURES['MARKET_INVENTORY'],i)]=r.randrange(0,80)
 for i in range(8):d[(b3.FEATURES['SHOP_AT'],i)]=r.randrange(-1,8);d[(b3.FEATURES['SHOP_SEEN'],i)]=r.randrange(0,3)
 return b3.SyntheticObservation(d)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--controller',default='bonsai-kg/integrations/stateful_v6/assets/controller.bkgc');ap.add_argument('--ticks',type=int,default=512);ap.add_argument('--seed',type=int,default=40404);a=ap.parse_args();controller=Path(a.controller);controller=controller if controller.is_absolute() else HERE.parents[1]/controller;p3=b3.load_compiled(controller);p4=b4.compile_bkg3_to_v4(p3);v3=b3.VM(p3);v4=b4.VM(p4);r=random.Random(a.seed);bad=[]
 for t in range(a.ticks):
  o=make_obs(r,t%720);x=v3.tick(o);y=v4.tick(o)
  key3=(x.status,x.fault,x.action.compact(),x.blackboard);key4=(y.status,y.fault,y.action.compact(),y.blackboard)
  if key3!=key4:bad.append((t,key3,key4));break
 print({'ticks':a.ticks,'mismatches':len(bad),'bkg3_nodes':len(p3.nodes),'bkg3_state':p3.state_count,'bkg4_nodes':len(p4.nodes),'bkg4_state':p4.state_count,'state_slots_elided':p3.state_count-p4.state_count,'bkg4_max_visits':p4.analysis[0].max_visits})
 if bad:print('MISMATCH',bad[0])
 return 1 if bad else 0
if __name__=='__main__':raise SystemExit(main())
