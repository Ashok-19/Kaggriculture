from __future__ import annotations

import argparse
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"python"))
import bonsai_kg as bkg
from kaggle_environments import make

MASK=(1<<64)-1

def mix(h,x):return ((h^(x&0xffffffff))*1099511628211)&MASK

def feature_hash(obs):
    h=1469598103934665603
    def add(fid,arg=0):
        nonlocal h
        h=mix(h,fid);h=mix(h,arg);h=mix(h,obs.feature(fid,arg))
    plain=[bkg.FEATURES[x] for x in ("ALWAYS","STEP","DAY","HOUR","OWN_MONEY","OPP_MONEY","OWN_UNITS","OPP_UNITS","OWN_LAND","OPP_LAND","OWN_HIRES_TODAY","OWN_SHED_TOTAL","SHOP_COUNT")]
    for fid in plain:add(fid)
    for i in range(9):add(bkg.FEATURES["MARKET_PRICE"],i);add(bkg.FEATURES["MARKET_INVENTORY"],i)
    for i in range(12):add(bkg.FEATURES["OWN_SHED"],i)
    for i in range(5):add(bkg.FEATURES["OWN_SEED"],i)
    for u in range(40):
        for item in range(12):add(bkg.FEATURES["OWN_INVENTORY"],(u<<8)|item)
    for i in range(8):add(bkg.FEATURES["SHOP_AT"],i);add(bkg.FEATURES["SHOP_SEEN"],i)
    for u in range(40):
        for name in ("OWN_WORKER_X","OWN_WORKER_Y","OPP_WORKER_X","OPP_WORKER_Y"):add(bkg.FEATURES[name],u)
    for fid in range(bkg.FEATURES["OWN_TILE_KIND"],bkg.FEATURES["OPP_TILE_FERT_AVAILABLE"]+1):
        for tile in range(100):add(fid,tile)
    return h

def scripted(step):
    out={"farmer":["PASS"],"market":[]}
    if step==0:out["market"]=[["BUY_SEED","WHEAT",3],["BUY_PRODUCT","WHEAT",2],["HIRE"]]
    elif step==1:out["farmer"]=["PICKUP","WHEAT",1]
    elif step==2:out["farmer"]=["PLANT","WHEAT"]
    elif step==3:out["farmer"]=["WATER"]
    elif step==4:out["market"]=[["BUY_ANIMAL","GOOSE",1]]
    elif step==5:out["farmer"]=["PICKUP","GOOSE",1]
    elif step==6:out["farmer"]=["WEST"]
    elif step==7:out["farmer"]=["BUILD_COOP"]
    elif step==8:out["farmer"]=["PLACE","GOOSE",1]
    elif step==9:out["farmer"]=["CARE"]
    return out

def main(args):
    t=0
    def agent(raw):
        nonlocal t
        o=bkg.DictObservation(raw);print(f"T={t} H={feature_hash(o)}");a=scripted(int(raw.get("step",0) or 0));t+=1;return a
    env=make("kaggriculture",configuration={"episodeSteps":args.steps,"seed":args.seed},debug=True);env.run([agent,"pass"]);return 0
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seed",type=int,default=12345);ap.add_argument("--steps",type=int,default=80);raise SystemExit(main(ap.parse_args()))
