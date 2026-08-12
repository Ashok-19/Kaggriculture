import importlib.util, pathlib
HERE=pathlib.Path(__file__).resolve().parent
# Start from the proven v23 candidate (blocked YARN rule, H7, global late-start 528).
spec=importlib.util.spec_from_file_location('meta_gold_v29_base', HERE/'meta_gold_candidate.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
v23=__import__('meta_gold_v23_latestart'); v10=v23.base; v8=v10.base; v1=v8.base
_ORIG_ROUTE=v1._route; _ORIG_USE=v8._use_secondary
_PILK={0:False,1:False}

def _route(obs,branch):
    step=int(obs.get('step',0) or 0)
    if branch=='kaito':
        shops=list((obs.get('town',{}) or {}).get('unlocked_shops',[]) or [])
        if len(shops)>=2 and shops[0]=='YARN_STORE' and shops[1]=='PIZZA_SHOP':
            if step>=192:return v1.YARN
            if step>=14:return v1.CORE
    return _ORIG_ROUTE(obs,branch)
v1._route=_route

def _melon_count(f):
    return sum(1 for row in (f.get('tiles',[]) or []) for t in row if isinstance(t,dict) and t.get('kind')=='PLANT' and t.get('crop')=='MELON')
def _pilk_like(obs):
    seat=int(obs.get('player',0) or 0);f=(obs.get('farms') or [])[1-seat]
    return len(f.get('hands',[]) or [])==8 and float(f.get('money',0) or 0)==96.0 and _melon_count(f)==10

def _use_secondary(obs,seat):
    if _PILK.get(int(seat),False) and int(obs.get('step',0) or 0)>=432:return True
    return _ORIG_USE(obs,seat)
v8._use_secondary=_use_secondary
_orig=base.agent
def agent(obs,configuration=None):
    step=int(obs.get('step',0) or 0);seat=int(obs.get('player',0) or 0)
    if step==0:_PILK[seat]=False
    if step==13:_PILK[seat]=_pilk_like(obs)
    return _orig(obs,configuration)
