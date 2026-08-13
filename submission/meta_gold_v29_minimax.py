import copy, importlib.util, pathlib
HERE=pathlib.Path(__file__).resolve().parent
# Start from the proven v23 candidate (blocked YARN rule, H7, global late-start 528).
spec=importlib.util.spec_from_file_location('meta_gold_v29_base', HERE/'meta_gold_candidate.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
v23=__import__('meta_gold_v23_latestart'); v10=v23.base; v8=v10.base; v1=v8.base
_ORIG_ROUTE=v1._route; _ORIG_USE=v8._use_secondary
_YMIX=copy.deepcopy(v1.CURRENT)
_YMIX_STEPS=[120, 121, 129, 192, 193, 198, 204]
for _s in _YMIX_STEPS:
    _a=_YMIX[_s]
    def _sw(x): return ['SHEEP' if y=='COW' else y for y in x] if isinstance(x,list) else x
    _a['farmer']=_sw(_a.get('farmer'))
    _a['hands']=[_sw(x) for x in (_a.get('hands') or [])]
    _a['market']=[_sw(x) for x in (_a.get('market') or [])]
_PILK={0:False,1:False}
_NONCORE={0:False,1:False}
_COREHIGH={0:False,1:False}
_FERT241={0:0,1:0}
_FERT_ELITE={0:False,1:False}
_R2H10={0:False,1:False}

_MELON_FAST=copy.deepcopy(v1.CURRENT)
for _s,_cmd in {249:['EAST'],250:['SOUTH'],251:['SOUTH'],252:['SOUTH'],253:['DROP'],254:['WEST'],255:['CARE'],256:['COLLECT_FERTILIZER'],257:['NORTH'],258:['NORTH'],259:['NORTH'],260:['NORTH'],261:['WEST'],262:['WATER'],263:['PASS']}.items():
    if len(_MELON_FAST[_s].get('hands') or [])>9: _MELON_FAST[_s]['hands'][9]=_cmd
for _s,_cmd in {250:['EAST'],251:['EAST'],252:['SOUTH'],253:['SOUTH'],254:['SOUTH'],255:['DROP'],256:['WEST'],257:['WEST'],258:['WEST'],259:['WEST'],260:['NORTH'],261:['NORTH'],262:['WATER'],263:['PASS']}.items():
    if len(_MELON_FAST[_s].get('hands') or [])>13: _MELON_FAST[_s]['hands'][13]=_cmd
for _s,_cmd in {253:['EAST'],254:['EAST'],255:['EAST'],256:['EAST'],257:['SOUTH'],258:['SOUTH'],259:['SOUTH'],260:['DROP'],261:['PASS'],262:['PASS'],263:['PASS']}.items():
    if len(_MELON_FAST[_s].get('hands') or [])>10: _MELON_FAST[_s]['hands'][10]=_cmd
for _s,_cmd in {249:['SOUTH'],250:['PLANT','MELON'],251:['WATER'],252:['WEST'],253:['PLANT','MELON'],254:['WATER'],255:['WEST'],256:['WEST'],257:['PLANT','MELON'],258:['WATER'],259:['NORTH'],260:['WATER'],261:['EAST'],262:['WATER'],263:['PASS']}.items():
    if len(_MELON_FAST[_s].get('hands') or [])>11: _MELON_FAST[_s]['hands'][11]=_cmd
_MELON_FAST[253]['market']=[['SELL','MELON',6]]+list(_MELON_FAST[253].get('market') or [])
_MELON_FAST[257]['market']=[list(_o) for _o in (_MELON_FAST[257].get('market') or []) if not (len(_o)>=3 and _o[0]=='SELL' and _o[1]=='MELON' and int(_o[2])==6)]
_m260=[]
for _o in (_MELON_FAST[260].get('market') or []):
    _o=list(_o)
    if len(_o)>=3 and _o[0]=='SELL' and _o[1]=='MELON' and int(_o[2])==6: _o[2]=12
    _m260.append(_o)
_MELON_FAST[260]['market']=_m260
_MELON_FAST[262]['market']=[list(_o) for _o in (_MELON_FAST[262].get('market') or []) if not (len(_o)>=3 and _o[0]=='SELL' and _o[1]=='MELON' and int(_o[2])==6)]
_MELON_FAST_START=249

def _route(obs,branch):
    step=int(obs.get('step',0) or 0)
    shops=list((obs.get('town',{}) or {}).get('unlocked_shops',[]) or [])
    seat=int(obs.get('player',0) or 0)
    if _NONCORE.get(seat,False) and shops and shops[0]=='YARN_STORE' and _YMIX_STEPS and step>=min(_YMIX_STEPS):
        return _YMIX
    # High-cash core openings can be recognized at step 13. If their first
    # revealed shop is YARN, switch only after the first structural divergence.
    if _COREHIGH.get(seat,False) and shops and shops[0]=='YARN_STORE' and step>=144:
        if step>=192: return v1.YARN
        return v1.CORE
    # Core-family YARN+PIZZA is causally known once shop #2 appears at step 144.
    # CURRENT and CORE share the route through step 143, so this late switch is
    # structurally compatible even when the early Kaito money signature shifted.
    if (not _NONCORE.get(seat,False)) and len(shops)>=2 and shops[0]=='YARN_STORE' and shops[1]=='PIZZA_SHOP' and step>=144:
        if step>=192: return v1.YARN
        return v1.CORE
    if branch=='kaito':
        if len(shops)>=2 and shops[0]=='YARN_STORE' and shops[1]=='PIZZA_SHOP':
            if step>=192:return v1.YARN
            if step>=14:return v1.CORE
    if branch!='kaito' and step>=_MELON_FAST_START and _R2H10.get(seat,False):
        return _MELON_FAST
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
    if step==0:
        _PILK[seat]=False; _NONCORE[seat]=False; _COREHIGH[seat]=False; _R2H10[seat]=False; _FERT241[seat]=0; _FERT_ELITE[seat]=False
    if step==13:
        _PILK[seat]=_pilk_like(obs)
        _NONCORE[seat]=not v1._core_opening_like(obs)
        f=(obs.get('farms') or [])[1-seat]
        _COREHIGH[seat]=(not _NONCORE[seat]) and float(f.get('money',0) or 0)>=7.0
        _FERT_ELITE[seat]=len(f.get('hands',[]) or [])==5
    if step==240:
        _R2H10[seat]=True
    a=_orig(obs,configuration)
    if _FERT_ELITE.get(seat,False) and v1._BRANCH.get(seat,'global')!='kaito':
        # Rolling debt: any fertilizer sold early is repaid by suppressing the
        # next scheduled fertilizer sales in chronological order.
        rem=max(0,int(_FERT241.get(seat,0))); out=[]
        for raw in (a.get('market') or []):
            o=list(raw)
            if rem and len(o)>=3 and o[0]=='SELL' and o[1]=='FERTILIZER':
                cut=min(rem,max(0,int(o[2]))); o[2]-=cut; rem-=cut
                if o[2]<=0: continue
            out.append(o)
        if rem!=int(_FERT241.get(seat,0)):
            a=copy.deepcopy(a); a['market']=out
        _FERT241[seat]=rem
        _windows={241:(12,281),278:(9,305),302:(12,329),326:(8,353),350:(7,377),406:(7,425),521:(13,545),601:(2,617)}
        if step in _windows:
            cap,horizon=_windows[step]
            shed=((obs.get('private',{}) or {}).get('shed',{}) or {})
            stock=max(0,int(shed.get('FERTILIZER',0) or 0))
            planned=sum(max(0,int(o[2])) for o in (a.get('market') or []) if len(o)>=3 and o[0]=='SELL' and o[1]=='FERTILIZER')
            reserve=0
            route=v1._route(obs,v1._BRANCH.get(seat,'global'))
            for s in range(step+1,min(horizon,len(route))):
                ra=route[s]
                for cmd in [ra.get('farmer') or ['PASS'],*((ra.get('hands') or []))]:
                    if isinstance(cmd,list) and len(cmd)>=3 and cmd[0]=='PICKUP' and cmd[1]=='FERTILIZER':
                        reserve+=max(0,int(cmd[2]))
            q=min(cap,max(0,stock-planned-reserve))
            if q>0 and len(a.get('market') or [])<10:
                a=copy.deepcopy(a); a['market']=list(a.get('market') or [])+[['SELL','FERTILIZER',q]]; _FERT241[seat]+=q
    return a
