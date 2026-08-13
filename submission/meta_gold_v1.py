import collections, copy, json, pathlib, sys
HERE=pathlib.Path(__file__).resolve().parent
import runtime_gateway as rg

CURRENT=copy.deepcopy(rg.ROUTES['lev'][:174]+rg.ROUTES['v27'][174:])
CORE=copy.deepcopy(json.load(open(HERE/'kaito44_majority_route.json')))
YARN=copy.deepcopy(json.load(open(HERE/'kaito_yarn5_route.json')))
# q11 universal opening
m=[list(o) for o in CURRENT[0].get('market',[])]
for o in m:
    if len(o)>=3 and o[0]=='BUY_PRODUCT' and o[1]=='WHEAT': o[2]=8
w=next(o for o in m if len(o)>=3 and o[0]=='BUY_PRODUCT' and o[1]=='WHEAT')
others=[o for o in m if o is not w]
hi=[o for o in others if o[0]=='HIRE']; an=[o for o in others if o[0]=='BUY_ANIMAL']; mel=[o for o in others if len(o)>=2 and o[0]=='BUY_SEED' and o[1]=='MELON']; ws=[o for o in others if len(o)>=2 and o[0]=='BUY_SEED' and o[1]=='WHEAT']; rest=[o for o in others if o not in hi+an+mel+ws]
CURRENT[0]['market']=[w]+hi+an+mel+ws+rest

_BRANCH={0:'global',1:'global'}
_DEBT={0:{},1:{}}
_RESCUE={0:None,1:None}
_ORIG_WEED=rg._weed_repair
LOW={'FEED','COLLECT_FERTILIZER'}
PREEMPT=('FERTILIZER','WOOL','MILK','STRAWBERRY','MELON')

# Extend dependency replay to MELON and hard-reset stale transactions at midnight.
def _weed_repair(obs,action,route,step,state):
    action=rg._align(rg._copy_action(action),obs)
    farm=rg._farm(obs,int(rg._get(obs,'player',0) or 0))
    positions=[rg._get(farm,'farmer'),*list(rg._get(farm,'hands',[]) or [])]
    unit=[action.get('farmer',['PASS']),*list(action.get('hands') or [])]
    active=state['weed']
    if step%24==0: active.clear()
    for actor,tx in list(active.items()):
        idx=0 if actor=='farmer' else int(actor)+1
        if idx>=len(unit): active.pop(actor,None); continue
        age=step-tx['start']; intended=tx['intended']
        dep=(bool(intended and intended[0]=='BUILD_PASTURE') or bool(len(intended)>=2 and intended[0]=='PLANT' and intended[1] in ('WHEAT','STRAWBERRY','MELON')))
        if age==1: unit[idx]=list(intended)
        elif dep and age>=2:
            unit[idx]=rg._trace_actor_action(route,step-1,actor)
            if rg._trace_actor_action(route,step,actor)[0]=='PASS' or age>24: active.pop(actor,None)
        elif 2<=age<=1+rg.WEED_REPLAY_STEPS: unit[idx]=rg._trace_actor_action(route,step-1,actor)
        else: active.pop(actor,None)
    for idx,(pos,intended) in enumerate(zip(positions,unit)):
        actor='farmer' if idx==0 else idx-1
        if actor in active or not isinstance(intended,list) or not intended or intended[0] not in ('BUILD_PASTURE','PLANT'): continue
        tile=rg._tile_at(farm,pos)
        if isinstance(tile,dict) and tile.get('kind')=='WEED': active[actor]={'start':step,'intended':list(intended)}; unit[idx]=['DIG']
    action['farmer']=unit[0] if unit else ['PASS']; action['hands']=unit[1:]
    return rg._align(action,obs)
rg._weed_repair=_weed_repair

def _counts(f):
    c={'WHEAT':0,'MELON':0,'PASTURE':0,'COW':0,'SHEEP':0}
    for row in f.get('tiles',[]):
        for t in row:
            if not isinstance(t,dict): continue
            if t.get('kind')=='PLANT' and t.get('crop') in ('WHEAT','MELON'): c[t['crop']]+=1
            elif t.get('kind')=='PASTURE':
                c['PASTURE']+=1; a=t.get('animal');
                if a in ('COW','SHEEP'): c[a]+=1
    return c

def _kaito_like(obs):
    seat=int(obs.get('player',0)); f=obs.get('farms',[])[1-seat]; c=_counts(f)
    return len(f.get('hands',[]))==4 and tuple(f.get('farmer',[]))==(3,4) and float(f.get('money',0))==10.0 and (c['WHEAT'],c['MELON'],c['PASTURE'],c['COW'],c['SHEEP'])==(3,2,3,1,2)

def _core_opening_like(obs):
    # Coarse public production-architecture family, not identity: this structure
    # is shared by multiple v27/Lev descendants and historical opponents.
    seat=int(obs.get('player',0)); f=obs.get('farms',[])[1-seat]; c=_counts(f)
    return (c['WHEAT'],c['MELON'],c['PASTURE'],c['COW'],c['SHEEP'])==(3,2,3,1,2)

def _route(obs,branch):
    step=int(obs.get('step',0))
    if branch=='kaito' and step>=14:
        shops=list((obs.get('town',{}) or {}).get('unlocked_shops',[]) or [])
        if step>=192 and shops and shops[0]=='YARN_STORE': return YARN
        return CORE
    if branch!='kaito' and 14<=step<20: return CORE
    return CURRENT

def _reduce(action,product,qty):
    out=rg._copy_action(action); market=[]; rem=max(0,int(qty))
    for raw in out.get('market',[]):
        o=list(raw)
        if rem and len(o)>=3 and o[0]=='SELL' and o[1]==product:
            cut=min(rem,max(0,int(o[2]))); o[2]-=cut; rem-=cut
            if o[2]<=0: continue
        market.append(o)
    out['market']=market; return out

def _preempt(obs,action,route,step,seat):
    debt=_DEBT[seat]
    for p in PREEMPT:
        due=int(debt.pop((step,p),0) or 0)
        if due: action=_reduce(action,p,due)
    if step+1>=len(route): return action
    shed=((obs.get('private',{}) or {}).get('shed',{}) or {})
    for p in PREEMPT:
        if len(action.get('market') or [])>=10: break
        future=sum(max(0,int(o[2])) for o in (route[step+1].get('market') or []) if len(o)>=3 and o[0]=='SELL' and o[1]==p)
        if future<=0: continue
        stock=max(0,int(shed.get(p,0) or 0)); qty=min(future,stock)
        if qty<=0 or any(len(o)>=3 and o[0]=='SELL' and o[1]==p for o in (action.get('market') or [])): continue
        action=rg._copy_action(action); action['market']=list(action.get('market') or [])+[['SELL',p,qty]]
        debt[(step+1,p)]=int(debt.get((step+1,p),0))+qty
    return action

def _actor(route,step,idx): return rg._actor_action(route,step,idx)
def _move(pos,a):
    if not a or a[0] not in rg.MOVE_DELTA: return tuple(pos)
    dx,dy=rg.MOVE_DELTA[a[0]]; return (int(pos[0])+dx,int(pos[1])+dy)
def _out_actor(a,idx):
    if idx==0: return list(a.get('farmer') or ['PASS'])
    h=list(a.get('hands') or []); return list(h[idx-1] if idx-1<len(h) else ['PASS'])
def _set(a,idx,cmd):
    out=rg._copy_action(a)
    if idx==0: out['farmer']=list(cmd); return out
    h=[list(x) for x in (out.get('hands') or [])]
    if idx-1<len(h): h[idx-1]=list(cmd); out['hands']=h
    return out

def _bfs(farm,start,target):
    start=tuple(start); target=tuple(target)
    if start==target:return []
    tiles=rg._get(farm,'tiles',[]) or []; H=len(tiles); W=len(tiles[0]) if H else 0
    q=collections.deque([start]); prev={start:None}; how={}
    while q:
        p=q.popleft()
        for cmd,(dx,dy) in rg.MOVE_DELTA.items():
            n=(p[0]+dx,p[1]+dy)
            if n in prev or n[0]<0 or n[1]<0 or n[0]>=W or n[1]>=H or rg._tile_at(farm,n)=='LOCKED': continue
            prev[n]=p; how[n]=cmd
            if n==target:
                seq=[]; cur=n
                while cur!=start: seq.append([how[cur]]); cur=prev[cur]
                return list(reversed(seq))
            q.append(n)
    return None

def _forecast_target(obs,action,route,step,ai,end):
    farm=rg._farm(obs,int(obs.get('player',0))); ps=rg._unit_positions(farm)
    if ai>=len(ps): return None
    p=tuple(ps[ai]); trace='farmer' if ai==0 else ai-1
    for k in range(step,end): p=_move(p,_out_actor(action,ai) if k==step else rg._trace_actor_action(route,k-1,trace))
    return p

def _forecast_helper(obs,action,route,step,h,start):
    farm=rg._farm(obs,int(obs.get('player',0))); ps=rg._unit_positions(farm)
    if h>=len(ps):return None
    p=tuple(ps[h])
    for k in range(step,start): p=_move(p,_out_actor(action,h) if k==step else _actor(route,k,h))
    return p

def _tail(route,start,end,h):
    low=[]
    for k in range(start,end):
        a=_actor(route,k,h); op=a[0] if a else 'PASS'
        if op in rg.MOVE_DELTA or op=='PASS': continue
        if op in LOW:
            low.append((k,list(a)))
            if len(low)>1:return None
        else:return None
    return low

def _find_rescue(obs,action,route,step,seat):
    if step%24==0 or step%24<12:return None
    end=step-step%24+23
    if end>=719:return None
    active=(rg._STATE.get(seat,{}) or {}).get('weed',{}) or {}; farm=rg._farm(obs,seat); ps=rg._unit_positions(farm)
    for actor,tx in list(active.items()):
        ai=0 if actor=='farmer' else int(actor)+1; intended=list(tx.get('intended') or [])
        if len(intended)<2 or intended[:2]!=['PLANT','STRAWBERRY']: continue
        if any((_actor(route,k,ai) or ['PASS'])[0]=='PASS' for k in range(step,end)): continue
        de=rg._trace_actor_action(route,end-1,actor); ce=rg._trace_actor_action(route,end,actor)
        if len(de)<2 or de[:2]!=['PLANT','STRAWBERRY'] or ce!=['WATER']: continue
        target=_forecast_target(obs,action,route,step,ai,end)
        if target is None:continue
        cand=[]
        for h in range(ai+1,len(ps)):
            if h-1 in active:continue
            for start in range(step,end):
                hp=_forecast_helper(obs,action,route,step,h,start); path=_bfs(farm,hp,target) if hp is not None else None
                if path is None or len(path)>end-start:continue
                low=_tail(route,start,end,h)
                if low is None:continue
                cmds=path+[['PASS'] for _ in range((end-start)-len(path))]+[['WATER']]
                cand.append(((len(low),len(cmds),-start,h),start,h,cmds))
        if cand:
            cand.sort(key=lambda x:x[0]); best=cand[0]
            if best[1]==step:return {'start':step,'helper':best[2],'commands':best[3]}
    return None

def agent(obs,configuration=None):
    step=int(obs.get('step',0)); seat=int(obs.get('player',0))
    if step==0:
        _BRANCH[seat]='global'; _DEBT[seat].clear(); _RESCUE[seat]=None
    if step==13: _BRANCH[seat]='kaito' if _kaito_like(obs) else 'global'
    branch=_BRANCH[seat]; route=_route(obs,branch)
    old=rg.PRIMARY_ROUTE
    try: rg.PRIMARY_ROUTE=route; a=rg.agent(obs,configuration)
    finally: rg.PRIMARY_ROUTE=old
    if step==13:
        a=rg._copy_action(a); market=list(a.get('market') or [])
        if branch=='kaito':
            bridge=[['SELL','WHEAT',6],['BUY_SEED','WHEAT',1],['BUY_SEED','MELON',2]]
        else:
            q=3 if _core_opening_like(obs) else 1
            bridge=[['SELL','WHEAT',q],['BUY_SEED','MELON',1]]
        a['market']=(market+bridge)[:10]
    if branch=='kaito':
        a=_preempt(obs,a,route,step,seat)
        a=rg._rank_sell_slots(obs,a,configuration)
        market=[list(o) for o in (a.get('market') or [])]; sells=[o for o in market if len(o)>=3 and o[0]=='SELL']
        if sells:
            rest=[o for o in market if not (len(o)>=3 and o[0]=='SELL')]; a=rg._copy_action(a); a['market']=sells+rest
    # The rescue planner is route/state based, not Kaito-specific.  Apply it to
    # every branch so a weed-delayed STRAWBERRY replay can borrow an otherwise
    # low-value helper and finish the dependency before the day boundary.
    if step%24==0:_RESCUE[seat]=None
    plan=_RESCUE.get(seat)
    if plan is None: plan=_find_rescue(obs,a,route,step,seat); _RESCUE[seat]=plan
    if plan is not None:
        off=step-int(plan['start'])
        if 0<=off<len(plan['commands']):a=_set(a,int(plan['helper']),plan['commands'][off])
        if off>=len(plan['commands'])-1:_RESCUE[seat]=None
    return a
