import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from kaggriculture_bonsai.agents.shop_router_0909.reference_port import agent as mech

# Compact source programme only. No donor route rows are loaded or consulted.
_PROGRAMME = json.loads(Path(__file__).with_name("p0_programme.json").read_text())
CROPS = {k: tuple(v) for k, v in _PROGRAMME["crops"].items()}
ANIMALS = {k: tuple(v) for k, v in _PROGRAMME["animals"].items()}
HIRES = [int(v) for v in _PROGRAMME["hires"]]
LAND_DAYS = {int(v) for v in _PROGRAMME["land_days"]}
CROP_SEGMENTS = {(int(x), int(y)): [(int(d), str(c)) for d, c in segs]
                 for x, y, segs in _PROGRAMME["crop_segments"]}
ANIMAL_ACQ = [(int(d), (int(x), int(y)), str(sp)) for d, x, y, sp in _PROGRAMME["animal_acq"]]
PRODUCTS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
ANIMAL_NAMES=('GOOSE','COW','SHEEP')
SHED={(4,4),(5,4),(4,5),(5,5)}

@dataclass(frozen=True)
class Task:
    kind: str
    site: tuple
    start: int
    end: int
    item: str

@dataclass
class SeatState:
    day: int = -1
    task_ids: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    cursor: list = field(default_factory=list)
    completed: set = field(default_factory=set)
    failed: int = 0
    carried_over: int = 0
    fert_loads: int = 0


def crop_segment(site, day):
    xs=CROP_SEGMENTS.get(site,())
    cur=None; end=30
    for i,(d,c) in enumerate(xs):
        if d<=day:
            cur=(d,c); end=xs[i+1][0] if i+1<len(xs) else 30
    if cur is None or day>=end:return None
    return cur[0],cur[1],end

def animal_owner(site, day):
    xs=[x for x in ANIMAL_ACQ if x[1]==site and x[0]<=day]
    if not xs:return None
    a=max(xs,key=lambda x:x[0]); acq,_,sp=a
    future=min([d for d,c in CROP_SEGMENTS.get(site,()) if d>acq],default=30)
    return a if day<future else None

def task_catalog(day):
    out=[]
    # Ownership resolves crop/animal overlap. Asset programme, not observed scheduling.
    sites=set(CROP_SEGMENTS)
    sites.update(site for _,site,_ in ANIMAL_ACQ)
    for site in sorted(sites,key=lambda p:(p[1],p[0])):
        ao=animal_owner(site,day)
        cs=crop_segment(site,day)
        if ao is not None and (cs is None or ao[0]>=cs[0]):
            acq,_,sp=ao
            out.append(Task('animal_acquire' if day==acq else 'animal_service',site,acq,30,sp))
        elif cs is not None:
            start,crop,end=cs
            out.append(Task('crop_acquire' if day==start else 'crop_service',site,start,end,crop))
    return out

def expected_roles(day): return 1+HIRES[day]

def spawn_positions(hands):
    ps=[(4,4)]; acc=[(4,4),(5,4),(4,5),(5,5)]
    for _ in range(hands):
        occ=[ps.count(p) for p in acc]; k=min(range(4),key=lambda i:(occ[i],i)); ps.append(acc[k])
    return ps

def manhattan(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])

def _assign_roles_uncached(day,tasks):
    n=expected_roles(day); starts=spawn_positions(HIRES[day])
    animals=[i for i,t in enumerate(tasks) if t.kind.startswith('animal')]
    crops=[i for i,t in enumerate(tasks) if t.kind.startswith('crop')]
    # Reserved-role ownership: bounded animal pool and crop pool. The compiler may
    # optimize only geometric route binding/order; it never changes obligations.
    na=min(max(2,n//3),4,n)
    animal_workers=list(range(na)); crop_workers=list(range(na,n)) or [0]
    roles=[[] for _ in range(n)]
    def body(t):
        return 2 if t.kind.startswith('crop') else (2 if t.kind=='animal_acquire' else 3)
    def route_cost(w,route):
        cur=starts[w]; c=0
        for ti in route:
            t=tasks[ti]; c += manhattan(cur,t.site)+body(t); cur=t.site
        return c
    def bind_insert(ids,workers):
        remaining=list(ids)
        while remaining:
            best=None
            for j,ti in enumerate(remaining):
                for w in workers:
                    old=route_cost(w,roles[w])
                    for k in range(len(roles[w])+1):
                        cand=roles[w][:k]+[ti]+roles[w][k:]
                        newc=route_cost(w,cand)
                        # Minimize maximum route length first, then insertion increment.
                        maxc=max([newc if q==w else route_cost(q,roles[q]) for q in workers])
                        score=(maxc,newc-old,newc,w,k,tasks[ti].site[1],tasks[ti].site[0])
                        if best is None or score<best[0]: best=(score,j,w,k,ti)
            _,j,w,k,ti=best; remaining.pop(j); roles[w].insert(k,ti)
    bind_insert(animals,animal_workers)
    bind_insert(crops,crop_workers)
    return roles

_ROLE_CACHE={}
def assign_roles(day,tasks):
    tpl=_ROLE_CACHE.get(day)
    if tpl is None:
        tpl=tuple(tuple(r) for r in _assign_roles_uncached(day,tasks))
        _ROLE_CACHE[day]=tpl
    return [list(r) for r in tpl]

def pos_of(farm,u):
    if u==0:return tuple(farm['farmer'])
    hs=farm['hands']; return tuple(hs[u-1]) if u-1<len(hs) else None

def inv_of(obs,u):
    invs=obs['private']['inventories']
    return invs[u] if u<len(invs) else {}

def tile_of(farm,site):
    x,y=site; return farm['tiles'][y][x]

def step_toward(p,q):
    x,y=p;tx,ty=q
    if x<tx:return ['EAST']
    if x>tx:return ['WEST']
    if y<ty:return ['SOUTH']
    if y>ty:return ['NORTH']
    return ['PASS']

def nearest_shed(p):return min(SHED,key=lambda q:(manhattan(p,q),q[1],q[0]))

def crop_action(task,tile,day):
    crop=task.item; first,mature,ongoing=CROPS[crop]
    if tile is None:return ['PLANT',crop],False
    if not isinstance(tile,dict): return ['PASS'],False
    kind=tile.get('kind')
    if kind=='WEED':return ['DIG'],False
    if kind!='PLANT' or tile.get('crop')!=crop:
        # Programme transition: clear prior non-animal asset, then establish declared crop.
        if kind in ('PLANT','COOP','PASTURE') and not tile.get('animal'):return ['DIG'],False
        return ['PASS'],False
    if not tile.get('watered_today',False):return ['WATER'],False
    age=day-int(tile.get('planted_day',task.start))
    harvest_age = first if ongoing else mature
    if age>=harvest_age and int(tile.get('yield_units',0))>0:return ['HARVEST'],False
    # After a finite harvest the next observation is EMPTY and will replant unless the segment ends.
    return ['PASS'],True

def animal_action(task,tile,day,inv,pos):
    sp=task.item; first,interval,build=ANIMALS[sp]
    if task.kind=='animal_acquire':
        if tile is not None and not isinstance(tile,dict): return ['PASS'],False
        if tile and tile.get('animal')==sp:return ['PASS'],True
        if int(inv.get(sp,0))<=0:
            if pos in SHED:return ['PICKUP',sp,1],False
            return step_toward(pos,nearest_shed(pos)),False
        if tile is None:return [build],False
        if tile.get('kind')=='WEED':return ['DIG'],False
        want='COOP' if sp=='GOOSE' else 'PASTURE'
        if tile.get('kind')!=want:
            if not tile.get('animal'):return ['DIG'],False
            return ['PASS'],False
        return ['PLACE',sp],False
    # Persistent lifecycle service. Missing declared livestock is a recovery acquisition, never silently completed.
    if not isinstance(tile,dict) or tile.get('animal')!=sp:
        if int(inv.get(sp,0))<=0:
            if pos in SHED:return ['PICKUP',sp,1],False
            return step_toward(pos,nearest_shed(pos)),False
        if tile is None:return [build],False
        if not isinstance(tile,dict):return ['PASS'],False
        if tile.get('kind')=='WEED':return ['DIG'],False
        want='COOP' if sp=='GOOSE' else 'PASTURE'
        if tile.get('kind')!=want:
            if not tile.get('animal'):return ['DIG'],False
            return ['PASS'],False
        return ['PLACE',sp],False
    if not tile.get('fed_today',False):
        if int(inv.get('WHEAT',0))<=0:
            if pos in SHED:return ['PICKUP','WHEAT',1],False
            return step_toward(pos,nearest_shed(pos)),False
        return ['FEED'],False
    if not tile.get('cared_today',False):return ['CARE'],False
    if tile.get('fertilizer_available',False):return ['COLLECT_FERTILIZER'],False
    age=day-int(tile.get('placed_day',task.start))
    if age>=first and int(tile.get('yield_units',0))>0:return ['HARVEST'],False
    return ['PASS'],True

def total_item(obs,item):
    n=int(obs['private']['shed'].get(item,0))
    for inv in obs['private']['inventories']:
        n += int(inv.get(item,0))
    return n

def task_due(task,obs):
    if not task.kind.startswith('crop_service'): return True
    p=int(obs['player']); farm=obs['farms'][p]; tile=tile_of(farm,task.site); day=int(obs['step'])//24
    if not isinstance(tile,dict) or tile.get('kind')!='PLANT' or tile.get('crop')!=task.item: return True
    first,mature,ongoing=CROPS[task.item]; age=day-int(tile.get('planted_day',task.start))
    if int(tile.get('consecutive_unwatered',0))>=1: return True
    if ongoing and int(tile.get('yield_units',0))>0: return True
    if not ongoing:
        w0=(mature+1)//2
        if age>=w0: return True  # build finite yield through its harvest window
    return False

def task_blocked(task,obs,u):
    p=int(obs['player']); farm=obs['farms'][p]; tile=tile_of(farm,task.site)
    if not isinstance(tile,dict) and tile is not None: return True  # locked geometry
    if task.kind.startswith('animal'):
        if isinstance(tile,dict) and tile.get('animal')==task.item: return False
        return total_item(obs,task.item)<=0
    if task.kind.startswith('crop'):
        if isinstance(tile,dict) and tile.get('kind')=='PLANT' and tile.get('crop')==task.item: return False
        return int(obs['private']['seeds'].get(task.item,0))<=0
    return False

def task_action(task,obs,u):
    p=int(obs['player']); farm=obs['farms'][p]; pos=pos_of(farm,u)
    if pos is None:return ['PASS'],False
    tile=tile_of(farm,task.site); inv=inv_of(obs,u); day=int(obs['step'])//24
    # Input pickup is part of the complete service/acquisition tour.
    if task.kind.startswith('animal'):
        missing = not (isinstance(tile,dict) and tile.get('animal')==task.item)
        # Acquisition and recovery service both own the livestock cargo handoff.
        if missing and int(inv.get(task.item,0))<=0:
            if pos in SHED:return ['PICKUP',task.item,1],False
            return step_toward(pos,nearest_shed(pos)),False
        # Ordinary service input is acquired before leaving the shed.
        if not missing and int(inv.get('WHEAT',0))<=0 and pos in SHED:
            return ['PICKUP','WHEAT',1],False
        if pos!=task.site:
            return step_toward(pos,task.site),False
        return animal_action(task,tile,day,inv,pos)
    if pos!=task.site:return step_toward(pos,task.site),False
    if isinstance(tile,dict) and tile.get('kind')=='PLANT' and tile.get('crop')==task.item:
        if task.item in ('STRAWBERRY','MELON','TOMATO') and int(inv.get('FERTILIZER',0))>0:
            if int(tile.get('fertilized_until_day',-1)) < day and not tile.get('watered_today',False):
                return ['FERTILIZE'],False
    return crop_action(task,tile,day)

def finance_intents(obs,day):
    p=int(obs['player']); shed=obs['private']['shed']; farm=obs['farms'][p]
    orders=[]
    # Realization receipts first. Preserve only current feed reserve.
    live=0; unfed=0
    for _,site,sp in ANIMAL_ACQ:
        t=tile_of(farm,site)
        if isinstance(t,dict) and t.get('animal')==sp:
            live += 1
            if not t.get('fed_today',False): unfed += 1
    reserve=unfed
    high_value_crop=any(t.kind.startswith('crop') and t.item in ('STRAWBERRY','MELON','TOMATO') for t in task_catalog(day))
    for item in ('STRAWBERRY','MELON','MILK','WOOL','EGG','CARROT','TOMATO','FERTILIZER'):
        q=int(shed.get(item,0))
        if item=='FERTILIZER' and high_value_crop: q=max(0,q-2)
        if q>0: orders.append(['SELL',item,q])
    total_wheat=total_item(obs,'WHEAT')
    excess=max(0,total_wheat-reserve)
    if excess:
        sell_q=min(int(shed.get('WHEAT',0)),excess)
        if sell_q: orders.append(['SELL','WHEAT',sell_q])
    # Labour programme is retried until the declared daily count is actually realized.
    hires_done=int(farm.get('hires_today',0)); desired=HIRES[day]
    orders += [['HIRE'] for _ in range(max(0,desired-hires_done))]
    # Maintenance contracts outrank expansion: feed existing livestock before new capital assets.
    feed_need=max(0,reserve-total_wheat)
    if feed_need: orders.append(['BUY_PRODUCT','WHEAT',feed_need])
    # Missing crop establishment/renewal seeds are derived from current legal asset state.
    seed_need=defaultdict(int)
    for site in CROP_SEGMENTS:
        cs=crop_segment(site,day)
        ao=animal_owner(site,day)
        if cs is None or (ao is not None and ao[0]>=cs[0]): continue
        start,crop,end=cs; t=tile_of(farm,site)
        if not (isinstance(t,dict) and t.get('kind')=='PLANT' and t.get('crop')==crop):
            seed_need[crop]+=1
        elif not CROPS[crop][2]:
            age=day-int(t.get('planted_day',start))
            if age>=CROPS[crop][1] and int(t.get('yield_units',0))>0 and day+1<end: seed_need[crop]+=1
    seeds=obs['private']['seeds']
    for crop,n in sorted(seed_need.items()):
        need=max(0,n-int(seeds.get(crop,0)))
        if need: orders.append(['BUY_SEED',crop,need])
    # Expansion and new acquisition consume only resources left after maintenance obligations.
    target_land=3 if day>=11 else (2 if day>=6 else 1)
    if len(farm.get('unlocked_quadrants', ['NW'])) < target_land: orders.append(['BUY_LAND'])
    missing=defaultdict(int)
    for acq,site,sp in ANIMAL_ACQ:
        if acq>day: continue
        future=min([d for d,c in CROP_SEGMENTS.get(site,()) if d>acq],default=30)
        if day>=future: continue
        t=tile_of(farm,site)
        if not (isinstance(t,dict) and t.get('animal')==sp): missing[sp]+=1
    for sp,n in sorted(missing.items()):
        need=max(0,n-total_item(obs,sp))
        if need: orders.append(['BUY_ANIMAL',sp,need])
    return orders

def land_target(day): return 3 if day>=11 else (2 if day>=6 else 1)

def land_pending(obs,day):
    farm=obs['farms'][int(obs['player'])]
    return len(farm.get('unlocked_quadrants',['NW'])) < land_target(day)

def realization_action(obs,u,day):
    # Financing-dependent realization tour: only interrupt service while a declared land
    # dependency is unresolved. Cargo is deposited non-destructively with PLACE.
    if not land_pending(obs,day): return None
    p=int(obs['player']); farm=obs['farms'][p]; pos=pos_of(farm,u); inv=inv_of(obs,u)
    for item in ('WOOL','MILK','EGG','STRAWBERRY','MELON','CARROT','TOMATO'):
        q=int(inv.get(item,0))
        if q>0:
            if pos not in SHED: return step_toward(pos,nearest_shed(pos))
            return ['PLACE',item,q]
    return None

class EventP0:
    def __init__(self):self.state={}
    def __call__(self,obs,cfg=None):
        step=int(obs['step']);day=step//24;hour=step%24;p=int(obs['player']);farm=obs['farms'][p]
        st=self.state.setdefault(p,SeatState())
        if st.day!=day:
            # Any unfinished semantic tasks remain explicit failures; next day's programme is regenerated from persistent asset obligations.
            if st.day>=0:
                for w,role in enumerate(st.roles):
                    if st.cursor[w]<len(role):st.carried_over += len(role)-st.cursor[w]
            tasks=task_catalog(day); st.day=day; st.roles=assign_roles(day,tasks); st.cursor=[0]*len(st.roles); st.task_ids=tasks; st.completed=set(); st.fert_loads=0
        n=1+len(farm['hands']); units=[]
        # Explicit terminal realization programme: once the horizon closes, stop taking
        # new productive obligations and return all carried output to the shed.
        if step>=710:
            for u in range(n):
                pos=pos_of(farm,u); inv=inv_of(obs,u)
                if inv:
                    if pos not in SHED: units.append(step_toward(pos,nearest_shed(pos)))
                    else: units.append(['DROP'])
                else:
                    units.append(step_toward(pos,nearest_shed(pos)) if pos not in SHED else ['PASS'])
            act={'farmer':units[0] if units else ['PASS'],'hands':units[1:],'market':[]}
            # Sell shed stock continuously while workers arrive; step 718 still uses exact
            # Shop liquidation below to include same-turn DROP projections.
            if step<718:
                for item in PRODUCTS:
                    q=int(obs['private']['shed'].get(item,0))
                    if q>0 and len(act['market'])<10: act['market'].append(['SELL',item,q])
                return act
            return mech.liquidate(mech.FarmView(obs))
        for u in range(n):
            if u>=len(st.roles):units.append(['PASS']);continue
            role=st.roles[u]
            # Complete animal-tour cargo contract: at the shed, batch all feed / livestock
            # needed by this role's remaining owned sites before leaving the service cluster.
            pos=pos_of(farm,u); inv=inv_of(obs,u)
            if pos in SHED and st.cursor[u] < len(role):
                rem=[st.task_ids[ti] for ti in role[st.cursor[u]:] if st.task_ids[ti].kind.startswith('animal')]
                if rem:
                    feed_need=0
                    acquire=defaultdict(int)
                    for t in rem:
                        tile=tile_of(farm,t.site)
                        if isinstance(tile,dict) and tile.get('animal')==t.item:
                            if not tile.get('fed_today',False): feed_need += 1
                        else:
                            acquire[t.item] += 1
                    have=int(inv.get('WHEAT',0)); stock=int(obs['private']['shed'].get('WHEAT',0))
                    if feed_need>have and stock>0:
                        q=min(feed_need-have,stock); units.append(['PICKUP','WHEAT',q]); continue
                    loaded=False
                    for sp in sorted(acquire):
                        have_sp=int(inv.get(sp,0)); stock_sp=int(obs['private']['shed'].get(sp,0))
                        if acquire[sp]>have_sp and stock_sp>0:
                            q=min(acquire[sp]-have_sp,stock_sp); units.append(['PICKUP',sp,q]); loaded=True; break
                    if loaded: continue
                # Source-derived fertilizer+water motif: at most two crop service roles per day
                # carry one fertilizer unit for high-value crop visits.
                crop_rem=[st.task_ids[ti] for ti in role[st.cursor[u]:] if st.task_ids[ti].kind.startswith('crop')]
                if crop_rem and st.fert_loads < 2 and int(inv.get('FERTILIZER',0))<=0:
                    if int(obs['private']['shed'].get('FERTILIZER',0))>0 and any(t.item in ('STRAWBERRY','MELON','TOMATO') and task_due(t,obs) for t in crop_rem):
                        units.append(['PICKUP','FERTILIZER',1]); st.fert_loads += 1; continue
            # Advance only on observed task completion; no absolute action cursor.
            attempts=0
            while st.cursor[u]<len(role) and attempts < max(1,len(role)-st.cursor[u]):
                ti=role[st.cursor[u]]; task=st.task_ids[ti]
                if not task_due(task,obs):
                    st.completed.add(ti); st.cursor[u]+=1; attempts=0; continue
                if task_blocked(task,obs,u):
                    role.append(role.pop(st.cursor[u])); attempts += 1; continue
                a,done=task_action(task,obs,u)
                if done:
                    st.completed.add(ti);st.cursor[u]+=1;attempts=0;continue
                units.append(a);break
            else:
                # Once a reserved tour is complete, it may realize carried sale cargo for a
                # still-pending land financing dependency; active service is never interrupted.
                ra=realization_action(obs,u,day) if st.cursor[u]>=len(role) else None
                units.append(ra if ra is not None else ['PASS'])
        if not units:units=[['PASS']]
        act={'farmer':units[0],'hands':units[1:],'market':[]}
        if step==0:
            act['market']=[['BUY_PRODUCT','WHEAT',13],['SELL','WHEAT',13],['BUY_PRODUCT','WHEAT',13]]
        elif step==1:
            act['market']=[['SELL','WHEAT',13],['BUY_PRODUCT','WHEAT',5],['HIRE'],['HIRE'],['HIRE'],['HIRE'],['HIRE'],['BUY_ANIMAL','COW',2],['BUY_ANIMAL','SHEEP',2]]
        elif day==0:
            # Remaining opening seeds are programme-derived and bought once worker execution begins.
            if step==2:act['market']=[['BUY_SEED','MELON',12],['BUY_SEED','WHEAT',7]]
        else:
            # Event-driven financing queue: every still-unsatisfied declared obligation is retried after
            # current observable realization receipts. This changes timing, never the strategy object.
            act['market']=finance_intents(obs,day)[:10]
        if step==718:
            # Reuse the proven Shop terminal realization mechanic exactly: reachable worker
            # cargo is dropped and projected shed stock is sold in value order.
            return mech.liquidate(mech.FarmView(obs))
        return act
