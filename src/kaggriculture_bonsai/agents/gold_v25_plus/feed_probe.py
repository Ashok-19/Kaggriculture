"""Small CPU causal probe of observed BUILD wheat-pickup shortages."""
import json
from .agent import GoldV25Plus
from .benchmark import OUT
from kaggriculture_bonsai.agents.gold_v25 import overlay
from kaggriculture_bonsai.agents.gold_v25.extract import reference
from kaggriculture_bonsai.evaluation import cpp_oracle
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


class FeedProbe(GoldV25Plus):
    def __init__(self,start=144,stop=647):
        super().__init__(576,718,9)
        self.events=[]
        self.start,self.stop=start,stop

    def before_market(self,action,obs):
        step=obs['step'];seat=obs.get('player',0)
        if step<self.start or step>=self.stop or len(action['market'])>=10:
            return action
        view=overlay._View(obs,seat,self.parent.chassis.cfg)
        route=self.parent.chassis.players[seat]['route']
        future=self.parent.chassis.routes[route][step+1]
        demand=sum(max(0,u[2] if len(u)>2 else 1) for u in [future['farmer']]+future['hands']
                   if u[:2]==['PICKUP','WHEAT'])
        projected=self.parent.chassis._projected_shed(action,view)
        available=projected.get('WHEAT',0)
        for order in action['market']:
            if len(order)>2 and order[1]=='WHEAT':
                if order[0]=='BUY_PRODUCT':available+=max(0,order[2])
                elif order[0]=='SELL':available-=min(available,max(0,order[2]))
        missing=max(0,demand-available)
        if missing:
            action['market'].append(['BUY_PRODUCT','WHEAT',missing])
            self.events.append({'step':step,'demand':demand,'available':available,'buy':missing})
        return action


class StructureProbe(FeedProbe):
    def __call__(self,obs,cfg=None):
        action=super().__call__(obs,cfg)
        view=overlay._View(obs,obs.get('player',0),self.parent.chassis.cfg)
        units=[action['farmer']]+action['hands']
        for i,unit in enumerate(units):
            tile=overlay._tile_at(view.tiles,view.positions[i])
            kind=tile.get('kind') if isinstance(tile,dict) else None
            inv=view.inv(i)
            for animal,structure,build in (('COW','PASTURE','BUILD_PASTURE'),('SHEEP','PASTURE','BUILD_PASTURE'),('GOOSE','COOP','BUILD_COOP')):
                if inv.get(animal,0)<=0:continue
                if unit[:2]==['PLACE',animal] and kind is None and tile!='LOCKED':
                    units[i]=[build]
                    break
                if kind==structure and tile.get('animal') is None and overlay._is_noop(unit,tile,inv,view.seeds,view.positions[i],view.board):
                    units[i]=['PLACE',animal]
                    break
        action['farmer'],action['hands']=units[0],units[1:]
        return action


class DispatchProbe(StructureProbe):
    def __init__(self):
        super().__init__()
        self.jobs={}
        self.last={}
        self.targets={}
        from .repair_controller import controller
        from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg
        self.repair_vm=bkg.VM(controller())

    def __call__(self,obs,cfg=None):
        seat=obs.get('player',0);step=obs['step']
        result=self.repair_vm.tick(obs,max_market_orders=0)
        if result.fault:raise RuntimeError('repair controller fault')
        if not result.blackboard[0]:return self.parent(obs,cfg)
        if step==0 or step<=self.last.get(seat,-1):
            self.jobs.pop(seat,None)
            self.targets[seat]={}
        self.last[seat]=step
        action=super().__call__(obs,cfg)
        view=overlay._View(obs,seat,self.parent.chassis.cfg)
        state=self.parent.chassis.players[seat]
        targets=self.targets.setdefault(seat,{})
        for pending in state['pending'].values():
            for pos,build in pending:
                if build[0] in ('BUILD_PASTURE','BUILD_COOP'):targets[tuple(pos)]=build
        for pos in view.positions:
            tile=overlay._tile_at(view.tiles,pos)
            if isinstance(tile,dict) and tile.get('kind') in ('PASTURE','COOP'):targets.pop(tuple(pos),None)
        if not self.jobs.get(seat) and action['farmer']==['PASS']:
            x,y=view.positions[0]
            for (tx,ty),build in sorted(targets.items(),key=lambda item:item[0][1]*10+item[0][0]):
                if build[0] not in ('BUILD_PASTURE','BUILD_COOP'):continue
                distance=abs(tx-x)+abs(ty-y)
                length=2*distance+1
                if not 0<distance<=2 or step+length>718:continue
                route=self.parent.chassis.routes[state['route']]
                if any(route[t]['farmer']!=['PASS'] for t in range(step,step+length)):continue
                moves=[['EAST' if tx>x else 'WEST'] for _ in range(abs(tx-x))]+[['SOUTH' if ty>y else 'NORTH'] for _ in range(abs(ty-y))]
                opposite={'EAST':'WEST','WEST':'EAST','NORTH':'SOUTH','SOUTH':'NORTH'}
                self.jobs[seat]=moves+[build]+[[opposite[u[0]]] for u in reversed(moves)]
                del targets[(tx,ty)]
                break
        if self.jobs.get(seat):action['farmer']=self.jobs[seat].pop(0)
        return action


class SeedProbe(DispatchProbe):
    def __init__(self):
        super().__init__()
        self.seed_requirements={}
        from .repair_controller import controller
        self.repair_vm=bkg.VM(controller(seed_reserve=True))

    def __call__(self,obs,cfg=None):
        action=super().__call__(obs,cfg)
        step=obs['step'];seat=obs.get('player',0)
        if not 144<=step<647:return action
        view=overlay._View(obs,seat,self.parent.chassis.cfg)
        seeds=dict(view.seeds);planted=set()
        for i,u in enumerate([action['farmer']]+action['hands']):
            if len(u)<2 or u[0]!='PLANT':continue
            pos=tuple(view.positions[i]);tile=overlay._tile_at(view.tiles,pos)
            if tile is None and pos not in planted and seeds.get(u[1],0)>0:
                seeds[u[1]]-=1;planted.add(pos)
        route_id=self.parent.chassis.players[seat]['route']
        route=self.parent.chassis.routes[route_id]
        if route_id not in self.seed_requirements:
            def preceding(future,need):
                plants={k:0 for k in bkg.ITEMS[:5]}
                buys=dict(plants)
                for u in [future['farmer']]+future['hands']:
                    if len(u)>1 and u[0]=='PLANT':plants[u[1]]+=1
                for o in future['market']:
                    if len(o)>2 and o[0]=='BUY_SEED':buys[o[1]]+=max(0,o[2])
                return {k:plants[k]+max(0,need[k]-buys[k]) for k in plants}
            tails=[]
            for tail in (2,3):
                need={k:0 for k in bkg.ITEMS[:5]}
                for t in range(718,647,-1):need=preceding(self.parent.chassis.routes[tail][t],need)
                tails.append(need)
            table=[None]*649
            table[648]={k:max(tails[0][k],tails[1][k]) for k in tails[0]}
            for t in range(647,-1,-1):table[t]=preceding(route[t],table[t+1])
            self.seed_requirements[route_id]=table
        for order in action['market']:
            if len(order)<3 or order[0]!='BUY_SEED':continue
            item=order[1];required=self.seed_requirements[route_id][step+1][item]
            order[2]=min(order[2],max(0,required-seeds.get(item,0)))
            seeds[item]=seeds.get(item,0)+max(0,order[2])
        return action


def main():
    out=OUT/'round5';out.mkdir(exist_ok=True)
    cases=[]
    for seed in (990057,990042,990017,990031):
        for seat in (0,1):
            candidate=FeedProbe();rival=reference()['agent']
            agents=[rival,rival];agents[seat]=candidate
            money,_,_=cpp_oracle.run(*agents,seed=seed)
            margin=money[seat]-money[1-seat]
            cases.append({'seed':seed,'seat':seat,'money':money,'margin':margin,'events':candidate.events})
            print(seed,seat,margin,len(candidate.events),flush=True)
    (out/'causal_probe.json').write_text(json.dumps({'status':'BUILD_ONLY_PYTHON_PROBE_NOT_NATIVE_CANDIDATE','cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
