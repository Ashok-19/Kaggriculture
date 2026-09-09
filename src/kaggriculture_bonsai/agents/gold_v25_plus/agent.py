"""Experimental endgame specialist with exact parent fallback outside its gate."""
import json
from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
from kaggriculture_bonsai.agents.gold_v25.extract import ROOT
from kaggriculture_bonsai.agents.gold_v25 import overlay
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


def controller(start, end, mode=5):
    active={'type':'SetBlackboard','slot':'active','value':mode}
    if mode in (9,17,33,65):
        active={'type':'If','condition':{'type':'Condition','lhs':{'feature':'STEP'},'op':'GE','rhs':708},
                'then':{'type':'SetBlackboard','slot':'active','value':5},'else':{'type':'SetBlackboard','slot':'active','value':1 if mode==9 else mode}}
    return bkg.compile_policy({'schema_version':4, 'name':'GOLD_V25_PLUS_ENDGAME_GATE', 'blackboard':[{'name':'active','type':'I32','initial':0}],
        'tree':{'type':'If','condition':{'type':'Sequence','children':[
            {'type':'Condition','lhs':{'feature':'STEP'},'op':'GE','rhs':start},
            {'type':'Condition','lhs':{'feature':'STEP'},'op':'LE','rhs':end}]},
            'then':active,
            'else':{'type':'SetBlackboard','slot':'active','value':0}}})


class GoldV25Plus:
    def __init__(self, start, end, mode):
        self.parent = GoldV25Agent()
        self.vm = bkg.VM(controller(start,end,mode))
        self.mode = mode

    def __call__(self, obs, cfg=None):
        action = self.parent(obs,cfg)
        action = self.before_market(action,obs)
        result = self.vm.tick(obs,max_market_orders=0)
        if result.fault:
            raise RuntimeError('specialist BKG4 fault')
        if not result.blackboard[0] or not self.mode:
            return action
        view = overlay._View(obs, int(obs.get('player',0)), self.parent.chassis.cfg)
        mode=result.blackboard[0]
        if mode & 2:
            units = [action['farmer']] + action['hands']
            for i,pos in enumerate(view.positions):
                if overlay._shed_adjacent(pos,view.board) and view.inv(i):
                    units[i] = ['DROP']
            action['farmer'], action['hands'] = units[0], units[1:]
        projected = self.parent.chassis._projected_shed(action,view)
        if mode in (33,65) and obs['step']%4 and all(not o or o[0]=='SELL' for o in action['market']):
            planned={}
            already={o[1] for o in action['market'] if o and o[0]=='SELL'}
            route=self.parent.chassis.players[int(obs.get('player',0))]['route']
            for dt in range(1,(2 if mode==33 else 3)+1):
                nxt=obs['step']+dt
                if nxt>718 or nxt%72==0:
                    break
                for order in self.parent.chassis.routes[route][nxt]['market']:
                    if order and order[0]=='SELL' and order[1] in bkg.ITEMS[1:8]:
                        planned[order[1]]=planned.get(order[1],0)+max(0,order[2])
                if nxt%4==0:
                    break
            for k in bkg.ITEMS[1:8]:
                n=min(planned.get(k,0),projected.get(k,0))
                if k not in already and n>0 and view.prices.get(k,0)>=2 and len(action['market'])<10:
                    action['market'].append(['SELL',k,n])
        if mode & 16:
            if any(o and o[0]!='SELL' for o in action['market']):
                return action
            for k in bkg.ITEMS[1:8]:
                if projected.get(k,0)<=0:
                    continue
                found=False
                for order in action['market']:
                    if order and order[0]=='SELL' and order[1]==k:
                        order[2]=projected[k]
                        found=True
                if not found and len(action['market'])<10:
                    action['market'].append(['SELL',k,projected[k]])
        if mode & 4:
            action['market'] = [['SELL',k,projected[k]] for k in overlay.PRODUCTS if projected.get(k,0)>0]
        if mode & 1 and all(not o or o[0]=='SELL' for o in action['market']):
            action['market'].sort(key=lambda o: -view.prices.get(o[1],0)*min(projected.get(o[1],0),o[2]) if o else 0)
        return action

    def before_market(self,action,obs):
        return action


_DEFAULT=None


def agent(observation, configuration=None):
    """Entry for the explicitly selected isolated candidate, never the champion."""
    global _DEFAULT
    if _DEFAULT is None:
        path=ROOT/'workspace/gold_v25_plus/active_candidate.json'
        if not path.exists():path=ROOT/'workspace/gold_v25_plus/round2/composition.json'
        candidate=json.loads(path.read_text())['candidate']
        if candidate['mode']==1025:
            from .feed_probe import SeedProbe
            _DEFAULT=SeedProbe()
        else:
            _DEFAULT=GoldV25Plus(candidate['start'],candidate['end'],candidate['mode'])
    return _DEFAULT(observation,configuration)
