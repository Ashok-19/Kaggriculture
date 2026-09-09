"""Compare legal observations on the weakest fresh BUILD case, not holdouts."""
import json
from .benchmark import OUT
from .agent import GoldV25Plus
from kaggriculture_bonsai.agents.gold_v25.extract import reference
from kaggriculture_bonsai.evaluation import cpp_oracle


def main():
    data=json.loads((OUT/'round3/diagnostic.json').read_text())
    case=min(data['candidate'],key=lambda r:r['margin'])
    seed,seat=case['seed'],case['seat']
    game=cpp_oracle._load_bridge().Game(seed,{})
    own=GoldV25Plus(576,718,9);other=reference()['agent']
    agents=[other,other];agents[seat]=own
    events=[];last=0
    while not game.done:
        obs=[game.observation(p) for p in (0,1)]
        actions=[a(o) for a,o in zip(agents,obs)]
        step=obs[0]['step'];farms=obs[0]['farms']
        margin=farms[seat]['money']-farms[1-seat]['money']
        units=[[a['farmer']]+a['hands'] for a in actions]
        differences=[i for i,(a,b) in enumerate(zip(*units)) if a!=b]
        if differences or (margin!=last and (step<576 or step%24==0)):
            events.append({'step':step,'margin':margin,'unit_differences':differences,
                'units':[[u[i] for i in differences] for u in units],
                'market':[a['market'] for a in actions],
                'shed':[o['private']['shed'] for o in obs],
                'inventory':[[o['private']['inventories'][i] for i in differences] for o in obs],
                'seeds':[o['private']['seeds'] for o in obs]})
        last=margin;game.step(*actions)
    result={'seed':seed,'seat':seat,'events':events}
    (OUT/'round3/failure_diagnosis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
