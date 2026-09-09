"""Locate BUILD money asymmetry without inspecting downstream seeds."""
import json
from kaggriculture_bonsai.agents.gold_v25.extract import reference
from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
from kaggriculture_bonsai.evaluation import cpp_oracle
from .benchmark import OUT


def main():
    game = cpp_oracle._load_bridge().Game(910001, {})
    agents = [GoldV25Agent(), reference()['agent']]
    events, previous, last_actions = [], 0, None
    while not game.done:
        obs = [game.observation(p) for p in (0, 1)]
        margin = obs[0]['farms'][0]['money'] - obs[0]['farms'][1]['money']
        if margin != previous:
            events.append({'step': obs[0]['step'], 'old_margin': previous, 'new_margin': margin,
                           'previous_actions': last_actions, 'private': [o['private'] for o in obs],
                           'market': obs[0]['market']})
        previous = margin
        last_actions = [a(o) for a,o in zip(agents,obs)]
        game.step(*last_actions)
    (OUT/'failure_diagnosis.json').write_text(json.dumps({'seed':910001,'events':events},indent=2)+'\n')
    print(json.dumps(events,indent=2))


if __name__=='__main__':
    main()
