from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
from typing import Any,Dict

ROOT=Path(__file__).resolve().parents[2];KAG=ROOT/'kaggle-environments-master';CHAMP=ROOT/'bonsai-kg/integrations/bkg4_champion';sys.path.insert(0,str(KAG));sys.path.insert(0,str(CHAMP))
from kaggle_environments import make
from runtime import BKG4ChampionAgent


def fixed_tape_agent(tape):
 actions=tape['actions']
 def agent(obs,config=None):
  s=int(obs.get('step',0) or 0);return copy.deepcopy(actions[s if s<len(actions) else -1])
 return agent


def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,default=Path('scratch/bkg4_search/discovery_cases.jsonl'));ap.add_argument('--prefix',type=int,default=72);ap.add_argument('--out',type=Path,default=Path('scratch/bkg4_search/base0_prefix72'));a=ap.parse_args();cases=(ROOT/a.cases).resolve() if not a.cases.is_absolute() else a.cases;out=(ROOT/a.out).resolve() if not a.out.is_absolute() else a.out;out.mkdir(parents=True,exist_ok=True);rows=[json.loads(x) for x in cases.read_text().splitlines() if x.strip()];records=[]
 for ci,c in enumerate(rows):
  tape_path=Path(c['tape']);tape=json.loads(tape_path.read_text());champ=BKG4ChampionAgent();acts=[];prefix_meta={}
  def candidate(obs,config=None):
   nonlocal prefix_meta
   step=int(obs.get('step',0) or 0)
   if step==a.prefix:
    prefix_meta={'high_labor':bool(champ.exploits.high_labor),'carrot_latched':bool(champ.exploits.carrot_latched),'h4_trigger':bool(champ.legacy.h4_trigger),'external_ok':bool(champ.legacy.external_ok),'state':champ.state()}
   action=champ(obs,config);acts.append(copy.deepcopy(action));return action
  env=make('kaggriculture',configuration={'episodeSteps':720,'seed':int(c['seed'])},debug=False);opp=fixed_tape_agent(tape);env.run([candidate,opp] if int(c['seat'])==0 else [opp,candidate]);final=env.steps[-1];base_path=out/f'case{ci:03d}_base0.json';obj={'source':{'case':ci,'seed':int(c['seed']),'seat':int(c['seat']),'team':c['team'],'opponent_tape':str(tape_path),'prefix':a.prefix,'base0_identity':champ.identity['combined_agent_identity']},'actions':acts};base_path.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8');ext=champ.legacy.external_route
  rec={**c,'case':ci,'base0_tape':str(base_path),'base0_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'prefix':a.prefix,'prefix_meta':prefix_meta,'h4_trigger':bool(prefix_meta.get('h4_trigger',False)),'search_eligible':not bool(prefix_meta.get('h4_trigger',False)),'external_route':int(ext) if ext is not None else None,'base_margin':float(final[int(c['seat'])].reward)-float(final[1-int(c['seat'])].reward),'base_reward':float(final[int(c['seat'])].reward),'opp_reward':float(final[1-int(c['seat'])].reward)};records.append(rec);print(json.dumps({'case':ci,'team':c['team'],'seat':c['seat'],'margin':rec['base_margin'],'h4':rec['h4_trigger'],'external_route':rec['external_route']},ensure_ascii=False,sort_keys=True),flush=True)
 (out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records),encoding='utf-8');print(json.dumps({'cases':len(records),'eligible':sum(r['search_eligible'] for r in records),'h4':sum(r['h4_trigger'] for r in records),'wins':sum(r['base_margin']>0 for r in records),'losses':sum(r['base_margin']<0 for r in records),'ties':sum(r['base_margin']==0 for r in records)},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
