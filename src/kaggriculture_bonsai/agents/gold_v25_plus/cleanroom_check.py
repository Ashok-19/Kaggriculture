"""Copy into the package and run with python -E from outside the repository."""
import json
import main

farm={'money':3000,'farmer':[4,4],'hands':[],'tiles':[[None]*10 for _ in range(10)],'unlocked_quadrants':['NW'],'hires_today':0}
obs={'step':708,'player':0,'farms':[farm,farm],'town':{'unlocked_shops':[]},
     'market':{'inventory':{'EGG':10000},'prices':{'MILK':100,'WOOL':100}},
     'private':{'shed':{'WOOL':3,'MILK':2},'seeds':{},'inventories':[{}]}}
assert main.agent(obs)['market']==[['SELL','WOOL',3],['SELL','MILK',2]]
print(json.dumps({'standalone_import':'PASS','specialist_action':'PASS'}))
