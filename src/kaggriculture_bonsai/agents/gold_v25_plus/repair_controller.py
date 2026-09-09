"""Compact frozen gates for the round-7 recovery composition."""
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


def controller(seed_reserve=False):
    tree={'type':'SetBlackboard','slot':'active','value':48}
    for step,value in ((144,120 if seed_reserve else 56),(576,121 if seed_reserve else 57),(647,49),(708,53),(719,0)):
        tree={'type':'If','condition':{'type':'Condition','lhs':{'feature':'STEP'},'op':'GE','rhs':step},
              'then':{'type':'SetBlackboard','slot':'active','value':value},'else':tree}
    return bkg.compile_policy({'schema_version':4,'name':'GOLD_V25_PLUS_REPAIR_V7',
        'blackboard':[{'name':'active','type':'I32','initial':-800 if seed_reserve else -700}],'tree':tree})
