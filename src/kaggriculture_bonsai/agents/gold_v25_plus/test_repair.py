"""Focused action/state/reset coverage for the construction recovery queue."""
import ctypes as c
import unittest
from pathlib import Path
from .feed_probe import SeedProbe
from .benchmark import OUT
from kaggriculture_bonsai.agents.gold_v25.native import NativeAgent
from kaggriculture_bonsai.agents.gold_v25.extract import ASSETS
from kaggriculture_bonsai.agents.gold_v25.test_gold_v25 import observation
from kaggriculture_bonsai.evaluation import cpp_oracle
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


def snapshot(native,seat):
    fn=native.lib.gold_repair_snapshot
    fn.argtypes=[c.c_void_p,c.c_int,c.POINTER(c.c_int),c.c_int];fn.restype=c.c_int
    data=(c.c_int*122)();assert fn(native.handle,seat,data,122)==122
    names={v:k for k,v in bkg.UNIT_OPS.items()}
    return {'last':data[0],'targets':{(i%10,i//10):[names[data[2+i]]] for i in range(100) if data[2+i]},
            'jobs':[[names[data[102+i*4]]] for i in range(data[1])]}


class RepairTests(unittest.TestCase):
    def test_build_histories_and_resets(self):
        native=NativeAgent(Path('/tmp/gold_v25_plus_package/gold_cpu.so'),assets=Path('/tmp/gold_v25_plus_package/assets'))
        self.addCleanup(native.close)
        python=SeedProbe()
        for seed in (990057,1020047,1050019,1080046):
            game=cpp_oracle._load_bridge().Game(seed,{})
            while not game.done:
                actions=[]
                for seat in (0,1):
                    obs=game.observation(seat);expected=python(obs);actual=native(obs)
                    self.assertEqual(expected,actual,(seed,seat,obs['step']))
                    state=snapshot(native,seat)
                    self.assertEqual(state,{'last':python.last[seat],'targets':python.targets[seat],'jobs':python.jobs.get(seat,[])})
                    actions.append(actual)
                game.step(*actions)
        for step in (33,33,0,144,719,0):
            for seat in (0,1):
                obs=observation(step,seat)
                self.assertEqual(python(obs),native(obs))
                self.assertEqual(snapshot(native,seat),{'last':python.last[seat],'targets':python.targets[seat],'jobs':python.jobs.get(seat,[])})


if __name__=='__main__':unittest.main()
