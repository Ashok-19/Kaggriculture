"""One focused semantic/regression group for the new bounded specialist."""
import copy
import json
import unittest
import shutil
import tempfile
from pathlib import Path
from .agent import GoldV25Plus, controller
from .benchmark import OUT
from .package import PACKAGE
from kaggriculture_bonsai.agents.gold_v25.native import NativeAgent
from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
from kaggriculture_bonsai.agents.gold_v25.extract import ASSETS
from kaggriculture_bonsai.agents.gold_v25.test_gold_v25 import observation
from kaggriculture_bonsai.runtime import bonsai_kg_v4 as bkg


class SpecialistTests(unittest.TestCase):
    def test_anticipation_modes_cpu_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            assets=Path(directory)
            for name in ('controller.bkgc','routes.bkv6','arity.bin','identity.json'):
                shutil.copyfile(ASSETS/name,assets/name)
            for mode in (33,65):
                bkg.save_compiled(controller(0,718,mode),assets/'specialist.bkgc')
                python=GoldV25Plus(0,718,mode)
                native=NativeAgent(PACKAGE/'gold_cpu.so',assets=assets)
                try:
                    for step in (*range(719),719,72,0):
                        obs=observation(step)
                        obs['private']['shed']={'WHEAT':7,'STRAWBERRY':8,'MILK':2,'WOOL':3}
                        self.assertEqual(python(obs),native(obs),(mode,step))
                finally:
                    native.close()

    def test_gate_parent_fallback_and_native_actions(self):
        c=json.loads((PACKAGE/'candidate.json').read_text())['candidate']
        vm=bkg.VM(controller(c['start'],c['end'],c['mode']))
        python=GoldV25Plus(c['start'],c['end'],c['mode'])
        parent=GoldV25Agent()
        native=NativeAgent(PACKAGE/'gold_cpu.so',assets=PACKAGE/'assets')
        self.addCleanup(native.close)
        for step in (*range(719),719,708,0):
            obs=observation(step)
            obs['private']['shed']={'WHEAT':7,'MILK':2,'WOOL':3}
            obs['private']['inventories'][0]={'WOOL':3,'MILK':2}
            expected=python(obs);actual=native(obs);fallback=parent(obs)
            self.assertEqual(expected,actual)
            active=vm.tick(obs).blackboard[0]
            self.assertEqual(bool(active),c['start']<=step<=c['end'])
            self.assertEqual(expected['farmer'],fallback['farmer'])
            self.assertEqual(expected['hands'],fallback['hands'])
            if not active:
                self.assertEqual(expected,fallback)
            elif step>=708:
                self.assertTrue(all(o[0]=='SELL' for o in expected['market']))
                values=[obs['market']['prices'][o[1]]*o[2] for o in expected['market']]
                self.assertEqual(values,sorted(values,reverse=True))
            elif any(o and o[0]!='SELL' for o in fallback['market']):
                self.assertEqual(expected,fallback)


if __name__=='__main__':unittest.main()
