from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INTEGRATION = HERE.parent / "integrations" / "stateful_v6"
ORACLE_DIR = ROOT / "scratch" / "bonsai_cpu" / "stateful_core_v6"
sys.path.insert(0, str(INTEGRATION))
sys.path.insert(0, str(HERE.parent / "python"))
import bonsai_kg as bkg
from build_assets import LEAF_ID, ROUTE_IDS, build_route_pack
from runtime import RoutePack, StatefulV6BKG3Agent


def load_oracle():
    spec = importlib.util.spec_from_file_location("stateful_core_v6_oracle_test", ORACLE_DIR / "main.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def obs(step=144, shops=None, milk=100, tomato=100, own_money=1000, opp_money=1000, hands=0):
    tiles0 = [[None for _ in range(10)] for _ in range(10)]
    tiles1 = [[None for _ in range(10)] for _ in range(10)]
    prices = {name: 100 for name in bkg.ITEMS[:9]}
    prices["MILK"] = milk
    prices["TOMATO"] = tomato
    return {
        "step": step, "day": step // 24, "hour": step % 24, "player": 0,
        "farms": [
            {"money": own_money, "tiles": tiles0, "farmer": [0,0], "hands": [[0,0] for _ in range(hands)], "unlocked_quadrants": ["NW"]},
            {"money": opp_money, "tiles": tiles1, "farmer": [0,0], "hands": [], "unlocked_quadrants": ["NW"]},
        ],
        "town": {"unlocked_shops": list(shops or [])},
        "market": {"prices": prices, "inventory": {}},
        "private": {"shed": {}, "seeds": {}, "inventories": [{} for _ in range(hands + 1)]},
    }


STEP144_CASES = [
    ("A1",29,dict(shops=["BAKERY","BRUNCH_SPOT"],milk=192)),
    ("A2",64,dict(shops=["ICE_CREAM_SHOP","BRUNCH_SPOT"],milk=192)),
    ("B",64,dict(shops=["BAKERY","PET_CAFE"],milk=192,own_money=310)),
    ("C1",64,dict(shops=["BAKERY","PET_CAFE"],milk=192,own_money=311)),
    ("C2",32,dict(shops=["BAKERY","SMOOTHIE_SHOP"],milk=192,own_money=311)),
    ("D1",79,dict(shops=["BAKERY","BRUNCH_SPOT"],milk=193,opp_money=56)),
    ("D2",32,dict(shops=["BAKERY","BRUNCH_SPOT"],milk=193,opp_money=57)),
    ("E",64,dict(shops=["BAKERY","PET_CAFE"],milk=193)),
    ("F",70,dict(shops=["PIZZA_SHOP","BRUNCH_SPOT"],milk=193)),
    ("G",63,dict(shops=["BAKERY","PIZZA_SHOP"],milk=193)),
    ("H",22,dict(shops=["YARN_STORE","BRUNCH_SPOT"],tomato=62)),
    ("I",34,dict(shops=["PET_CAFE","YARN_STORE"],tomato=62)),
    ("J1",77,dict(shops=["YARN_STORE","YARN_STORE"],tomato=62)),
    ("J2",34,dict(shops=["YARN_STORE","SMOOTHIE_SHOP"],tomato=62)),
    ("K",46,dict(shops=["YARN_STORE","BRUNCH_SPOT"],tomato=63)),
]


class StatefulV6Translation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = load_oracle()
        cls.agent = StatefulV6BKG3Agent(INTEGRATION / "assets")

    def bkg_select144(self, observation):
        self.agent.reset()
        self.agent(observation)
        s = self.agent.state()
        return s["leaf"], s["route"]

    def bkg_select216(self, leaf, route, observation):
        self.agent.reset()
        self.agent.vm.state.blackboard[0] = LEAF_ID[leaf]
        self.agent.vm.state.blackboard[1] = route
        self.agent.vm.state.blackboard[2] = 1
        self.agent.vm.state.blackboard[3] = 0
        self.agent.last_step = 215
        self.agent(observation)
        return self.agent.state()["effective_route"]

    def test_01_step144_branch_matrix_exact(self):
        seen = set()
        for leaf, route, kw in STEP144_CASES:
            o = obs(step=144, **kw)
            self.assertEqual(self.oracle._select144(copy.deepcopy(o)), (leaf, route), leaf)
            self.assertEqual(self.bkg_select144(copy.deepcopy(o)), (leaf, route), leaf)
            seen.add(leaf)
        self.assertEqual(seen, set(LEAF_ID))

    def test_01b_missing_shop_semantics_exact(self):
        o=obs(step=144,shops=[],milk=192)
        self.assertEqual(self.oracle._select144(copy.deepcopy(o)),("A1",29))
        self.assertEqual(self.bkg_select144(copy.deepcopy(o)),("A1",29))
        o216=obs(step=216,shops=[])
        self.assertEqual(self.oracle._select216("B",copy.deepcopy(o216),64),64)
        self.assertEqual(self.bkg_select216("B",64,copy.deepcopy(o216)),64)

    def test_02_step216_branch_matrix_exact(self):
        cases = [
            ("A1",29,dict(shops=["BAKERY","BRUNCH_SPOT"]),29),
            ("A1",29,dict(shops=["SMOOTHIE_SHOP","BRUNCH_SPOT"]),32),
            ("A1",29,dict(shops=["ICE_CREAM_SHOP","BRUNCH_SPOT"],own_money=715),29),
            ("A1",29,dict(shops=["ICE_CREAM_SHOP","BRUNCH_SPOT"],own_money=716),23),
            ("B",64,dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"]),64),
            ("B",64,dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"]),46),
            ("C1",64,dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"]),64),
            ("C1",64,dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"]),34),
            ("D2",32,dict(shops=["BAKERY","BRUNCH_SPOT","SMOOTHIE_SHOP"]),32),
            ("D2",32,dict(shops=["BAKERY","BRUNCH_SPOT","YARN_STORE"]),46),
            ("E",64,dict(shops=["BAKERY","BRUNCH_SPOT"]),79),
            ("E",64,dict(shops=["PIZZA_SHOP","BRUNCH_SPOT"]),63),
            ("E",64,dict(shops=["PET_CAFE","BRUNCH_SPOT","SMOOTHIE_SHOP"]),64),
            ("E",64,dict(shops=["PET_CAFE","BRUNCH_SPOT","YARN_STORE"]),46),
        ]
        unchanged = {"A2":64,"C2":32,"D1":79,"F":70,"G":63,"H":22,"I":34,"J1":77,"J2":34,"K":46}
        for leaf,current,kw,expected in cases:
            o=obs(step=216,**kw)
            self.assertEqual(self.oracle._select216(leaf,copy.deepcopy(o),current),expected)
            self.assertEqual(self.bkg_select216(leaf,current,copy.deepcopy(o)),expected)
        for leaf,current in unchanged.items():
            o=obs(step=216,shops=["YARN_STORE","YARN_STORE","YARN_STORE"])
            self.assertEqual(self.oracle._select216(leaf,copy.deepcopy(o),current),current)
            self.assertEqual(self.bkg_select216(leaf,current,copy.deepcopy(o)),current)

    def test_03_complete_python_route_pack_parity(self):
        identity=json.loads((INTEGRATION/"assets/identity.json").read_text())
        pack=RoutePack(INTEGRATION/"assets/routes.bkv6",identity)
        comparisons=0
        for rid in ROUTE_IDS:
            source=json.loads((ORACLE_DIR/f"route{rid}.json").read_text())["actions"]
            for step,a in enumerate(source):
                hands=len(a.get("hands",[]) or [])
                self.assertEqual(pack.action(rid,step,hands),a,(rid,step))
                comparisons+=1
        self.assertEqual(comparisons,7909)

    def test_04_dynamic_hand_adaptation_exact(self):
        identity=json.loads((INTEGRATION/"assets/identity.json").read_text())
        pack=RoutePack(INTEGRATION/"assets/routes.bkv6",identity)
        source=json.loads((ORACLE_DIR/"route64.json").read_text())["actions"]
        for step in (0,143,144,216,718,719):
            src=source[min(step,len(source)-1)];saved=list(src.get("hands",[]) or [])
            for n in (0,1,4,8,12,15):
                expected={"farmer":copy.deepcopy(src.get("farmer") or ["PASS"]),"hands":[copy.deepcopy(saved[i]) if i<len(saved) else ["PASS"] for i in range(n)],"market":copy.deepcopy(list(src.get("market",[]) or []))}
                self.assertEqual(pack.action(64,step,n),expected,(step,n))

    def test_05_step_boundaries_and_latching(self):
        oracle=load_oracle();new=StatefulV6BKG3Agent(INTEGRATION/"assets")
        kw=dict(shops=["BAKERY","BRUNCH_SPOT"],milk=192,tomato=100,own_money=500,opp_money=100)
        for step in (0,143,144,145,215,216,217,719):
            o=obs(step=step,**kw)
            self.assertEqual(new(copy.deepcopy(o)),oracle.agent(copy.deepcopy(o)),step)
            s=new.state()
            self.assertEqual(s["stage144_done"],step>=144)
            self.assertEqual(s["stage216_done"],step>=216)
            if step>=144:self.assertEqual((s["leaf"],s["effective_route"]),("A1",29))
            else:self.assertEqual(s["effective_route"],64)

    def test_06_episode_reset_and_step_rollback(self):
        oracle=load_oracle();new=StatefulV6BKG3Agent(INTEGRATION/"assets")
        pre=obs(step=100,shops=["YARN_STORE","BRUNCH_SPOT"],tomato=63)
        self.assertEqual(new(copy.deepcopy(pre)),oracle.agent(copy.deepcopy(pre)))
        reset0=obs(step=0,shops=["BAKERY","BRUNCH_SPOT"],milk=192)
        self.assertEqual(new(copy.deepcopy(reset0)),oracle.agent(copy.deepcopy(reset0)))
        self.assertFalse(new.state()["stage144_done"]);self.assertEqual(new.state()["effective_route"],64)
        pre=obs(step=200,shops=["YARN_STORE","BRUNCH_SPOT"],tomato=63)
        self.assertEqual(new(copy.deepcopy(pre)),oracle.agent(copy.deepcopy(pre)))
        self.assertEqual(new.state()["leaf"],"K")
        rollback=obs(step=50,shops=["BAKERY","BRUNCH_SPOT"],milk=192)
        self.assertEqual(new(copy.deepcopy(rollback)),oracle.agent(copy.deepcopy(rollback)))
        self.assertFalse(new.state()["stage144_done"]);self.assertEqual(new.state()["effective_route"],64)

    def test_07_identity_and_asset_rebuild_are_deterministic(self):
        identity=json.loads((INTEGRATION/"assets/identity.json").read_text())
        rebuilt,meta=build_route_pack(ORACLE_DIR)
        self.assertEqual(meta["pack_sha256"],identity["route_pack"]["pack_sha256"])
        self.assertEqual(rebuilt,(INTEGRATION/"assets/routes.bkv6").read_bytes())
        cp=bkg.load_compiled(INTEGRATION/"assets/controller.bkgc")
        self.assertEqual(cp.semantic_hash,identity["controller_semantic_hash"])
        rebuilt_cp=bkg.compile_policy(json.loads((INTEGRATION/"assets/controller.json").read_text()))
        self.assertEqual(bkg.semantic_payload_bytes(cp),bkg.semantic_payload_bytes(rebuilt_cp))


if __name__ == "__main__":
    unittest.main(verbosity=2)
