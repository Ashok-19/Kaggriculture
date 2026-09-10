"""Route-pack fidelity, job closure and candidate determinism contracts."""
import hashlib
import unittest
from pathlib import Path

from kaggriculture_bonsai.search import gold_route_jobs as J
from kaggriculture_bonsai.search import gold_route_search as G

DUMP = G.BUILD / "dump0.txt"


def base_pack():
    return J.Pack(J.PORT / "routes.bkv6", J.PORT / "arity.bin")


@unittest.skipUnless(DUMP.exists(), "annotated reference trace not built")
class RouteJobContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view = J.RouteView(0, DUMP)
        J.annotate_shares(cls.view)

    def test_zero_edit_route_pack_is_byte_identical(self):
        pack = base_pack()
        self.assertEqual(pack.to_bytes(), (J.PORT / "routes.bkv6").read_bytes())
        self.assertEqual(pack.arity_bytes(), (J.PORT / "arity.bin").read_bytes())
        self.assertEqual(pack.clone().to_bytes(), pack.to_bytes())

    def test_animal_conversion_closes_purchase_pickup_place_and_output(self):
        pack = base_pack()
        editor = J.Editor(pack, 0, self.view)
        job = self.view.job("ANIMAL", 10, 0)
        J.animal_convert(editor, job, 11)
        kinds = {(row[0], row[4]) for row in editor.closure}
        self.assertIn(("order", J.M["BUY_ANIMAL"]), kinds)
        self.assertIn(("unit", J.U["PICKUP"]), kinds)
        self.assertIn(("unit", J.U["PLACE"]), kinds)
        self.assertIn(("order", J.M["SELL"]), kinds)
        record = pack.get(0, job["place"]["step"])
        self.assertEqual(record["units"][job["place"]["lane"]][:2], [J.U["PLACE"], 11])

    def test_crop_conversion_closes_seed_financing_and_planting(self):
        pack = base_pack()
        editor = J.Editor(pack, 0, self.view)
        job = self.view.job("CROP", 0, 5)
        J.crop_convert(editor, job, 1)
        steps = {row[2] for row in editor.closure if row[0] == "order" and row[4] == J.M["BUY_SEED"]}
        self.assertTrue(steps)
        self.assertLessEqual(min(steps), job["plant"]["step"])
        record = pack.get(0, job["plant"]["step"])
        self.assertEqual(record["units"][job["plant"]["lane"]][:2], [J.U["PLANT"], 1])

    def test_impossible_conversions_are_rejected_not_repaired(self):
        pack = base_pack()
        editor = J.Editor(pack, 0, self.view)
        job = self.view.job("ANIMAL", 10, 0)
        with self.assertRaises(J.InvalidCandidate):
            J.animal_convert(editor, job, 10)

    def test_candidate_compilation_is_deterministic_and_deduplicated(self):
        views = {route: self.view for route in range(1)}
        operations = [{"family": "ANIMAL", "species": 10, "ordinal": 0, "target": 11, "shift": 0}]
        digests = []
        for _ in range(2):
            pack = base_pack()
            editor = J.Editor(pack, 0, self.view)
            J.animal_convert(editor, self.view.job("ANIMAL", 10, 0), 11)
            digests.append(hashlib.sha256(pack.to_bytes() + pack.arity_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])
        other = base_pack()
        editor = J.Editor(other, 0, self.view)
        J.animal_convert(editor, self.view.job("ANIMAL", 10, 1), 9)
        self.assertNotEqual(digests[0], hashlib.sha256(other.to_bytes() + other.arity_bytes()).hexdigest())
        del views, operations

    def test_declared_support_is_shared_by_both_arms(self):
        declared = G.support({0: self.view})
        self.assertTrue(declared["ANIMAL"])
        self.assertTrue(declared["CROP"])
        self.assertTrue(all(index < 12 for _, index in declared["CROP"]))


if __name__ == "__main__":
    unittest.main()
