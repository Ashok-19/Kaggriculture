"""Counterexample/refinement and deterministic BKG4 identity contracts."""
import unittest

from kaggriculture_bonsai.search.phase2_ccpr import identity, matches, metrics, program, refine


class CCPRContracts(unittest.TestCase):
    def test_counterexamples_drive_one_predicate_refinement(self):
        parent = program(359, 1)
        rows = []
        for seed in (1, 2):
            for seat in (0, 1):
                for money, delta in ((100, 20), (10, -30)):
                    rows.append({"features": [money] + [0] * 11, "delta": delta, "seed": seed,
                                 "seat": seat, "duration": 48, "root": len(rows), "transition": "T->W" if delta > 0 else "T->L"})
        child, operation, witnesses = refine(parent, rows, [(0, "GE", 100)], {identity(parent)})
        self.assertEqual(operation, "WHEN")
        self.assertEqual(len(witnesses), 4)
        self.assertEqual(child["mode"], parent["mode"])
        self.assertEqual(child["duration"], parent["duration"])
        self.assertTrue(all(matches(r["features"], child["predicates"]) == (r["delta"] > 0) for r in rows))
        self.assertEqual(refine(parent, list(reversed(rows)), [(0,"GE",100)], {identity(parent)})[:2], (child, operation))
        self.assertFalse(metrics(rows)["safe"])

    def test_stop_and_semantic_identity(self):
        p = program(359, 1, predicates=[(0,"GE",100),(1,"LE",5)])
        self.assertEqual(identity(p), identity(program(359,1,predicates=[(1,"LE",5),(0,"GE",100)])))
        rows = [{"delta": -1,"duration":48,"root":17}]
        child, operation, witnesses = refine(p, rows, [], {identity(p)})
        self.assertEqual(operation,"STOP")
        self.assertEqual(child["duration"],24)
        self.assertEqual(child["predicates"],p["predicates"])
        self.assertEqual(witnesses,[17])
        self.assertNotEqual(identity(child),identity(p))


if __name__ == "__main__":unittest.main()
