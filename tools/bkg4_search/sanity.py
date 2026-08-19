from __future__ import annotations

import json
from tools.bkg4_search.evolvability import EvolvabilityTree, sanity_fixture, ucb_score


def main() -> int:
    f = sanity_fixture()
    tree = EvolvabilityTree(f["elite"], f["floor"])
    robust = tree.add_accepted(0, f["elite"])
    tree.add_accepted(robust, f["elite_child"])
    brittle = tree.add_accepted(0, f["elite"])
    tree.add_accepted(brittle, f["brittle_child"])
    mediocre = tree.add_accepted(0, f["mediocre"])
    tree.add_accepted(mediocre, f["mediocre"])
    weak = tree.add_accepted(0, f["weak"])
    tree.add_accepted(weak, f["weak"])
    rows = {}
    for name, i in (("robust_elite",robust),("brittle_elite",brittle),("mediocre_broad",mediocre),("weak",weak)):
        n=tree.nodes[i]
        rows[name]={"v":n.v,"Q":n.q,"sigma":n.brittleness,"U_lambda1":ucb_score(n,tree.nodes[0].visits,0.0,1.0),"U_lambda0":ucb_score(n,tree.nodes[0].visits,0.0,0.0)}
    ok = rows["robust_elite"]["U_lambda1"] > rows["brittle_elite"]["U_lambda1"] and rows["robust_elite"]["U_lambda1"] > rows["mediocre_broad"]["U_lambda1"] and rows["mediocre_broad"]["U_lambda1"] > rows["weak"]["U_lambda1"]
    print(json.dumps({"ok":ok,"rows":rows},sort_keys=True))
    return 0 if ok else 1

if __name__=="__main__":raise SystemExit(main())
