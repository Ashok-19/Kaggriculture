"""Fresh parent self-play BUILD diagnosis, with frozen independent seed panels."""
import json
from pathlib import Path

from kaggriculture_bonsai.agents.gold_v25.extract import reference, ROOT
from kaggriculture_bonsai.agents.gold_v25.agent import GoldV25Agent
from kaggriculture_bonsai.evaluation import cpp_oracle

OUT = ROOT / "workspace/gold_v25_plus"


def panels():
    return {"BUILD": list(range(910000, 910016)), "INTERNAL_DEV": list(range(920000, 920032)),
            "FINAL_TARGET": list(range(930000, 930128)), "seats": [0, 1],
            "final_target_status": "UNOPENED", "parent": "BKG4_GOLD_V25",
            "target": "frozen notebook v25", "starting_head": "47b304c8954216b99705df92c510612d7068f6eb"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = panels()
    (OUT / "panels.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for seed in manifest["BUILD"]:
        for seat in (0, 1):
            parent = GoldV25Agent()
            rival = reference()["agent"]
            telemetry = []
            def subject(obs, cfg=None):
                action = parent(obs, cfg)
                step = obs["step"]
                if step % 24 == 23 or step >= 716:
                    own, other = obs["farms"][seat], obs["farms"][1-seat]
                    telemetry.append({"step": step, "margin": own["money"] - other["money"],
                        "route": parent.chassis.players[seat]["route"], "shed": obs["private"]["shed"],
                        "inventories": obs["private"]["inventories"] if step >= 716 else None,
                        "positions": [own["farmer"]] + own["hands"] if step >= 716 else None,
                        "prices": obs["market"]["prices"] if step >= 716 else None,
                        "action": action if step >= 716 else None})
                return action
            agents = [rival, rival]
            agents[seat] = subject
            rewards, _, _ = cpp_oracle.run(*agents, seed=seed)
            margin = rewards[seat] - rewards[1-seat]
            rows.append({"seed": seed, "seat": seat, "rewards": rewards, "margin": margin,
                         "outcome": "W" if margin > 0 else "L" if margin < 0 else "T", "telemetry": telemetry})
            print(seed, seat, margin, flush=True)
    result = {"games": len(rows), "wtl": {o: sum(r["outcome"] == o for r in rows) for o in "WTL"}, "cases": rows}
    (OUT / "parent_build.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(result["wtl"])


if __name__ == "__main__":
    main()
