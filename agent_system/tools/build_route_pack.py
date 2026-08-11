#!/usr/bin/env python3
"""Build a provenance-preserving public route pack used by the research agent.

The canonical inputs remain public artifacts/replays. The generated pack is a
compact development asset; it is not evidence that the routes are our original
work. Route metadata deliberately records observable provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalized_hash(actions) -> str:
    raw = json.dumps(actions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--competitor-dir", default="scratch/research/competitors")
    ap.add_argument("--replay-dir", default="scratch/research/replays")
    ap.add_argument("--output", default="agent_system/routes/public_routes.json")
    args = ap.parse_args()
    cdir, rdir = Path(args.competitor_dir), Path(args.replay_dir)

    v27 = load_module(cdir / "v27_exact_main.py", "route_v27")
    lev = load_module(cdir / "rayk_findings_main.py", "route_lev")
    breaking = load_module(cdir / "breaking_the_tie_main.py", "route_breaking")
    replay = json.loads((rdir / "91751097.json").read_text())
    jince = [
        copy.deepcopy(step[1].get("action") or {"farmer": ["PASS"], "hands": [], "market": []})
        for step in replay["steps"][1:720]
    ]

    routes = {
        "v27": {
            "actions": list(v27._REBALANCE_ACTIONS),
            "source": {
                "kind": "KAGGLE_NOTEBOOK_PUBLIC_ARTIFACT",
                "title": "25/27 Strict-Future | v27 Midgame Meta Reset",
                "observable_route_source_episode": 91493566,
                "route_sha256_notebook": "9080682756f5b9fc0f2cae25efdffa69474d40883f693b49f58be2c1893ae5d2",
            },
        },
        "lev": {
            "actions": list(lev._TRACE[:719]),
            "source": {
                "kind": "KAGGLE_NOTEBOOK_PUBLIC_ARTIFACT",
                "artifact": "rayk_findings_main.py",
                "family": "Lev-derived public route",
            },
        },
        "jince": {
            "actions": jince,
            "source": {
                "kind": "KAGGLE_PUBLIC_EPISODE",
                "episode_id": 91751097,
                "player": 1,
                "team": "Jince",
                "note": "observable public action trajectory; no hidden-source claim",
            },
        },
        "breaking": {
            "actions": list(breaking._ACTIONS),
            "source": {
                "kind": "KAGGLE_NOTEBOOK_PUBLIC_ARTIFACT",
                "artifact": "breaking_the_tie_main.py",
                "note": "public route plus counter-preemption controller source",
            },
        },
    }

    base = routes["v27"]["actions"]
    for name, entry in routes.items():
        actions = entry["actions"]
        if len(actions) != 719:
            raise RuntimeError(f"{name}: expected 719 actions, got {len(actions)}")
        entry["normalized_sha256"] = normalized_hash(actions)
        if name == "v27":
            entry["first_full_divergence_from_v27"] = None
            entry["first_core_divergence_from_v27"] = None
            continue
        entry["first_full_divergence_from_v27"] = next(
            (i for i, (a, b) in enumerate(zip(base, actions)) if a != b), None
        )
        entry["first_core_divergence_from_v27"] = next(
            (
                i
                for i, (a, b) in enumerate(zip(base, actions))
                if (a.get("farmer"), a.get("hands")) != (b.get("farmer"), b.get("hands"))
            ),
            None,
        )

    payload = {
        "schema_version": 1,
        "routes": routes,
        "synchronization": {
            "v27_to_jince": {
                "checkpoint_step": 124,
                "verified_core_identical_before_checkpoint": True,
                "verified_cumulative_market_totals_identical_before_checkpoint": True,
                "runtime_enabled_by_default": False,
            }
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({"output": str(out), "bytes": out.stat().st_size,
                      "routes": {k: v["normalized_sha256"] for k, v in routes.items()}}, indent=2))


if __name__ == "__main__":
    main()
