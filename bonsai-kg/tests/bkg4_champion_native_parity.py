from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KAG_ENV = ROOT / "kaggle-environments-master"
BKG_PY = ROOT / "bonsai-kg" / "python"
CHAMPION = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion"
for p in (KAG_ENV, BKG_PY, CHAMPION):
    sys.path.insert(0, str(p))

from kaggle_environments import make
import bonsai_kg_v4 as bkg
from runtime import BKG4ChampionAgent


class RecordingObservation:
    def __init__(self, raw: dict):
        self.inner = bkg.DictObservation(raw)
        self.values: dict[tuple[int, int, int], int] = {}

    def feature(self, fid: int, a0: int = 0, a1: int = 0) -> int:
        v = int(self.inner.feature(fid, a0, a1))
        self.values[(int(fid), int(a0), int(a1))] = v
        return v


def snapshot(case_id: int, tick_id: int, res) -> str:
    a = res.action
    s = f"C={case_id} T={tick_id} S={res.status} F={res.fault} U={a.n_units}:" + ";".join(
        f"{op},{arg},{n}" for op, arg, n in a.units
    )
    s += f" M={len(a.market)}:" + ";".join(f"{op},{item},{n}" for op, item, n in a.market)
    s += " B=" + ",".join(str(x) for x in res.blackboard)
    s += " N=" + ",".join(f"{x}/{y}" for x, y in res.node_state)
    return s


def write_seq(path: Path, frames: list[dict[tuple[int, int, int], int]]) -> None:
    lines = [f"BKS2 2 {len(frames)}"]
    for frame in frames:
        lines.append(str(len(frame)))
        for (fid, a0, a1), v in sorted(frame.items()):
            lines.append(f"{fid} {a0} {a1} {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1810000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--work", type=Path, default=Path("scratch/bkg4_search/native_parity_feed"))
    ap.add_argument("--cpp", type=Path, default=Path("scratch/bonsai_kg/v4_cpp_differential_runner"))
    ap.add_argument("--cuda", type=Path, default=Path("scratch/bonsai_kg/v4_cuda_differential_runner"))
    args = ap.parse_args()
    work = (ROOT / args.work).resolve() if not args.work.is_absolute() else args.work
    work.mkdir(parents=True, exist_ok=True)
    asset_dir = CHAMPION / "champions" / "feed_deferral_v1"
    policy_path = asset_dir / "controller.bkgc"
    cp = bkg.load_compiled(policy_path)

    manifest = []
    expected: list[str] = []
    case_id = 0
    for k in range(args.seeds):
        seed = args.seed_start + k
        for seat in (0, 1):
            champion = BKG4ChampionAgent(asset_dir)
            oracle_vm = bkg.VM(cp)
            frames = []
            tick = 0

            def agent(raw, configuration=None):
                nonlocal tick
                rec = RecordingObservation(dict(raw))
                res = oracle_vm.tick(rec, max_market_orders=0, trace=False)
                frames.append(dict(rec.values))
                expected.append(snapshot(case_id, tick, res))
                tick += 1
                return champion(raw, configuration)

            env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 720}, debug=True)
            env.run([agent, "pass"] if seat == 0 else ["pass", agent])
            seq = work / f"case_{case_id:02d}.bks"
            write_seq(seq, frames)
            manifest.append((policy_path, seq))
            case_id += 1

    mf = work / "manifest.txt"
    mf.write_text("".join(f"{p} {s}\n" for p, s in manifest), encoding="utf-8")
    mismatches = {}
    for name, exe in (("cpp", args.cpp), ("cuda", args.cuda)):
        path = (ROOT / exe).resolve() if not exe.is_absolute() else exe
        got = subprocess.run([str(path), str(mf)], check=True, capture_output=True, text=True).stdout.splitlines()
        bad = []
        if len(got) != len(expected):
            bad.append(("length", len(expected), len(got)))
        for i, (want, have) in enumerate(zip(expected, got)):
            if want != have:
                bad.append((i, want, have))
                if len(bad) >= 10:
                    break
        mismatches[name] = bad
    report = {
        "cases": case_id,
        "records": len(expected),
        "policy_hash": cp.semantic_hash,
        "nodes": len(cp.nodes),
        "bb": len(cp.bb),
        "state": cp.state_count,
        "mismatches": {k: len(v) for k, v in mismatches.items()},
        "first": {k: (v[0] if v else None) for k, v in mismatches.items()},
    }
    print(json.dumps(report, sort_keys=True))
    return 1 if any(mismatches.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
