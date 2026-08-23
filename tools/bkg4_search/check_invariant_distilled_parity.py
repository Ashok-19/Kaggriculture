from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tools.bkg4_search.invariant_evaluator import load_invariant_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--cuda-results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cuda = load_invariant_results(args.cuda_results)
    checked = 0
    for name, spec in sorted(manifest["modules"].items(), key=lambda x: int(x[1]["candidate_index"])):
        assets = Path(spec["assets"])
        cmd = [
            str(args.runner), "--controller", str(assets / "controller.bkgc"),
            "--routes", str(assets / "routes.bkv6"), "--legacy-routes", str(assets / "legacy_routes.bkv6"),
            "--cases", str(args.cases),
        ]
        for rewrite in spec.get("rewrites") or ():
            cmd.extend(("--rewrite", ",".join(str(x) for x in rewrite)))
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"distilled C++ runner failed for {name}: {proc.stderr[-2000:]}")
        rows = [line.split("\t") for line in proc.stdout.splitlines() if line.strip()]
        expected = cuda[int(spec["candidate_index"])]
        if len(rows) != len(expected):
            raise ValueError(f"case count mismatch for {name}")
        for fields, truth in zip(rows, expected):
            _, own, opp, flags, trace = (int(x) for x in fields)
            got = (own, opp, flags, trace)
            want = (truth.own_money, truth.opp_money, truth.flags, truth.action_trace_hash)
            if got != want:
                raise AssertionError(f"distilled parity mismatch {name}: {got} != {want}")
            checked += 1
    print(json.dumps({"modules": len(manifest["modules"]), "cases_per_module": len(cuda[0]), "exact_rows": checked, "mismatches": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
