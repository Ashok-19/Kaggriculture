#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="agent_system/build")
    ap.add_argument("--out-dir", default="submission/build")
    ap.add_argument("--archive", default="submission/submission.tar.gz")
    ns = ap.parse_args()
    build = Path(ns.build_dir); out = Path(ns.out_dir)
    subprocess.run(["cmake", "-S", "agent_system", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "-j", "4"], check=True)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    submission_files = (
        "main.py", "runtime_gateway.py",
        "meta_gold_v1.py", "meta_gold_v8.py", "meta_gold_v10.py",
        "meta_gold_v23_latestart.py", "meta_gold_candidate.py", "meta_gold_v29_minimax.py",
        "kaito44_majority_route.json", "kaito_yarn5_route.json",
    )
    for name in submission_files:
        shutil.copy2(Path("submission") / name, out / name)
    shutil.copy2("agent_system/routes/public_routes.json", out / "routes.json")
    shutil.copy2(build / "libkag_agent.so", out / "libkag_agent.so")
    lib = ctypes.CDLL(str((out / "libkag_agent.so").resolve()))
    lib.kag_agent_abi_version.restype = ctypes.c_int
    if lib.kag_agent_abi_version() != 1: raise RuntimeError("native ABI smoke failed")
    archive = Path(ns.archive); archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        for p in sorted(out.iterdir()): tf.add(p, arcname=p.name)
    print(f"built {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
