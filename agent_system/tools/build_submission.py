#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import tarfile
from pathlib import Path


SHADOW_MAIN = """from meta_gold_v29_minimax import agent as _meta_agent


def agent(observation):
    return _meta_agent(observation)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="agent_system/build")
    ap.add_argument("--out-dir", default="submission/build")
    ap.add_argument("--archive", default="submission/submission.tar.gz")
    ns = ap.parse_args()
    build = Path(ns.build_dir); out = Path(ns.out_dir)
    subprocess.run(["cmake", "-S", "agent_system", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "-j", "2"], check=True)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    submission_files = (
        "main.py", "runtime_gateway.py", "donor.json",
        "meta_gold_v1.py", "meta_gold_v8.py", "meta_gold_v10.py",
        "meta_gold_v23_latestart.py", "meta_gold_candidate.py", "meta_gold_v29_minimax.py",
        "kaito44_majority_route.json", "kaito_yarn5_route.json",
        "h4_pre.json", "shadow_market_david.json", "arda_donor.json",
        "h4_donor.json", "w14_donor.json",
    )
    for name in submission_files:
        shutil.copy2(Path("submission") / name, out / name)
    shutil.copy2("agent_system/routes/public_routes.json", out / "routes.json")
    shutil.copy2(build / "libkag_agent.so", out / "libkag_agent.so")

    # Preserve the exact q8-v601 fallback in an isolated sibling module graph.
    shadow = out / "q8_shadow"
    shadow.mkdir()
    for name in (
        "runtime_gateway.py", "meta_gold_v1.py", "meta_gold_v8.py", "meta_gold_v10.py",
        "meta_gold_v23_latestart.py", "meta_gold_candidate.py",
        "kaito44_majority_route.json", "kaito_yarn5_route.json",
    ):
        shutil.copy2(Path("submission") / name, shadow / name)
    shutil.copy2("submission/q8_fert_v601.py", shadow / "meta_gold_v29_minimax.py")
    (shadow / "main.py").write_text(SHADOW_MAIN)
    shutil.copy2("agent_system/routes/public_routes.json", shadow / "routes.json")
    shutil.copy2(build / "libkag_agent.so", shadow / "libkag_agent.so")

    lib = ctypes.CDLL(str((out / "libkag_agent.so").resolve()))
    lib.kag_agent_abi_version.restype = ctypes.c_int
    if lib.kag_agent_abi_version() != 1: raise RuntimeError("native ABI smoke failed")

    # Kaggle validates raw Python with exec(code, env={}) and does not define
    # __file__. Mirror those loader semantics so packaging-path bugs fail here.
    raw_main = out / "main.py"
    import sys
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(out.resolve()))
        raw_env = {}
        exec(compile(raw_main.read_text(), str(raw_main), "exec"), raw_env)
        if not callable(raw_env.get("agent")):
            raise RuntimeError("raw-loader smoke failed: agent callable missing")
    finally:
        sys.path[:] = old_path

    # The raw-loader smoke imports sibling modules and may create bytecode caches.
    # Keep the submission archive source-only apart from the intentional native library.
    for pycache in out.rglob("__pycache__"):
        shutil.rmtree(pycache)
    archive = Path(ns.archive); archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        for p in sorted(x for x in out.rglob("*") if x.is_file()):
            tf.add(p, arcname=p.relative_to(out))
    print(f"built {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
