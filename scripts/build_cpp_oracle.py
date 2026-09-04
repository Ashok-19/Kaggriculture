"""Build the pybind11 bridge for the Python interpreter running this script."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "native" / "cpp"
OUT = ROOT / "workspace" / "build" / "cpp"
OUT.mkdir(parents=True, exist_ok=True)
suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
target = OUT / f"kaggriculture_cpp{suffix}"
includes = shlex.split(subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True))
flags = [
    "-O3", "-march=native", "-flto", "-fipa-pta", "-fno-semantic-interposition",
    "-fvisibility=hidden", "-std=c++17", "-shared", "-fPIC", "-pthread",
]
pgo_mode = os.environ.get("KAG_CPP_PGO", "").strip().lower()
pgo_dir = OUT / "pgo"
if pgo_mode == "generate":
    shutil.rmtree(pgo_dir, ignore_errors=True)
    pgo_dir.mkdir(parents=True, exist_ok=True)
    flags.append(f"-fprofile-generate={pgo_dir}")
elif pgo_mode == "use":
    if not pgo_dir.exists() or not any(pgo_dir.rglob("*.gcda")):
        raise SystemExit(f"PGO profile data not found under {pgo_dir}; run KAG_CPP_PGO=generate and train first")
    flags.extend((f"-fprofile-use={pgo_dir}", "-fprofile-correction"))
elif pgo_mode:
    raise SystemExit("KAG_CPP_PGO must be unset, 'generate', or 'use'")
cmd = [
    "g++", *flags,
    *includes, f"-I{CPP / 'include'}", str(CPP / "src" / "pybridge.cpp"), "-o", str(target),
]
print(" ".join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True)
print(target)
