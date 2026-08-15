"""Build the pybind11 bridge for the Python interpreter running this script."""
from __future__ import annotations

import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "cpp-engine"
OUT = ROOT / "scratch" / "cpp-engine-build"
OUT.mkdir(parents=True, exist_ok=True)
suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
target = OUT / f"kaggriculture_cpp{suffix}"
includes = shlex.split(subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True))
cmd = [
    "g++", "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC",
    *includes, f"-I{CPP / 'include'}", str(CPP / "src" / "pybridge.cpp"), "-o", str(target),
]
print(" ".join(shlex.quote(x) for x in cmd))
subprocess.run(cmd, check=True)
print(target)
