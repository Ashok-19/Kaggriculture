"""Build the deterministic standalone SHOP_ROUTER_0909_PLUS archive."""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PORT = ROOT / "workspace/shop_router_0909/port"
PLUS = ROOT / "workspace/shop_router_0909/plus"
OUT = ROOT / "workspace/shop_router_0909/plus_submission"
STAGE = OUT / "package"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "assets").mkdir(parents=True)
    shutil.copy2(HERE / "main.py", STAGE / "main.py")
    shutil.copy2(HERE / "gold_runtime.py", STAGE / "gold_runtime.py")
    for name in ("routes.bkv6", "arity.bin"):
        shutil.copy2(PORT / name, STAGE / "assets" / name)
    shutil.copy2(PLUS / "controller.bkgc", STAGE / "assets/controller.bkgc")
    identity = {
        "candidate": "BKG4_SHOP_ROUTER_0909_PLUS",
        "controller_sha256": sha(STAGE / "assets/controller.bkgc"),
        "route_pack_sha256": sha(STAGE / "assets/routes.bkv6"),
        "arity_sha256": sha(STAGE / "assets/arity.bin"),
        "parent": "BKG4_SHOP_ROUTER_0909",
        "official_engine": "1.32.7",
        "candidate_spec": json.loads((ROOT / "workspace/shop_router_0909/active_candidate.json").read_text())["candidate"],
    }
    (STAGE / "assets/identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    provenance = (ROOT / "src/kaggriculture_bonsai/agents/shop_router_0909/PROVENANCE.md").read_text()
    (STAGE / "PROVENANCE.md").write_text(provenance + "\nThis archive contains the isolated CPU adapter, frozen plus controller, and parent route assets only.\n")
    license_lines = (ROOT / "src/kaggriculture_bonsai/agents/gold_v25/overlay.py").read_text().splitlines(True)[4:184]
    (STAGE / "LICENSE").write_text("".join(line[2:] if line.startswith("# ") else line for line in license_lines))
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-x", "c++", "-Inative/bkg4/include", "-Inative/bkg4/stateful", str(ROOT / "src/kaggriculture_bonsai/agents/shop_router_0909/reference_port/bridge.cu"), "-o", str(STAGE / "shop_cpu.so")], cwd=ROOT, check=True)
    for path in STAGE.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o755 if path.name in ("main.py", "shop_cpu.so") else 0o644)
            os.utime(path, (0, 0))
    archive = OUT / "submission_shop_router_0909_plus_bkg4.tar.gz"
    OUT.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    files = sorted(path for path in STAGE.rglob("*") if path.is_file())
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(STAGE)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
    hashes = [f"archive_sha256  {sha(archive)}", f"archive_bytes  {archive.stat().st_size}"]
    hashes.extend(f"{sha(path)}  {path.relative_to(STAGE)}" for path in files)
    (OUT / "PACKAGE_HASHES.txt").write_text("\n".join(hashes) + "\n")
    print(archive)
    print("\n".join(hashes))


if __name__ == "__main__":
    main()
