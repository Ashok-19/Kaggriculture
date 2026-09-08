"""Build a deterministic standalone BKG4 GOLD V25 tarball."""
from pathlib import Path
import gzip
import hashlib
import os
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PORT = ROOT / "workspace/gold_v25_port"
OUT = ROOT / "workspace/gold_v25_submission"
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
    for name in ("controller.bkgc", "routes.bkv6", "arity.bin", "identity.json"):
        shutil.copy2(PORT / name, STAGE / "assets" / name)
    # The source overlay contains the complete Apache-2.0 text; retain only its license block.
    license_lines = (HERE.parent / "overlay.py").read_text().splitlines(True)[4:184]
    (STAGE / "LICENSE").write_text("".join(line[2:] if line.startswith("# ") else line for line in license_lines))
    provenance = (HERE.parent / "PROVENANCE.md").read_text()
    (STAGE / "PROVENANCE.md").write_text(provenance + "\nThis archive contains only the isolated CPU adapter, native runtime, and fixed assets.\n")
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-x", "c++", "-Inative/bkg4/include", "-Inative/bkg4/stateful", str(HERE.parent / "bridge.cu"), "-o", str(STAGE / "gold_cpu.so")], cwd=ROOT, check=True)
    for path in STAGE.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o755 if path.name in ("main.py", "gold_cpu.so") else 0o644)
            os.utime(path, (0, 0))
    archive = OUT / "submission_gold_v25_bkg4.tar.gz"
    if archive.exists():
        archive.unlink()
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(STAGE.rglob("*")):
                    if path.is_file():
                        info = tar.gettarinfo(str(path), arcname=str(path.relative_to(STAGE)))
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
    hashes = [f"archive_sha256  {sha(archive)}", f"archive_bytes  {archive.stat().st_size}"]
    for path in sorted(STAGE.rglob("*")):
        if path.is_file():
            hashes.append(f"{sha(path)}  {path.relative_to(STAGE)}")
    (OUT / "PACKAGE_HASHES.txt").write_text("\n".join(hashes) + "\n")
    print(archive)
    print("\n".join(hashes))


if __name__ == "__main__":
    main()
