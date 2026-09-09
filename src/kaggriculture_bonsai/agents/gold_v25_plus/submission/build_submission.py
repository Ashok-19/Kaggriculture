"""Publish the exact validated V8 archive in a separate submission directory."""
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from kaggriculture_bonsai.agents.gold_v25.extract import ROOT, sha
from kaggriculture_bonsai.agents.gold_v25_plus import package

EVIDENCE = ROOT / 'workspace/gold_v25_plus/round8'
OUT = ROOT / 'workspace/gold_v25_plus_submission'


def main():
    release = json.loads((EVIDENCE / 'release.json').read_text())
    expected = json.loads((EVIDENCE / 'package.json').read_text())
    assert release['status'] == 'COMPLETE_OBSERVED_100_PERCENT_WITH_ZERO_PROTECTED_REGRESSION'
    # Rebuild without rewriting the frozen validation evidence.
    with tempfile.TemporaryDirectory(prefix='gold_plus_submission_') as directory:
        temporary = Path(directory)
        for name in ('composition.json', 'specialist.bkgc'):
            shutil.copyfile(EVIDENCE / name, temporary / name)
        package.main(temporary)
        assert json.loads((temporary / 'package.json').read_text()) == expected
        archive = Path('/tmp/submission_gold_v25_plus.tar.gz')
        assert sha(archive.read_bytes()) == release['package_sha256']
        cleanroom = temporary / 'cleanroom'
        cleanroom.mkdir()
        with tarfile.open(archive, 'r:gz') as tar:
            assert sorted(tar.getnames()) == sorted(expected['files'])
            assert all(member.isfile() for member in tar.getmembers())
            tar.extractall(cleanroom, filter='data')
        for name, digest in expected['files'].items():
            assert sha((cleanroom / name).read_bytes()) == digest
        check = ROOT / 'src/kaggriculture_bonsai/agents/gold_v25_plus/cleanroom_check.py'
        shutil.copyfile(check, cleanroom / 'check.py')
        subprocess.run([str(ROOT / '.venv/bin/python'), '-E', str(cleanroom / 'check.py')],
                       cwd=cleanroom, check=True)
        OUT.mkdir(parents=True, exist_ok=True)
        stage = OUT / 'package'
        for name in expected['files']:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cleanroom / name, destination)
        destination = OUT / 'submission_gold_v25_plus_bkg4.tar.gz'
        shutil.copyfile(archive, destination)
        hashes = [f"archive_sha256  {release['package_sha256']}",
                  f"archive_bytes  {destination.stat().st_size}"]
        hashes.extend(f'{digest}  {name}' for name, digest in sorted(expected['files'].items()))
        (OUT / 'PACKAGE_HASHES.txt').write_text('\n'.join(hashes) + '\n')
        print(destination)
        print('\n'.join(hashes[:2]))


if __name__ == '__main__':
    main()
