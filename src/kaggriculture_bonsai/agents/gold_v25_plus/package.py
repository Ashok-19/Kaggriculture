"""Build an isolated standalone candidate package; never touch parent packaging."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from .benchmark import OUT
from kaggriculture_bonsai.agents.gold_v25.extract import ROOT, HERE, ASSETS

PACKAGE=Path('/tmp/gold_v25_plus_package')


def main(evidence=OUT):
    (PACKAGE/'assets').mkdir(parents=True,exist_ok=True)
    for name in ('controller.bkgc','routes.bkv6','arity.bin','identity.json'):
        shutil.copyfile(ASSETS/name,PACKAGE/'assets'/name)
    shutil.copyfile(evidence/'specialist.bkgc',PACKAGE/'assets/specialist.bkgc')
    shutil.copyfile(evidence/'composition.json',PACKAGE/'candidate.json')
    shutil.copyfile(HERE/'submission/gold_runtime.py',PACKAGE/'gold_runtime.py')
    source=ROOT/'src/kaggriculture_bonsai/agents/gold_v25_plus'
    shutil.copyfile(source/'submission_main.py',PACKAGE/'main.py')
    identity=json.loads((evidence/'composition.json').read_text())
    (PACKAGE/'PROVENANCE.md').write_text((HERE/'PROVENANCE.md').read_text()+'\n\nThis separate ReCurSiON candidate adds BKG4-gated market-order specialists. '+json.dumps(identity['candidate'])+'. Mode 5 liquidates projected products, sorted by descending current value. Mode 9 uses only all-SELL/no-op stable order priority before step708, then mode5 through 718. Parent unit actions, parent state updates and all four donor tapes are unchanged.\n')
    # Preserve the entire inherited notice, not an abbreviated license excerpt.
    text=(HERE/'overlay.py').read_text();license_block=text.split('# Mechanically extracted',1)[0]
    (PACKAGE/'LICENSE').write_text(license_block)
    with (PACKAGE/'PROVENANCE.md').open('a') as provenance:
        provenance.write('Mode 17 sells available non-input products on all-SELL/no-op turns before step708, then uses mode5. Wheat and fertilizer orders remain unchanged before the endgame phase.\n')
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-Inative/bkg4/include','-Inative/bkg4/stateful',str(source/'bridge.cpp'),'-o',str(PACKAGE/'gold_cpu.so')],cwd=ROOT,check=True)
    names=['main.py','gold_runtime.py','gold_cpu.so','LICENSE','PROVENANCE.md','candidate.json']+['assets/'+n for n in ('controller.bkgc','routes.bkv6','arity.bin','identity.json','specialist.bkgc')]
    hashes={name:hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest() for name in sorted(names)}
    (evidence/'package.json').write_text(json.dumps({'candidate':identity['identity'],'files':hashes},indent=2)+'\n')
    with open('/tmp/submission_gold_v25_plus.tar.gz','wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:
            with tarfile.open(fileobj=z,mode='w') as archive:
                for name in sorted(hashes):
                    path=PACKAGE/name;info=archive.gettarinfo(str(path),arcname=name);info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
                    with path.open('rb') as data:archive.addfile(info,data)
    print(PACKAGE)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=OUT);args=parser.parse_args();main(args.evidence)
