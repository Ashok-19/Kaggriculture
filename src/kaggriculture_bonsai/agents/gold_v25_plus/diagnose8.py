"""Fresh BUILD diagnosis after the official recovery canary failed."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out=OUT/'round8';out.mkdir(exist_ok=True)
    manifest={'BUILD':list(range(1080000,1080128)),'INTERNAL_DEV':list(range(1090000,1090064)),
              'official_canary':list(range(1100000,1100032)),'seats':[0,1]}
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    candidate={'id':700,'start':144,'end':647,'mode':513}
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],manifest['BUILD'],'r8_parent')
    rows=run('/tmp/gold_plus_cuda',[candidate],manifest['BUILD'],'r8_diagnostic')
    summary=summarize(parent,rows)
    (out/'diagnostic.json').write_text(json.dumps({'parent':parent,'improved':rows,'summary':summary},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2));print(json.dumps(sorted(rows,key=lambda r:r['margin'])[:4],indent=2))


if __name__=='__main__':main()
