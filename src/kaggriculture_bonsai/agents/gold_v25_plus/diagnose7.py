"""New BUILD cases after the frozen composition failed INTERNAL_DEV."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out=OUT/'round7';out.mkdir(exist_ok=True)
    manifest={'BUILD':list(range(1050000,1050128)),
              'INTERNAL_DEV':list(range(1060000,1060064)),
              'official_canary':list(range(1070000,1070032)),'seats':[0,1]}
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    parent=run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],manifest['BUILD'],'new_parent')
    rows=run('/tmp/gold_plus_cuda',[{'id':600,'start':144,'end':647,'mode':257}],manifest['BUILD'],'new_repair')
    (out/'diagnostic.json').write_text(json.dumps({'parent':parent,'improved':rows,'summary':summarize(parent,rows)},separators=(',',':'))+'\n')
    print(json.dumps(summarize(parent,rows),indent=2))
    print(json.dumps(sorted(rows,key=lambda r:r['margin'])[:4],indent=2))


if __name__=='__main__':main()
