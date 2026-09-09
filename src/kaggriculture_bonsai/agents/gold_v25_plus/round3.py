"""Fresh BUILD diagnostics after rejecting the frozen second composition."""
import json
from .benchmark import OUT
from .search import run
from .screen import summarize


def main():
    out = OUT / 'round3'
    out.mkdir(exist_ok=True)
    seeds = list(range(990000, 990064))
    candidate = {'id':108, 'start':576, 'end':718, 'mode':9}
    manifest = {'BUILD':seeds, 'INTERNAL_DEV':list(range(1000000,1000064)),
                'official_canary':list(range(1010000,1010032)), 'seats':[0,1],
                'question':'Diagnose fresh BUILD failures of the frozen market-only composition before proposing another repair.'}
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    parent = run('/tmp/gold_plus_cuda',[{'id':0,'start':0,'end':0,'mode':0}],seeds,'round3_parent')
    candidate_rows = run('/tmp/gold_plus_cuda',[candidate],seeds,'round3_diagnostic')
    summary = summarize(parent,candidate_rows)
    (out/'diagnostic.json').write_text(json.dumps({'parent':parent,'candidate':candidate_rows,'summary':summary},separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2))
    print(json.dumps(sorted(candidate_rows,key=lambda r:r['margin'])[:6],indent=2))


if __name__ == '__main__':
    main()
