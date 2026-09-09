"""Own-schedule anticipation without crossing the next demand cycle."""
import json
from .benchmark import OUT
from .search3 import main as search


def main():
    out=OUT/'round4';out.mkdir(exist_ok=True)
    manifest=json.loads((OUT/'round3/panels.json').read_text())
    manifest['question']='Anticipate own scheduled non-input sales by two or three steps, without crossing the next four-step consumption boundary or shop unlock.'
    manifest['BUILD_source']='round3/panels.json'
    manifest['INTERNAL_DEV_previously_opened']=False
    (out/'panels.json').write_text(json.dumps(manifest,indent=2)+'\n')
    candidates=[{'id':300+i,'start':start,'end':718,'mode':mode}
                for i,(start,mode) in enumerate((s,m) for s in (0,288,432,576) for m in (33,65))]
    search(out,candidates,OUT/'round3/diagnostic.json',manifest['question'],'BKG4_GOLD_V25_PLUS_LEAD_V4')


if __name__=='__main__':main()
