import importlib.util, pathlib, os
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('meta_gold_v23_base', HERE/'meta_gold_v10.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
rg=base.base.rg
rg.EVAC_PASS_HORIZON=7
v8=base.base
_orig_use=v8._use_secondary
START=int(os.environ.get('LATE_START','624'))
def _use_secondary(obs,seat):
    if int(obs.get('step',0) or 0)>=START:return True
    return _orig_use(obs,seat)
v8._use_secondary=_use_secondary
agent=base.agent
