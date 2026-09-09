"""Standalone entry for the separately identified gold-v25-plus candidate."""
import hashlib
import json
from pathlib import Path

import gold_runtime
from gold_runtime import GoldRuntime

_RUNTIME=None


def agent(observation, configuration=None):
    global _RUNTIME
    if _RUNTIME is None:
        # Kaggle executes main.py via exec(), so __file__ is intentionally absent.
        root=Path(gold_runtime.__file__).resolve().parent
        identity=json.loads((root/'candidate.json').read_text())
        if hashlib.sha256((root/'assets/specialist.bkgc').read_bytes()).hexdigest()!=identity['gate_sha256']:
            raise ValueError('gold-v25-plus specialist hash mismatch')
        _RUNTIME=GoldRuntime()
    return _RUNTIME(observation,configuration)
