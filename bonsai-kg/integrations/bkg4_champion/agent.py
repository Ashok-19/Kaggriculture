from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runtime import BKG4ChampionAgent

_AGENT = BKG4ChampionAgent()


def agent(observation, configuration=None):
    return _AGENT(observation, configuration)
