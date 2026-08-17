from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runtime import StatefulV6BKG3Agent

_AGENT = StatefulV6BKG3Agent()

def agent(observation, configuration=None):
    return _AGENT(observation, configuration)
