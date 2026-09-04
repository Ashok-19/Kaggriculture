from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("KAGGRICULTURE_WORKSPACE", PROJECT_ROOT / "workspace"))
CHAMPION_ASSETS = PROJECT_ROOT / "assets" / "champion"
NATIVE_BUILD = WORKSPACE / "build"
