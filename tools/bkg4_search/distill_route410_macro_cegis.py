from __future__ import annotations

"""Compatibility entry point for the accepted P0.3 route410 distiller.

P0.3 briefly explored an older 24-turn wheat-scarcity/sheep-cycle variant under
this filename. That design is rejected. Keeping this entry point avoids leaving a
second implementation in local tooling: every import and execution delegates to
the canonical 18-turn `route410_wheat_liquidity_expansion_v1` distiller.
"""

from tools.bkg4_search.distill_route410_wheat_liquidity_expansion import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
