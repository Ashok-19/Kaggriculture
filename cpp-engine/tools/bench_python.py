import contextlib
import io
import time
from importlib.metadata import version

EXPECTED_VERSION = "1.32.7"
if version("kaggle-environments") != EXPECTED_VERSION:
    raise RuntimeError(f"benchmark requires kaggle-environments {EXPECTED_VERSION}")

from kaggle_environments import make


def noop(obs, config=None):
    return {"farmer": ["PASS"], "hands": [], "market": []}


with contextlib.redirect_stdout(io.StringIO()):
    make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1}, debug=False).run([noop, noop])

N = 8
t0 = time.perf_counter()
for i in range(N):
    with contextlib.redirect_stdout(io.StringIO()):
        make("kaggriculture", configuration={"episodeSteps": 720, "seed": i}, debug=False).run([noop, noop])
dt = time.perf_counter() - t0
print(f"{N} episodes in {dt:.3f}s = {N/dt:.3f} eps/sec ({1000*dt/N:.1f} ms/episode)")
