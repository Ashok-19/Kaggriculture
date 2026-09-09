"""Kaggle entry point for BKG4_SHOP_ROUTER_0909_PLUS."""
from gold_runtime import GoldRuntime

_RUNTIME = None


def agent(observation, configuration=None):
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = GoldRuntime()
    return _RUNTIME(observation, configuration)
