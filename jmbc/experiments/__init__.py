"""Experiment driver registry."""
from .common import run_single
from . import rbc, ks, general

_REGISTRY = {
    "rbc": rbc.run,
    "ks": ks.run,
    "general": general.run,
}


def get_driver(name: str):
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown experiment {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = ["run_single", "get_driver"]
