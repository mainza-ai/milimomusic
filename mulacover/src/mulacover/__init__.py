"""MuLaCover: controllable cover-song generation from audio or MIDI."""

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "MuLaCoverConfig": (".configuration", "MuLaCoverConfig"),
    "MuLaCover": (".modeling", "MuLaCover"),
    "MuLaCoverGenConfig": (".pipeline", "MuLaCoverGenConfig"),
    "MuLaCoverGenPipeline": (".pipeline", "MuLaCoverGenPipeline"),
    "SymbolicCondition": (".symbolic", "SymbolicCondition"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_rel, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(f"mulacover{mod_rel}")
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'mulacover' has no attribute '{name}'")


__all__ = [
    "MuLaCover",
    "MuLaCoverConfig",
    "MuLaCoverGenConfig",
    "MuLaCoverGenPipeline",
    "SymbolicCondition",
]
