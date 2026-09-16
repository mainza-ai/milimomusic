"""Restricted loading of tensor weights from historical YourMT3 checkpoints."""

import sys
import types
from contextlib import contextmanager
import torch


class _LegacyMetadata:
    """Inert holder for discarded tokenizer metadata."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


@contextmanager
def _scoped_module_mocker():
    """Temporarily installs dummy modules for legacy PyTorch Lightning checkpoint metadata."""
    modules_to_mock = [
        "utils",
        "utils.task_manager",
        "utils.tokenizer",
        "utils.note_event_dataclasses",
        "utils.event_codec",
    ]
    original_modules = {}
    created = []

    for mod_name in modules_to_mock:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
        else:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
            created.append(mod_name)

    sys.modules["utils.task_manager"].TaskManager = _LegacyMetadata
    sys.modules["utils.tokenizer"].NoteEventTokenizer = _LegacyMetadata
    sys.modules["utils.note_event_dataclasses"].EventRange = _LegacyMetadata
    sys.modules["utils.note_event_dataclasses"].Event = _LegacyMetadata
    sys.modules["utils.event_codec"].FastCodec = _LegacyMetadata

    try:
        yield
    finally:
        for mod_name in created:
            sys.modules.pop(mod_name, None)
        for mod_name, mod in original_modules.items():
            sys.modules[mod_name] = mod


def load_state_dict(path):
    """Read tensor weights without importing historical checkpoint classes.

    Known tokenizer metadata is mapped to inert holders and discarded.
    """
    with _scoped_module_mocker():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError("YourMT3 checkpoint must contain a state dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("YourMT3 checkpoint does not contain a state dictionary")

    return {
        str(key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and not str(key).startswith("pitchshift.")
    }

