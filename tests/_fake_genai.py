"""Minimal stand-in for google.genai.types, so core/ can be imported and
tested without the real SDK or any credentials."""

import sys
import types as pytypes


class Content:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class Part:
    def __init__(self, text):
        self.text = text

    @staticmethod
    def from_text(text):
        return Part(text)


class _Namespace(pytypes.SimpleNamespace):
    """Any other name (genai.Client, types.GenerateContentConfig, ...) is
    only referenced in type hints by the modules under test."""

    def __getattr__(self, name):
        return _stand_in(name)


def _stand_in(name):
    # Dunder lookups (__file__, __path__, ...) must fail as on a real module:
    # libraries imported later, e.g. torch, probe them on everything in sys.modules.
    if name.startswith("__"):
        raise AttributeError(name)
    return type(name, (), {})


def install():
    fake_google = pytypes.ModuleType("google")
    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.types = _Namespace(Content=Content, Part=Part)
    fake_genai.__getattr__ = _stand_in
    fake_google.genai = fake_genai
    sys.modules.setdefault("google", fake_google)
    sys.modules.setdefault("google.genai", fake_genai)
