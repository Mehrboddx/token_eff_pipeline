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


def install():
    fake_google = pytypes.ModuleType("google")
    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.types = pytypes.SimpleNamespace(Content=Content, Part=Part)
    fake_google.genai = fake_genai
    sys.modules.setdefault("google", fake_google)
    sys.modules.setdefault("google.genai", fake_genai)
