"""Minimal YAML config loader with `${oc.env:VAR,default}` interpolation.

Ported unchanged from merits-l-text / merits-l-llama so configs written for
those repos keep working here.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

_ENV_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


class AttrDict(dict):
    """dict that exposes keys as attributes (recursively)."""

    def __init__(self, mapping: Dict[str, Any] | None = None) -> None:
        super().__init__()
        if mapping:
            for k, v in mapping.items():
                self[k] = _wrap(v)

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def get(self, key, default=None):
        return super().get(key, default)


def _wrap(v: Any):
    if isinstance(v, dict):
        return AttrDict(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _resolve_env(value: str) -> str:
    def _sub(m: re.Match) -> str:
        name = m.group(1).strip()
        default = m.group(2)
        return os.environ.get(name, default if default is not None else "")
    return _ENV_RE.sub(_sub, value)


def _interpolate(obj: Any) -> Any:
    if isinstance(obj, str):
        return _resolve_env(obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(x) for x in obj]
    return obj


def load_config(path: str | Path) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AttrDict(_interpolate(raw))
