"""Helpers for launching upstream training scripts safely from this repo."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


def build_subprocess_env(*, disable_torchao: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(project_root)
        if not existing_pythonpath
        else f"{project_root}{os.pathsep}{existing_pythonpath}"
    )
    if disable_torchao:
        env["GENFINAL_DISABLE_TORCHAO"] = "1"
    return env


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True
