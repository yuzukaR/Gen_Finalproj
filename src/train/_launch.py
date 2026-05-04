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


def resolve_diffusers_example_script(env_var: str, relative_path: str) -> str:
    configured = os.environ.get(env_var)
    if configured:
        script = Path(configured).expanduser()
        if script.is_file():
            return str(script)
        raise SystemExit(f"{env_var} is set but does not exist: {script}")

    project_root = Path(__file__).resolve().parents[2]
    candidates = []

    diffusers_dir = os.environ.get("DIFFUSERS_DIR")
    if diffusers_dir:
        candidates.append(Path(diffusers_dir).expanduser() / relative_path)

    candidates.extend([
        Path("/content/diffusers") / relative_path,
        project_root / "diffusers" / relative_path,
        project_root.parent / "diffusers" / relative_path,
        Path.cwd() / "diffusers" / relative_path,
    ])

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return str(candidate)

    raise SystemExit(
        f"Set {env_var} to the path of diffusers/{relative_path}, "
        "or clone diffusers to /content/diffusers in Colab."
    )
