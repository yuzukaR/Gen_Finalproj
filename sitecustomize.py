"""Repo-local Python startup hooks used by subprocess-launched training scripts."""

from __future__ import annotations

import os


if os.environ.get("GENFINAL_DISABLE_TORCHAO") == "1":
    import importlib.util
    import importlib.metadata as importlib_metadata

    _orig_find_spec = importlib.util.find_spec
    _orig_version = importlib_metadata.version

    def _patched_find_spec(name: str, package: str | None = None):
        if name == "torchao" or name.startswith("torchao."):
            return None
        return _orig_find_spec(name, package)

    def _patched_version(name: str) -> str:
        if name == "torchao":
            raise importlib_metadata.PackageNotFoundError(name)
        return _orig_version(name)

    importlib.util.find_spec = _patched_find_spec
    importlib_metadata.version = _patched_version
