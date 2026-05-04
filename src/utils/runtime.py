"""Runtime compatibility helpers for transient notebook environments."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.util
import os

_TORCHAO_PATCHED = False


def disable_incompatible_torchao() -> None:
    """Hide incompatible torchao installs from PEFT/diffusers.

    Colab images sometimes preinstall `torchao` versions that are too old for
    the PEFT release required by current diffusers main. This project does not
    depend on torchao, so it is safe to make those installs invisible.
    """

    global _TORCHAO_PATCHED
    if _TORCHAO_PATCHED:
        return

    os.environ["GENFINAL_DISABLE_TORCHAO"] = "1"

    orig_find_spec = importlib.util.find_spec
    orig_version = importlib_metadata.version

    def patched_find_spec(name: str, package: str | None = None):
        if name == "torchao" or name.startswith("torchao."):
            return None
        return orig_find_spec(name, package)

    def patched_version(name: str) -> str:
        if name == "torchao":
            raise importlib_metadata.PackageNotFoundError(name)
        return orig_version(name)

    importlib.util.find_spec = patched_find_spec
    importlib_metadata.version = patched_version
    _TORCHAO_PATCHED = True
