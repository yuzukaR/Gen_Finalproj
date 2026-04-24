import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


@contextmanager
def track_run(log_path: str | Path, extra: dict[str, Any] | None = None):
    """Wrap a training/inference call to capture wallclock + peak VRAM into a JSON file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        yield
    finally:
        elapsed_s = time.time() - start
        peak_bytes = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
        record = {
            "elapsed_s": elapsed_s,
            "elapsed_min": elapsed_s / 60.0,
            "peak_vram_gb": peak_bytes / (1024**3),
            **(extra or {}),
        }
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2)
