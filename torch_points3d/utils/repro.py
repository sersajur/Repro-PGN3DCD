"""Reproducibility helpers: deterministic seeding and environment stamping."""
import logging
import random
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_deterministic(seed: int = 42) -> None:
    """Fix all known sources of randomness for repro-grade runs.

    Note: CPU↔GPU bit-exactness is NOT guaranteed even with this — only
    GPU↔GPU and CPU↔CPU stability across runs.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log.info(f"Determinism enabled (seed={seed}, cudnn.benchmark=False, tf32=False)")


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception as e:
        return f"<failed: {e}>"


def collect_repro_stamp(seed: int) -> Dict[str, Any]:
    """Snapshot of code revision, environment, hardware, and seed for the current run."""
    return {
        "seed": seed,
        "git_head": _run("git rev-parse HEAD"),
        "git_dirty": _run("git status --porcelain"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": _run("nvidia-smi -L"),
        "pip_freeze": _run("pip freeze"),
    }


def log_repro_stamp(stamp: Dict[str, Any]) -> None:
    """Print the reproducibility stamp to the configured logger."""
    log.info(f"seed: {stamp['seed']}")
    log.info(f"git HEAD: {stamp['git_head']}")
    if stamp.get("git_dirty"):
        log.warning(f"git working tree is dirty:\n{stamp['git_dirty']}")
    log.info(f"Python: {stamp['python']}, torch: {stamp['torch']}, numpy: {stamp['numpy']}")
    log.info(f"GPUs:\n{stamp['gpus']}")
    log.info(f"pip freeze (truncated to first 30 lines):\n" +
             "\n".join(stamp.get("pip_freeze", "").splitlines()[:30]))