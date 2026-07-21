from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _student_state(student) -> dict[str, torch.Tensor]:
    module = student.module if hasattr(student, "module") else student
    return module.state_dict()


def save_training_checkpoint(
    path: Path,
    student,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "student": _student_state(student),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        temporary,
    )
    temporary.replace(path)


def load_training_checkpoint(
    path: str | Path,
    student,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    module = student.module if hasattr(student, "module") else student
    module.load_state_dict(checkpoint["student"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    return int(checkpoint["step"])


def load_student_initialization(path: str | Path, student) -> None:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("student", checkpoint)
    student.load_state_dict(state, strict=True)
