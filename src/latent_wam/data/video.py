from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_selected_frames(
    path: str | Path,
    frame_indices: list[int],
    nominal_fps: float,
    decode_threads: int = 1,
) -> np.ndarray:
    """Decode nearest frames without loading a whole episode into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    desired = sorted(set(int(index) for index in frame_indices))
    if not desired:
        raise ValueError("At least one frame index is required")
    targets = {index: index / nominal_fps for index in desired}
    selected: dict[int, tuple[float, np.ndarray]] = {}
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = decode_threads
        start = max(0.0, targets[desired[0]] - 1.0)
        if stream.time_base is not None:
            container.seek(
                int(start / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        stop = targets[desired[-1]] + 1.0 / nominal_fps
        for frame in container.decode(stream):
            if frame.pts is None or stream.time_base is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < start - 1.0:
                continue
            for index, target in targets.items():
                distance = abs(timestamp - target)
                if distance <= 0.55 / nominal_fps:
                    previous = selected.get(index)
                    if previous is None or distance < previous[0]:
                        selected[index] = (distance, frame.to_ndarray(format="rgb24"))
            if timestamp > stop and len(selected) == len(desired):
                break
    missing = [index for index in desired if index not in selected]
    if missing:
        raise RuntimeError(f"Failed to decode frames {missing} from {path}")
    lookup = {index: selected[index][1] for index in desired}
    return np.stack([lookup[int(index)] for index in frame_indices], axis=0)


def preprocess_vjepa_clip(frames: np.ndarray, resolution: int) -> torch.Tensor:
    """Deterministic resize/center-crop with V-JEPA ImageNet normalization."""
    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    height, width = clip.shape[-2:]
    scale = resolution / min(height, width)
    resized = (max(resolution, round(height * scale)), max(resolution, round(width * scale)))
    clip = tvf.resize(clip, list(resized), InterpolationMode.BICUBIC, antialias=True)
    clip = tvf.center_crop(clip, [resolution, resolution])
    mean = torch.tensor(IMAGENET_MEAN, dtype=clip.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=clip.dtype).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    return clip.permute(1, 0, 2, 3).contiguous()
