from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("LatentWAM training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(value, source=0):
    if not dist.is_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def reduce_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.cpu())}


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
