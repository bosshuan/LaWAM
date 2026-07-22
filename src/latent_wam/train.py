from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from latent_wam.checkpoint import (
    load_student_initialization,
    load_training_checkpoint,
    save_training_checkpoint,
)
from latent_wam.config import ExperimentConfig, load_config, resolve_output_root
from latent_wam.data import InternDataA1Dataset, collate_training_batch
from latent_wam.distributed import (
    barrier,
    broadcast_object,
    cleanup,
    init_distributed,
    reduce_metrics,
)
from latent_wam.models import LatentWAM
from latent_wam.objective import JointObjective


def parse_args():
    parser = argparse.ArgumentParser(description="Train LatentWAM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--init-student")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--fixed-sample-index", type=int)
    parser.add_argument("--text-model")
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args) -> ExperimentConfig:
    model = config.model
    data = config.data
    train = config.train
    if args.checkpoint:
        model = dataclasses.replace(model, checkpoint=args.checkpoint)
    if args.text_model:
        model = dataclasses.replace(model, text_model=args.text_model)
    if args.data_root:
        data = dataclasses.replace(data, root=args.data_root)
    if args.fixed_sample_index is not None:
        data = dataclasses.replace(data, fixed_sample_index=args.fixed_sample_index)
    if args.max_steps is not None:
        train = dataclasses.replace(train, max_steps=args.max_steps)
    if args.resume:
        train = dataclasses.replace(train, resume=args.resume)
    if args.init_student:
        train = dataclasses.replace(train, init_student=args.init_student)
    config = dataclasses.replace(config, model=model, data=data, train=train)
    config.validate()
    return config


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(deterministic: bool) -> None:
    """Configure CUDA kernels before distributed/CUDA initialization.

    Scientific training keeps the optimized SDPA kernels and TF32 defaults.
    The controlled resume audit instead uses deterministic cuBLAS and the math
    SDPA backend so two independent processes can be compared bit for bit.
    """
    if deterministic:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError(
                "Deterministic training requires CUBLAS_WORKSPACE_CONFIG to be "
                f":4096:8 or :16:8, got {workspace!r}"
            )
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return

    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def runtime_audit_summary(deterministic: bool) -> dict[str, object]:
    summary: dict[str, object] = {
        "deterministic": deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp": torch.backends.cuda.math_sdp_enabled(),
    }
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        summary["cudnn_sdp"] = torch.backends.cuda.cudnn_sdp_enabled()
    return summary


def create_output_dir(config: ExperimentConfig, requested: str | None, rank: int) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if rank == 0 else None
        timestamp = broadcast_object(timestamp)
        path = resolve_output_root(config) / config.train.run_name / timestamp
    if rank == 0:
        for child in ("logs", "checkpoints", "metrics", "artifacts"):
            (path / child).mkdir(parents=True, exist_ok=True)
        with (path / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(dataclasses.asdict(config), handle, indent=2)
    barrier()
    return path


def configure_stage(student, stage: str):
    if stage not in {"future", "action_warmup", "joint"}:
        raise ValueError(f"Unknown training stage: {stage}")
    student.requires_grad_(True)
    for token in student.predictor.mask_tokens:
        token.requires_grad_(False)
    # The dense-prediction checkpoint includes a context output head and a
    # separate image-modality embedding. LaWAM uses only the future output head
    # and video modality. These strict-loaded parameters are intentionally
    # unused and must stay frozen; otherwise DDP with
    # find_unused_parameters=False fails after the first step.
    for name in ("predictor_proj_context", "img_mod_embed"):
        unused_pretrained_path = getattr(student.predictor, name, None)
        if unused_pretrained_path is not None:
            unused_pretrained_path.requires_grad_(False)
    if stage == "future":
        student.action_queries.requires_grad_(False)
        for module in (
            student.action_time_embedding,
            student.action_condition_projection,
            student.action_norm,
            student.action_head,
        ):
            module.requires_grad_(False)
    elif stage == "action_warmup":
        student.predictor.requires_grad_(False)
        student.future_mask_token.requires_grad_(False)


def build_optimizer(student, config: ExperimentConfig):
    pretrained, new = [], []
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        (pretrained if name.startswith("predictor.") else new).append(parameter)
    groups = []
    if pretrained:
        groups.append({"params": pretrained, "lr": config.train.predictor_lr, "name": "predictor"})
    if new:
        groups.append({"params": new, "lr": config.train.new_module_lr, "name": "new"})
    return torch.optim.AdamW(groups, weight_decay=config.train.weight_decay, betas=(0.9, 0.95))


def build_scheduler(optimizer, config: ExperimentConfig):
    if config.train.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    warmup = max(1, int(config.train.max_steps * config.train.warmup_fraction))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, config.train.max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return config.train.min_lr_ratio + (1.0 - config.train.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    configure_runtime(config.train.deterministic)
    rank, local_rank, world_size, device = init_distributed()
    seed_everything(config.train.seed + rank)
    output_dir = create_output_dir(config, args.output_dir, rank)

    dataset = InternDataA1Dataset(config, split="train")
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.train.seed)
        if world_size > 1
        else RandomSampler(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size_per_gpu,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
        drop_last=True,
        collate_fn=collate_training_batch,
    )
    model = LatentWAM.from_config(config)
    model.encoder.to(device=device, dtype=torch.bfloat16 if config.train.bf16 else torch.float32)
    model.text_encoder.to(device=device, dtype=torch.bfloat16 if config.train.bf16 else torch.float32)
    model.student.to(device=device)
    configure_stage(model.student, config.train.stage)
    if config.train.init_student:
        load_student_initialization(config.train.init_student, model.student)
    optimizer = build_optimizer(model.student, config)
    scheduler = build_scheduler(optimizer, config)
    student = DistributedDataParallel(
        model.student,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=config.train.find_unused_parameters,
        gradient_as_bucket_view=True,
    ) if world_size > 1 else model.student
    objective = JointObjective(config).to(device)
    start_step = 0
    if config.train.resume:
        start_step = load_training_checkpoint(config.train.resume, student, optimizer, scheduler)
    end_step = config.train.max_steps
    if args.stop_after is not None:
        if args.stop_after <= 0 or args.stop_after > config.train.max_steps:
            raise ValueError("--stop-after must be in (0, train.max_steps]")
        end_step = args.stop_after
    if start_step > end_step:
        raise ValueError(
            f"Checkpoint step {start_step} is later than requested stop step {end_step}"
        )

    if rank == 0:
        report = dataclasses.asdict(model.load_report)
        report.update(
            {
                "world_size": world_size,
                "samples": len(dataset),
                "output_dir": str(output_dir),
                "stage": config.train.stage,
                "requested_stop_after": end_step,
                "data": dataset.audit_summary(),
                "global_batch_size": (
                    config.train.batch_size_per_gpu
                    * config.train.grad_accum_steps
                    * world_size
                ),
                "runtime": runtime_audit_summary(config.train.deterministic),
                "text_encoder": {
                    "backend": config.model.text_backend,
                    "model": config.model.text_model,
                    "wrapper_class": type(model.text_encoder).__name__,
                    "encoder_class": type(
                        getattr(model.text_encoder, "encoder", model.text_encoder)
                    ).__name__,
                    "output_dim": model.text_encoder.output_dim,
                    "parameters": sum(
                        parameter.numel()
                        for parameter in model.text_encoder.parameters()
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.text_encoder.parameters()
                        if parameter.requires_grad
                    ),
                },
                "trainable_student_parameters": sum(
                    parameter.numel()
                    for parameter in model.student.parameters()
                    if parameter.requires_grad
                ),
                "optimizer_groups": {
                    group["name"]: sum(parameter.numel() for parameter in group["params"])
                    for group in optimizer.param_groups
                },
            }
        )
        with (output_dir / "artifacts" / "startup.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    optimizer.zero_grad(set_to_none=True)
    data_iterator = iter(loader)
    epoch = 0
    log_path = output_dir / "logs" / "train.jsonl"
    last_saved_step: int | None = None
    try:
        for step in range(start_step, end_step):
            step_started = time.perf_counter()
            data_time = 0.0
            torch.cuda.reset_peak_memory_stats(device)
            metrics_accum: dict[str, float] = {}
            for micro_step in range(config.train.grad_accum_steps):
                data_started = time.perf_counter()
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    epoch += 1
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(epoch)
                    data_iterator = iter(loader)
                    batch = next(data_iterator)
                data_time += time.perf_counter() - data_started
                batch = batch.to(device)
                should_sync = micro_step == config.train.grad_accum_steps - 1
                sync_context = nullcontext() if should_sync or world_size == 1 else student.no_sync()
                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.train.bf16):
                        context_features = model.encoder.encode_context(batch.student.context_rgb)
                        future_targets = (
                            None
                            if config.train.stage == "action_warmup"
                            else model.encoder.encode_target(batch.teacher.full_rgb)
                        )
                        text_features, text_valid = model.encode_text(
                            batch.student.instructions, device
                        )
                        prediction = student(
                            context_features,
                            batch.student,
                            text_features,
                            text_valid,
                        )
                        losses = objective(prediction, future_targets, batch.targets)
                        scaled_loss = losses.total / config.train.grad_accum_steps
                    scaled_loss.backward()
                for key, value in losses.detached().items():
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + value / config.train.grad_accum_steps
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                config.train.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            completed = step + 1
            if completed % config.train.log_every == 0:
                torch.cuda.synchronize(device)
                step_time = time.perf_counter() - step_started
                metrics_accum["grad_norm"] = float(gradient_norm)
                metrics_accum["step"] = float(completed)
                metrics_accum["step_time_sec"] = step_time
                metrics_accum["data_time_sec"] = data_time
                metrics_accum["samples_per_sec"] = (
                    config.train.batch_size_per_gpu
                    * config.train.grad_accum_steps
                    * world_size
                    / max(step_time, 1.0e-9)
                )
                metrics_accum["peak_memory_gib"] = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
                for group in optimizer.param_groups:
                    metrics_accum[f"lr_{group['name']}"] = group["lr"]
                metrics = reduce_metrics(metrics_accum, device)
                if rank == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                    print(json.dumps(metrics, sort_keys=True), flush=True)
            if completed % config.train.save_every == 0 and completed < end_step:
                barrier()
                if rank == 0:
                    save_training_checkpoint(
                        output_dir / "checkpoints" / f"step_{completed:08d}.pt",
                        student,
                        optimizer,
                        scheduler,
                        completed,
                        dataclasses.asdict(config),
                    )
                    last_saved_step = completed
                barrier()
        barrier()
        if rank == 0:
            if end_step == config.train.max_steps:
                save_training_checkpoint(
                    output_dir / "checkpoints" / "final.pt",
                    student,
                    optimizer,
                    scheduler,
                    end_step,
                    dataclasses.asdict(config),
                )
            elif last_saved_step != end_step:
                save_training_checkpoint(
                    output_dir / "checkpoints" / f"step_{end_step:08d}.pt",
                    student,
                    optimizer,
                    scheduler,
                    end_step,
                    dataclasses.asdict(config),
                )
        barrier()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
