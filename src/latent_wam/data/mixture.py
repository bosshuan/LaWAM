from __future__ import annotations

import bisect
import dataclasses
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from latent_wam.config import ExperimentConfig
from latent_wam.data.intern_data_a1 import InternDataA1Dataset
from latent_wam.types import TrainingBatch


class LeRobotMixtureDataset(Dataset[TrainingBatch]):
    """Concatenate explicit LeRobot v2.1 roots while retaining source identity."""

    def __init__(self, config: ExperimentConfig, split: str = "train"):
        roots = config.data.roots
        if len(roots) < 2:
            raise ValueError("LeRobotMixtureDataset requires at least two data roots")
        names = config.data.source_names or tuple(Path(root).name for root in roots)
        configured_weights = config.data.mixture_weights or (1.0,) * len(roots)
        weight_sum = sum(configured_weights)
        self.source_names = tuple(names)
        self.mixture_weights = tuple(weight / weight_sum for weight in configured_weights)
        self.sources: list[InternDataA1Dataset] = []
        self.source_offsets: list[int] = []
        self.source_sizes: list[int] = []
        total = 0
        for root in roots:
            source_data = dataclasses.replace(
                config.data,
                root=root,
                roots=(),
                source_names=(),
                mixture_weights=(),
                mixture_epoch_samples=None,
            )
            source_config = dataclasses.replace(config, data=source_data)
            dataset = InternDataA1Dataset(source_config, split=split)
            self.sources.append(dataset)
            self.source_offsets.append(total)
            self.source_sizes.append(len(dataset))
            total += len(dataset)
        self._cumulative = [
            offset + size
            for offset, size in zip(self.source_offsets, self.source_sizes)
        ]

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, index: int) -> TrainingBatch:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect.bisect_right(self._cumulative, index)
        local_index = index - self.source_offsets[source_index]
        sample = self.sources[source_index][local_index]
        source_name = self.source_names[source_index]
        for metadata in sample.targets.metadata:
            metadata["dataset_source"] = source_name
        return sample

    def audit_summary(self) -> dict[str, Any]:
        return {
            "backend": "lerobot_v21_mixture",
            "source_count": len(self.sources),
            "raw_samples": len(self),
            "effective_samples": len(self),
            "mixture_weights": {
                name: weight
                for name, weight in zip(self.source_names, self.mixture_weights)
            },
            "sources": [
                {
                    "name": name,
                    "root": str(dataset.root),
                    "samples": len(dataset),
                    "sampling_weight": weight,
                    "audit": dataset.audit_summary(),
                }
                for name, weight, dataset in zip(
                    self.source_names, self.mixture_weights, self.sources
                )
            ],
        }


class DistributedMixtureSampler(Sampler[int]):
    """Deterministic source-weighted sampling shared across distributed ranks."""

    def __init__(
        self,
        dataset: LeRobotMixtureDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        epoch_samples: int | None,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        requested = len(dataset) if epoch_samples is None else epoch_samples
        self.num_samples = math.ceil(requested / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.requested_epoch_samples = requested

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        weights = torch.tensor(self.dataset.mixture_weights, dtype=torch.double)
        source_draws = torch.multinomial(
            weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        global_indices = torch.empty(self.total_size, dtype=torch.long)
        for source_index, (offset, size) in enumerate(
            zip(self.dataset.source_offsets, self.dataset.source_sizes)
        ):
            positions = torch.nonzero(source_draws == source_index, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            local_indices = torch.randint(
                size,
                (positions.numel(),),
                generator=generator,
            )
            global_indices[positions] = local_indices + offset
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices.tolist())

    def audit_summary(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "replacement": True,
            "seed": self.seed,
            "requested_epoch_samples": self.requested_epoch_samples,
            "padded_epoch_samples": self.total_size,
            "samples_per_rank": self.num_samples,
            "source_probabilities": {
                name: weight
                for name, weight in zip(
                    self.dataset.source_names, self.dataset.mixture_weights
                )
            },
        }
