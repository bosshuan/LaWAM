from .intern_data_a1 import InternDataA1Dataset, collate_training_batch
from .mixture import DistributedMixtureSampler, LeRobotMixtureDataset
from .schema import ActionLossSpec, ActionSchema, ActionSchemaAdapter


def build_training_dataset(config, split="train"):
    if config.data.roots:
        return LeRobotMixtureDataset(config, split=split)
    return InternDataA1Dataset(config, split=split)

__all__ = [
    "ActionLossSpec",
    "ActionSchema",
    "ActionSchemaAdapter",
    "DistributedMixtureSampler",
    "InternDataA1Dataset",
    "LeRobotMixtureDataset",
    "build_training_dataset",
    "collate_training_batch",
]
