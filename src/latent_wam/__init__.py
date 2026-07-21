"""LatentWAM package."""

from .config import ExperimentConfig, load_config
from .types import JointPrediction, StudentInputs, TeacherInputs, TrainingTargets

__all__ = [
    "ExperimentConfig",
    "JointPrediction",
    "StudentInputs",
    "TeacherInputs",
    "TrainingTargets",
    "load_config",
]
