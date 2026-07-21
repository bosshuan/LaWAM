import inspect

from latent_wam.models.latent_wam import LatentWAM
from latent_wam.types import StudentInputs


def test_predict_cannot_accept_teacher_target():
    signature = inspect.signature(LatentWAM.predict)
    assert list(signature.parameters) == ["self", "inputs"]
    assert signature.parameters["inputs"].annotation in {"StudentInputs", StudentInputs}
