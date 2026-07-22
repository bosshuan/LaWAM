import json
from pathlib import Path

import pytest

from latent_wam.preflight import json_default


def test_preflight_report_serializes_transformers_sets():
    report = {
        "missing_keys": {"encoder.block.1", "encoder.block.0"},
        "path": Path("/tmp/t5-large"),
    }
    encoded = json.dumps(report, default=json_default)
    decoded = json.loads(encoded)
    assert decoded["missing_keys"] == ["encoder.block.0", "encoder.block.1"]
    assert decoded["path"] == "/tmp/t5-large"


def test_preflight_json_default_rejects_unknown_objects():
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())
