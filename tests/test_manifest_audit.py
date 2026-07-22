from pathlib import Path

import pytest

from latent_wam.manifest_audit import remap_path


def test_remap_h800_training_path_to_storage_view():
    mapped = remap_path(
        "/opt/huawei/dataset/data_w/processed/OXE",
        "/opt/huawei",
        "/home/ma-user/work",
    )
    assert mapped == Path("/home/ma-user/work/dataset/data_w/processed/OXE")


def test_remap_exact_prefix():
    assert remap_path("/opt/huawei", "/opt/huawei", "/home/ma-user/work") == Path(
        "/home/ma-user/work"
    )


def test_remap_rejects_unrelated_path():
    with pytest.raises(ValueError, match="not below configured prefix"):
        remap_path("/opt/huawei-extra/dataset", "/opt/huawei", "/home/ma-user/work")
