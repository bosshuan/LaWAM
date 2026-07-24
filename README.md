# LatentWAM

LatentWAM turns the paired V-JEPA 2.1 ViT-G encoder and predictor into one
joint semantic-future/action predictor. It does not render pixels and does not
attach a diffusion, flow, or standalone action expert.

The repository is self-contained. The minimal V-JEPA 2.1 model implementation
is vendored under `src/latent_wam/vendor/vjepa21` with its original MIT license;
the sibling `vjepa2` checkout is not needed at install or run time.

## Current architecture

- Frozen V-JEPA 2.1 ViT-G/16 2B (`vit_gigantic_xformers`: 48 blocks, width 1664).
- Paired 24-block, width-384 predictor loaded from the same local checkpoint.
- Blocks 0-11 predict masked future semantic tokens using the native visual path.
- Blocks 12-17 insert ten action queries. Actions read predicted future, never raw
  context tokens; future cannot yet read actions.
- Blocks 18-23 use time-aligned reciprocal attention between predicted future
  intervals and action queries.
- Four zero-gated conditioning adapters inject instruction, proprioception,
  past actions, embodiment, and schema information after blocks 4, 11, 17, 23.
- The predictor emits four 1664-dimensional future levels and a ten-step direct
  action chunk in one pass.

Teacher future frames are accepted only by `JointObjective` through the frozen
encoder. `LatentWAM.predict(StudentInputs)` has no target argument.

## Server paths

The checked-in configs use:

```text
/mnt/sfs_turbo/fyy/checkpoints/vjepa2/vjepa2_1_vitG_384.pt
/mnt/sfs_turbo/rl/InternData-A1/sim
```

No shell `export`, `PYTHONPATH`, `VJEPA2_ROOT`, network checkpoint loader, or
runtime import from another repository is used. Alternate paths can be passed
with `--checkpoint`, `--data-root`, and `--text-model`.

Outputs are created automatically below:

```text
outputs/<run-name>/<timestamp>/
  artifacts/
  checkpoints/
  logs/
  metrics/
```

## Conda environment

Create and activate the environment manually from the repository root:

```bash
conda create --name vjepa2-312 python=3.12 pip --yes
conda activate vjepa2-312
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

The run scripts use the Python interpreter from the active Conda environment
and fail early when no Conda environment is active. No environment setup script
is required. The final editable-install command is mandatory because this
repository uses a `src/` package layout; without it, commands such as
`python -m latent_wam.preflight` cannot import `latent_wam`.

## Completed 8xA100 validation

The A100 checkpoint audit, smoke tests, strict overfit checks, deterministic
resume audit, real T5-large smoke test, and 100-step full-`sim` Stage 1 pilot
have already completed. Their reports and logs remain under `outputs/` as
reproducibility evidence.

Do not run `scripts/train_8gpu.sh` for the current training workflow. It and the
old `configs/debug/` and `configs/train/` files are retained only to reproduce
the completed A100 validation. They use the historical `/mnt/sfs_turbo` paths
and the old InternData-A1 `sim` subset, not the current H800 data mixture.

Current work starts with the CPU storage-manifest audit below and then uses the
external 4x8 H800 launcher. Stage transitions will receive H800-specific
configs only after the multi-source engineering pilot passes.

## 4x8 H800 multi-source bring-up

Before allocating H800s, audit all five candidate datasets from a CPU node that
can see the persistent storage paths. They occupy six explicit roots because
InternData-A1 uses `real` and `sim_updated` as separate sub-sources and excludes
top-level `sim`. RoboMind is included in manifest validation while its
`stats_gr00t.json` metadata is audited. `storage-manifest-004` verified all 13
RoboMind subdatasets, and the next manifest run validates the resulting strict
adapter end to end. Do not start training until the complete multi-source
report passes. RoboTwin reads only its `Randomized` subdirectory. The training
config keeps its `/opt/huawei` paths; the audit maps that prefix to
`/home/ma-user/work` without editing the YAML and reads metadata only (no video
decoding or model-weight loading):

```bash
export LAWAM_RUN_ID=storage-manifest-006
bash /home/ma-user/work/dataset/d_env_wulan/LaWAM/scripts/h800/audit_storage_manifest.sh
```

The audit always writes two reports even if strict validation returns exit code
1: a server-local detailed report at
`outputs/preflight/storage_manifest/<run-id>.full.json` and a GitHub-sized
report at `outputs/preflight/storage_manifest/<run-id>.json`. The compact report
retains complete OXE and RoboMind sidecars while summarizing sidecars from the
other sources. Sync only the compact report. A nonzero exit means one or more
dataset schemas require a loader adapter or metadata correction; do not start
the 32-GPU pilot yet. This storage-view script deliberately bypasses
`conda activate` and invokes `vjepa2-312/bin/python3.12` directly, because the
copied Conda launcher retains an absolute `/opt/huawei` interpreter shebang.

To compact a legacy oversized report without rerunning the dataset audit:

```bash
export LAWAM_RUN_ID=storage-manifest-005
bash /home/ma-user/work/dataset/d_env_wulan/LaWAM/scripts/h800/compact_storage_manifest.sh
```

This reads `<run-id>.json` and writes `<run-id>-compact.json`, leaving the
original report unchanged.
For OXE adapter design, the report captures complete `stats_gr00t.json` and
`stats_delta_state.json` sidecars plus the first non-empty record from every
`episodes_stats.jsonl`; first-record capture is diagnostic only, while approved
episode normalization still parses and aggregates every record by frame count.

The ModelArts launcher is `scripts/h800/launch_4node.sh`. It consumes the
platform-provided `VC_WORKER_HOSTS`, `VC_TASK_INDEX`, `MA_NUM_HOSTS`, and
`MA_NUM_GPUS` variables and uses multi-node `torchrun` without `--standalone`.
Its server defaults are the actual shared checkout and offline model paths:

```bash
export LAWAM_REPO_ROOT=/opt/huawei/dataset/d_env_wulan/LaWAM
export LAWAM_CHECKPOINT=/opt/huawei/dataset/d_env_wulan/vjepa2/checkpoints/vjepa2_1_vitG_384.pt
export LAWAM_TEXT_MODEL=/opt/huawei/dataset/d_env_wulan/text/t5-large
export LAWAM_RUN_ID=manifest-001
export LAWAM_MODE=preflight
bash /opt/huawei/dataset/d_env_wulan/LaWAM/scripts/h800/launch_4node.sh
```

Those three path exports are optional unless the server layout changes, since
the launcher and H800 pilot config contain the same defaults. The corresponding
persistent-storage prefix is `/home/ma-user/work`; only CPU/read-only jobs
should use that view. Training jobs must continue to use `/opt/huawei`.

The scheduler should use this as its external job script; do not launch the
four-node job manually from an interactive terminal. Preflight runs once on
each node and writes both hardware/data/T5 and strict V-JEPA checkpoint reports
under `outputs/preflight/h800_multisource/<run-id>/`. Review all four node
reports before changing `LAWAM_MODE` to `pilot`.

The pilot config declares OXE, AgiBot-World, InternData-A1, RoboMind, and
RoboTwin Randomized as five equally weighted datasets across six explicit roots.
InternData-A1 `real` and `sim_updated` each receive half of InternData's pilot
weight; the top-level `sim` root receives none. Historical `sim` content nested
inside `sim_updated` remains valid. These weights are only for engineering
validation. RoboMind uses the explicitly configured
`robomind_joint_vector` adapter and falls back to its audited
`meta/stats_gr00t.json` normalization. OXE uses an explicit mixed-control
adapter: full `episodes_stats.jsonl` aggregation remains preferred for its 12
covered subdatasets, while the other 19 use audited `stats_gr00t.json`.
`stats_delta_state.json` is diagnostic metadata for derived Cartesian
representations and is not used to normalize the raw Parquet fields. OXE is
restricted to the future-only stage until SO(3) geodesic action loss is
implemented. Other opaque action vectors remain unsupported unless they
receive their own audited adapter.
Sampling first selects a source and then a sample within that source, so large
sources cannot silently dominate. The 32-GPU pilot keeps global batch 64 with
gradient accumulation 2 and logs the observed fraction from every source.
Final scientific weights must be selected after the strict manifest reports
have been reviewed.

## Dataset support

The first implementation targets the official InternData-A1 LeRobot v2.1
layout (`meta/info.json`, per-episode Parquet, and per-camera MP4). It discovers
subdatasets recursively, chooses the head/main camera, and constructs masked
joint/gripper schemas from each `info.json`. Normalization uses `meta/stats.json`
when present; the original `sim` release is supported by frame-weighted
aggregation of its LeRobot v2.1 `meta/episodes_stats.jsonl`. Audited datasets
such as RoboMind may use `meta/stats_gr00t.json` as a final fallback. LeRobot v3.0
directories are excluded explicitly rather than being misread as v2.1. That
`sim` support is retained only for reproducing completed A100 bring-up checks;
the H800 pilot and formal mixture select only the top-level `real` and
`sim_updated` roots. Nested historical content inside `sim_updated` is allowed.

InternData-A1 is distributed separately under its own CC BY-NC-SA 4.0 terms.
No dataset files are copied into this repository.
