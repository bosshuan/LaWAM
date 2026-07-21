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
conda create --name lawam-312 python=3.12 pip --yes
conda activate lawam-312
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

The run scripts use the Python interpreter from the active Conda environment
and fail early when no Conda environment is active. No environment setup script
is required. The final editable-install command is mandatory because this
repository uses a `src/` package layout; without it, commands such as
`python -m latent_wam.preflight` cannot import `latent_wam`.

## 8xA100 bring-up

Run these commands on the debug server from the repository root:

```bash
bash scripts/debug/preflight.sh --skip-checksum
bash scripts/debug/run_checkpoint_audit.sh
bash scripts/debug/run_tests.sh
bash scripts/debug/run_1gpu_smoke.sh
bash scripts/debug/run_8gpu_smoke.sh
bash scripts/debug/run_tiny_overfit.sh
```

After the Conda environment is active, the first five bring-up checks can also be
run sequentially with `bash scripts/debug/bringup_8gpu.sh`.

The smoke config deliberately uses the local hash text encoder so the training
path can be validated without downloading T5. Scientific training uses frozen
T5-large and `local_files_only: true`; either put `google-t5/t5-large` in the
server Hugging Face cache or pass its local directory:

```bash
bash scripts/train_8gpu.sh --text-model /absolute/local/path/to/t5-large
```

Before the scientific run, validate the same local T5 path that training will
use:

```bash
bash scripts/debug/preflight_full.sh --skip-checksum \
  --text-model /absolute/local/path/to/t5-large
```

Checkpoint and data paths can likewise be overridden with `--checkpoint` and
`--data-root`. Persistent server-only overrides may instead be placed in an
ignored config under `configs/local/` and passed with `--config`.

The default full-training command starts Stage 1, uses the configured 384
resolution, and never silently changes the backbone or latent space after OOM:

```bash
bash scripts/train_8gpu.sh \
  --text-model /absolute/local/path/to/t5-large
```

First reduce worker count, keep the already configured activation checkpointing,
or increase gradient accumulation if the canonical model runs out of memory.

The scientific schedule has separate configs for `stage1_future`,
`stage2_action_warmup`, and `stage3_joint`. Initialize a new stage from the
previous stage without importing its optimizer state:

```bash
bash scripts/train_8gpu.sh \
  --config configs/train/stage2_action_warmup.yaml \
  --init-student outputs/interndata_a1_stage1_future/<run>/checkpoints/final.pt \
  --text-model /absolute/local/path/to/t5-large
```

Then initialize Stage 3 from the Stage 2 final checkpoint in the same way using
`configs/train/stage3_joint.yaml`. Calling `scripts/train_8gpu.sh` without an
explicit config never skips directly to joint training.

Use `--resume` only when continuing the same stage and optimizer schedule.

## Dataset support

The first implementation targets the official InternData-A1 LeRobot v2.1
layout (`meta/info.json`, per-episode Parquet, and per-camera MP4). It discovers
subdatasets recursively, chooses the head/main camera, and constructs masked
joint/gripper schemas from each `info.json`. LeRobot v3.0 directories are
excluded explicitly rather than being misread as v2.1.

InternData-A1 is distributed separately under its own CC BY-NC-SA 4.0 terms.
No dataset files are copied into this repository.
