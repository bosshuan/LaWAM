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

## 8xA100 bring-up

Run these commands on the debug server from the repository root:

```bash
bash scripts/debug/preflight.sh --skip-checksum
bash scripts/debug/run_checkpoint_audit.sh
bash scripts/debug/run_tests.sh
bash scripts/debug/run_1gpu_smoke.sh
bash scripts/debug/run_8gpu_smoke.sh
```

After the Conda environment is active, the first five bring-up checks can also be
run sequentially with `bash scripts/debug/bringup_8gpu.sh`.

After the basic smoke tests, run the strict correctness suite on one A100. It
uses exactly one fixed anchor and constant learning rates, and independently
checks future-only, action-only, and joint optimization:

```bash
bash scripts/debug/run_strict_overfit.sh
```

Then compare an uninterrupted six-step run against a three-step run resumed
from checkpoint. The audit fails unless every student tensor matches exactly:

```bash
bash scripts/debug/run_resume_audit.sh
```

This audit intentionally uses a slower deterministic runtime: TF32, Flash
SDPA, memory-efficient SDPA, and cuDNN SDPA are disabled, while deterministic
cuBLAS and math SDPA are enabled. It first writes `pre_resume_audit.json` after
comparing both independent runs at step 3; only when that passes does it resume
the interrupted run and write the final `resume_audit.json` at step 6. These
settings are scoped to the audit config and do not change scientific training.

Once both checks pass, validate the real frozen T5-large path and run a short
8xA100 T5 smoke test:

```bash
bash scripts/debug/preflight_t5_a100.sh --skip-checksum
bash scripts/debug/run_8gpu_t5_smoke.sh
```

The T5 preflight loads the local tokenizer, validates the T5-large architecture
(`d_model=1024`, 24 encoder layers), and loads every T5 encoder weight on CPU.
Its JSON report is written under `outputs/preflight/`. The smoke launcher prints
and uses one explicit `outputs/interndata_a1_8gpu_t5_smoke/<run-id>/` directory;
the startup artifact records the actual text encoder class, width, parameter
count, and confirms that it is frozen.

After the real-T5 smoke test passes, run a 100-step Stage 1 engineering pilot
over every usable episode in all InternData-A1 `sim` subdatasets:

```bash
bash scripts/debug/run_stage1_full_sim_pilot.sh
```

Unlike the smoke configs, this pilot has no `max_subdatasets` or
`max_episodes_per_subdataset` limit. It uses the scientific microbatch and
gradient accumulation settings (global batch 64 on 8 GPUs), but writes to the
separate `outputs/interndata_a1_stage1_full_sim_pilot/<run-id>/` namespace and
stops after 100 optimizer steps. It does not replace the later 20k-step Stage 1
run on the final training mixture.

Pass `--text-model /absolute/local/path/to/t5-large` to both T5 commands when
the server path differs from the checked-in default.

The basic smoke config deliberately uses the local hash text encoder so the training
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

## 4x8 H800 multi-source bring-up

Before allocating H800s, audit all five manifests from a CPU node that can see
the persistent storage paths. The training config keeps its `/opt/huawei`
paths; the audit maps that prefix to `/home/ma-user/work` without editing the
YAML and reads metadata only (no video decoding or model-weight loading):

```bash
export LAWAM_RUN_ID=storage-manifest-001
bash /home/ma-user/work/dataset/d_env_wulan/LaWAM/scripts/h800/audit_storage_manifest.sh
```

The JSON is written to
`outputs/preflight/storage_manifest/<run-id>.json` even if strict validation
returns exit code 1. A nonzero exit means one or more dataset schemas require a
loader adapter or metadata correction; sync that report back for review and do
not start the 32-GPU pilot yet. This storage-view script deliberately bypasses
`conda activate` and invokes `vjepa2-312/bin/python3.12` directly, because the
copied Conda launcher retains an absolute `/opt/huawei` interpreter shebang.

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
RoboTwin as five explicit roots. Its equal source weights are only for
engineering validation. Sampling first selects a source and then a sample
within that source, so large sources cannot silently dominate. The 32-GPU
pilot keeps global batch 64 with gradient accumulation 2 and logs the observed
fraction from every source. Final scientific weights must be selected after
the strict manifest reports have been reviewed.

## Dataset support

The first implementation targets the official InternData-A1 LeRobot v2.1
layout (`meta/info.json`, per-episode Parquet, and per-camera MP4). It discovers
subdatasets recursively, chooses the head/main camera, and constructs masked
joint/gripper schemas from each `info.json`. Normalization uses `meta/stats.json`
when present; the original `sim` release is supported by frame-weighted
aggregation of its LeRobot v2.1 `meta/episodes_stats.jsonl`. LeRobot v3.0
directories are excluded explicitly rather than being misread as v2.1.

InternData-A1 is distributed separately under its own CC BY-NC-SA 4.0 terms.
No dataset files are copied into this repository.
