#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/debug/preflight.sh --skip-checksum
bash scripts/debug/run_tests.sh
bash scripts/debug/run_checkpoint_audit.sh
bash scripts/debug/run_1gpu_smoke.sh
bash scripts/debug/run_8gpu_smoke.sh
