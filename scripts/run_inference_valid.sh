#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

python "${REPO_ROOT}/run_inference_pipeline.py" \
  --image-dir "${REPO_ROOT}/IQIdata/processed/valid" \
  --fclip-ckpt "${REPO_ROOT}/local/fclip67.pth.tar" \
  --gauge-weights "${REPO_ROOT}/local/gauge.pt" \
  --output-dir "${REPO_ROOT}/outputs/gauge_infer_valid" \
  --vis
