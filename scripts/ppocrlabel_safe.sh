#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="${PPOCRLABEL_ENV_ROOT:-/home/cht/miniconda3/envs/paddlelabel}"

"${ENV_ROOT}/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

env_root = Path(sys.executable).resolve().parents[1]
model_root = Path("/home/cht/.paddlex/official_models")

# Import cv2 first so we can override the plugin path it injects on Linux.
import cv2  # noqa: F401

pyqt_plugins = env_root / "lib/python3.10/site-packages/PyQt5/Qt5/plugins"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(pyqt_plugins / "platforms")
os.environ["QT_PLUGIN_PATH"] = str(pyqt_plugins)
os.environ.pop("QT_QPA_FONTDIR", None)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

det_model_dir = model_root / "PP-OCRv5_mobile_det"
rec_model_dir = model_root / "PP-OCRv5_mobile_rec"

if det_model_dir.is_dir() and "--det_model_dir" not in sys.argv:
    sys.argv.extend(["--det_model_dir", str(det_model_dir)])
if rec_model_dir.is_dir() and "--rec_model_dir" not in sys.argv:
    sys.argv.extend(["--rec_model_dir", str(rec_model_dir)])

import PPOCRLabel.PPOCRLabel as appmod


class LazyPPStructureV3:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._impl = None

    def _ensure_impl(self):
        if self._impl is None:
            from paddleocr import PPStructureV3 as real_ppstructurev3

            self._impl = real_ppstructurev3(*self._args, **self._kwargs)
        return self._impl

    def predict(self, *args, **kwargs):
        return self._ensure_impl().predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ensure_impl(), name)


appmod.PPStructureV3 = LazyPPStructureV3

sys.exit(appmod.main())
PY
