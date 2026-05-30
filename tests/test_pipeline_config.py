import unittest
from types import SimpleNamespace

from gauge.config import (
    CorrectionConfig,
    EnhanceConfig,
    FClipConfig,
    GaugeConfig,
    OCRConfig,
    PipelineConfig,
)


def _args(**overrides):
    defaults = {
        "gauge_weights": None,
        "gauge_conf": None,
        "gauge_iou": None,
        "gauge_imgsz": None,
        "gauge_device": None,
        "gauge_select": None,
        "gauge_class": None,
        "fclip_ckpt": None,
        "fclip_device": None,
        "fclip_config": None,
        "fclip_params": None,
        "fclip_threshold": None,
        "ocr_device": None,
        "ocr_det_model_name": None,
        "ocr_det_model_dir": None,
        "ocr_rec_model_name": None,
        "ocr_rec_model_dir": None,
        "ocr_det_limit_side_len": None,
        "ocr_det_limit_type": None,
        "ocr_min_score": None,
        "ocr_number_range": None,
        "enable_ocr_orientation": None,
        "ocr_orientation_model": None,
        "ocr_orientation_device": None,
        "ocr_orientation_verbose": False,
        "enable_correction": None,
        "correction_model": None,
        "correction_device": None,
        "correction_verbose": None,
        "enhance_mode": None,
        "no_rotate": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PipelineConfigOverrideTest(unittest.TestCase):
    def test_apply_cli_overrides_preserves_nested_model_types(self) -> None:
        cfg = PipelineConfig().apply_cli_overrides(
            _args(
                gauge_conf=0.5,
                fclip_threshold=0.7,
                ocr_min_score=0.2,
                enable_correction=True,
                no_rotate=True,
            )
        )

        self.assertIsInstance(cfg.gauge, GaugeConfig)
        self.assertIsInstance(cfg.fclip, FClipConfig)
        self.assertIsInstance(cfg.ocr, OCRConfig)
        self.assertIsInstance(cfg.correction, CorrectionConfig)
        self.assertIsInstance(cfg.enhance, EnhanceConfig)
        self.assertEqual(cfg.gauge.conf, 0.5)
        self.assertEqual(cfg.fclip.threshold, 0.7)
        self.assertEqual(cfg.ocr.min_score, 0.2)
        self.assertTrue(cfg.correction.enabled)
        self.assertFalse(cfg.enhance.rotate_roi)

    def test_apply_cli_overrides_keeps_original_when_no_override(self) -> None:
        cfg = PipelineConfig()
        updated = cfg.apply_cli_overrides(_args())

        self.assertIs(updated, cfg)


if __name__ == "__main__":
    unittest.main()
