import py_compile
import runpy
import sys
import types
import unittest
from pathlib import Path


def _run_root_module(repo_root: Path, module_name: str) -> None:
    module_path = repo_root / f"{module_name}.py"
    before_path = list(sys.path)
    watched_modules = [
        module_name,
        "gauge",
        "gauge.services.ocr.factory",
        "gauge.services.ocr.infer",
        "gauge.services.ocr.normalize",
        "gauge.services.ocr.debug",
        "gauge.domain.statistics",
        "FClip",
        "dataset",
    ]
    before_modules = {name: sys.modules.get(name) for name in watched_modules}
    try:
        if module_name == "run_iqi_grade_infer":
            fake_fclip_inferencer = types.ModuleType("gauge.services.fclip.inferencer")
            fake_fclip_inferencer.FClipInferencer = object
            sys.modules["gauge.services.fclip.inferencer"] = fake_fclip_inferencer

            fake_ocr_infer = types.ModuleType("gauge.services.ocr.infer")
            fake_ocr_infer.infer_roi_ocr = lambda *args, **kwargs: {}
            sys.modules["gauge.services.ocr.infer"] = fake_ocr_infer

            fake_ocr_debug = types.ModuleType("gauge.services.ocr.debug")
            fake_ocr_debug.build_ocr_item_debug_images = lambda *args, **kwargs: {}
            fake_ocr_debug.draw_ocr_on_roi = lambda *args, **kwargs: None
            sys.modules["gauge.services.ocr.debug"] = fake_ocr_debug

            fake_domain_statistics = types.ModuleType("gauge.domain.statistics")
            fake_domain_statistics.build_ocr_statistics = lambda *args, **kwargs: {}
            sys.modules["gauge.domain.statistics"] = fake_domain_statistics

        runpy.run_path(str(module_path), run_name=f"__test_{module_name}__")
    finally:
        sys.path[:] = before_path
        for name, module in before_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class DebugScriptLayoutTest(unittest.TestCase):
    def test_source_packages_live_under_src(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"

        for package_name in ["gauge", "FClip", "dataset"]:
            self.assertTrue((src_dir / package_name).is_dir(), f"missing src package: {package_name}")
            self.assertFalse((repo_root / package_name).exists(), f"root package should be moved: {package_name}")

    def test_root_delivery_entrypoints_bootstrap_src_imports(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        for module_name in ["region_ocr_api", "region_SNR_api", "run_iqi_grade_infer"]:
            _run_root_module(repo_root, module_name)

    def test_debug_scripts_live_under_scripts_debug_and_compile(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        debug_dir = repo_root / "scripts" / "debug"

        moved_scripts = [
            debug_dir / "fclip_valid.py",
            debug_dir / "run_region_ocr_batch.py",
        ]
        for script in moved_scripts:
            self.assertTrue(script.is_file(), f"missing debug script: {script}")
            py_compile.compile(str(script), doraise=True)

        self.assertFalse((repo_root / "fclip_valid.py").exists())
        self.assertFalse((repo_root / "run_region_ocr_batch.py").exists())
        self.assertFalse((repo_root / "src" / "gauge" / "training" / "infer.py").exists())
        self.assertFalse((repo_root / "src" / "gauge" / "training" / "valid.py").exists())

    def test_training_helpers_do_not_execute_at_import(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "src" / "gauge" / "training" / "OBBtraintest.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("main()", source)


if __name__ == "__main__":
    unittest.main()
