import importlib
import importlib.abc
import sys
import unittest


class _BlockHeavyImports(importlib.abc.MetaPathFinder):
    blocked_roots = {"torch", "FClip", "ultralytics", "paddleocr"}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self.blocked_roots:
            raise AssertionError(f"Blocked heavy import during pure import: {fullname}")
        return None


class ImportBoundaryTest(unittest.TestCase):
    def test_pipeline_import_does_not_load_heavy_model_runtime(self) -> None:
        blocked = _BlockHeavyImports()
        removed = {}
        for name in list(sys.modules):
            if name.split(".", 1)[0] in blocked.blocked_roots or name.startswith("gauge"):
                removed[name] = sys.modules.pop(name)
        sys.meta_path.insert(0, blocked)
        try:
            module = importlib.import_module("gauge.pipeline")
            self.assertTrue(hasattr(module, "PipelineRunner"))
        finally:
            sys.meta_path.remove(blocked)
            for name in list(sys.modules):
                if name.startswith("gauge"):
                    sys.modules.pop(name)
            sys.modules.update(removed)


if __name__ == "__main__":
    unittest.main()
