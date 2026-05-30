import unittest

from gauge.imaging.visualization import (
    build_final_result_vis_image,
    build_wire_vis_image,
    save_debug_visualizations,
)


class VisualizationModuleTest(unittest.TestCase):
    def test_visualization_helpers_importable(self) -> None:
        """Verify that visualization helper functions are importable."""
        self.assertIsNotNone(build_wire_vis_image)
        self.assertIsNotNone(build_final_result_vis_image)
        self.assertIsNotNone(save_debug_visualizations)


if __name__ == "__main__":
    unittest.main()
