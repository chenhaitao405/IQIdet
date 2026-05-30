import unittest

import gauge.iqi_inferencer as inferencer
from gauge.imaging.visualization import (
    build_final_result_vis_image,
    build_wire_vis_image,
    save_debug_visualizations,
)


class VisualizationModuleTest(unittest.TestCase):
    def test_iqi_inferencer_reexports_visualization_helpers(self) -> None:
        self.assertIs(inferencer.build_wire_vis_image, build_wire_vis_image)
        self.assertIs(inferencer.build_final_result_vis_image, build_final_result_vis_image)
        self.assertIs(inferencer.save_debug_visualizations, save_debug_visualizations)


if __name__ == "__main__":
    unittest.main()
