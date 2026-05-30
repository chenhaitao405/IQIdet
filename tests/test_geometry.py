import unittest

import numpy as np

from gauge.geometry import project_ocr_items_to_image, project_roi_box_to_image


class GeometryProjectionTest(unittest.TestCase):
    def test_project_box_identity_matrix_without_rotation(self) -> None:
        box = [[1, 2], [3, 2], [3, 4], [1, 4]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(image_box, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(unrotated_box, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])

    def test_project_box_undoes_ccw90_rotation(self) -> None:
        box = [[2, 1], [4, 1]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=[10, 20],
            rotated=True,
        )

        self.assertEqual(unrotated_box, [[8.0, 2.0], [8.0, 4.0]])
        self.assertEqual(image_box, [[8.0, 2.0], [8.0, 4.0]])

    def test_missing_matrix_returns_roi_space_fallback(self) -> None:
        box = [[1, 2], [3, 4]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=None,
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(image_box, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(unrotated_box, [[1.0, 2.0], [3.0, 4.0]])

    def test_project_ocr_items_adds_image_and_unrotated_boxes(self) -> None:
        items = [{"text": "10FEJB", "box": [[1, 2], [3, 2], [3, 4], [1, 4]]}]
        projected = project_ocr_items_to_image(
            items,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(projected[0]["box_image"], [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(projected[0]["box_roi_unrotated"], [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(items[0].get("box_image"), None)


if __name__ == "__main__":
    unittest.main()
