import base64
import unittest

import cv2
import numpy as np

from gauge.runtime.region_runtime import decode_base64, register_region_service_shutdown, shutdown_region_runtime


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RegionRuntimeTest(unittest.TestCase):
    def test_decode_base64_accepts_data_url_prefix(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        payload = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        decoded = decode_base64(payload)

        self.assertEqual(decoded.shape[:2], (4, 5))

    def test_decode_base64_rejects_invalid_payload(self) -> None:
        with self.assertRaises(Exception) as caught:
            decode_base64("not-valid-base64")

        self.assertTrue(hasattr(caught.exception, "status_code"))
        self.assertEqual(caught.exception.status_code, 400)

    def test_registered_services_are_closed(self) -> None:
        service = _Closable()
        register_region_service_shutdown(lambda: service.close())

        shutdown_region_runtime()

        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
