import io
import json
import threading
import unittest
from unittest import mock

from gauge.ocr_runtime import PaddleOCRSubprocessClient


class _FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode = None
        self.wait_called = False
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_called = True
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class PaddleOCRSubprocessClientRuntimeTest(unittest.TestCase):
    def _client_without_start(self):
        client = object.__new__(PaddleOCRSubprocessClient)
        client.process = _FakeProcess()
        client._lock = threading.Lock()
        client.request_timeout_s = 1.0
        return client

    def test_request_uses_lock_and_returns_result(self) -> None:
        client = self._client_without_start()
        with mock.patch.object(
            client,
            "_readline_with_timeout",
            return_value=json.dumps({"ok": True, "result": {"rec_text": "X"}}) + "\n",
        ):
            result = client._request({"op": "recognize", "image": {"format": "png_base64", "data": ""}})

        self.assertEqual(result["result"]["rec_text"], "X")
        self.assertIn('"op": "recognize"', client.process.stdin.getvalue())

    def test_request_timeout_closes_process(self) -> None:
        client = self._client_without_start()
        with mock.patch.object(client, "_readline_with_timeout", side_effect=TimeoutError("OCR worker response timed out")):
            with self.assertRaises(RuntimeError) as caught:
                client._request({"op": "detect", "image": {"format": "png_base64", "data": ""}})

        self.assertIn("timed out", str(caught.exception))
        self.assertIsNone(client.process)

    def test_close_is_idempotent(self) -> None:
        client = self._client_without_start()

        client.close()
        client.close()

        self.assertIsNone(client.process)


if __name__ == "__main__":
    unittest.main()
