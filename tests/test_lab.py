import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import numpy as np
from convforge import TinyCNN, shape_dataset, CLASS_NAMES
from examples.serve_lab import make_server


class LaboratoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.model = TinyCNN(class_names=CLASS_NAMES)
        path = Path(self.directory.name)/"model.npz"
        self.model.save(path)
        self.server = make_server(port=0, checkpoint=path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def test_metadata_and_actual_numpy_prediction(self):
        with urlopen(self.base+"/api/metadata", timeout=5) as response:
            metadata = json.load(response)
        self.assertEqual(metadata["backend"], "NumPy")
        self.assertEqual(metadata["parameters"], 19910)
        x, _ = shape_dataset(1, seed=8)
        request = Request(self.base+"/api/predict", data=json.dumps({"pixels": x[0, 0].tolist()}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        np.testing.assert_allclose(result["probabilities"], self.model.predict_proba(x[:1])[0], atol=1e-7)
        self.assertEqual(len(result["conv1_maps"]), 8)
        self.assertEqual(len(result["conv2_maps"]), 16)
        self.assertEqual(np.array(result["sensitivity"]).shape, (24, 24))

    def test_invalid_pixels_rejected(self):
        request = Request(self.base+"/api/predict", data=b'{"pixels":[[2.0]]}',
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 400)

    def test_html_and_unknown_path(self):
        with urlopen(self.base, timeout=5) as response:
            html = response.read().decode()
        self.assertIn("What does a CNN see?", html)
        self.assertNotIn("__LAB_DATA__", html)
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base+"/unknown", timeout=5)
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
