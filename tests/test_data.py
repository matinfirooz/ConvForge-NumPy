import gzip
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from convforge import shape_dataset, load_idx, batches


class DataTests(unittest.TestCase):
    def test_generator_deterministic_balanced_and_bounded(self):
        x, y = shape_dataset(5, seed=4)
        a, b = shape_dataset(5, seed=4)
        np.testing.assert_array_equal(x, a)
        np.testing.assert_array_equal(y, b)
        np.testing.assert_array_equal(np.bincount(y), [5]*6)
        self.assertEqual(x.shape, (30, 1, 24, 24))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(((x >= 0) & (x <= 1)).all())

    def test_split_seeds_generate_distinct_examples(self):
        x, _ = shape_dataset(2, seed=4)
        y, _ = shape_dataset(2, seed=5)
        self.assertFalse(np.array_equal(x, y))

    def test_batches_cover_every_example_once(self):
        x = np.arange(35).reshape(7, 5)
        y = np.arange(7)
        seen = []
        for bx, by in batches(x, y, batch_size=3, rng=np.random.default_rng(2)):
            np.testing.assert_array_equal(bx[:, 0], by*5)
            seen.extend(by.tolist())
        self.assertEqual(sorted(seen), list(range(7)))

    def test_idx_raw_and_gzip(self):
        pixels = np.array([[[0, 127, 255], [13, 20, 32]], [[100, 0, 1], [0, 255, 42]]], dtype=np.uint8)
        image_data = struct.pack(">IIII", 2051, 2, 2, 3)+pixels.tobytes()
        label_data = struct.pack(">II", 2049, 2)+bytes([4, 9])
        with tempfile.TemporaryDirectory() as directory:
            for compressed in (False, True):
                ip = Path(directory)/("images.gz" if compressed else "images")
                lp = Path(directory)/("labels.gz" if compressed else "labels")
                ip.write_bytes(gzip.compress(image_data) if compressed else image_data)
                lp.write_bytes(gzip.compress(label_data) if compressed else label_data)
                x, y = load_idx(ip, lp)
                np.testing.assert_allclose(x[:, 0], pixels.astype(np.float32)/255)
                np.testing.assert_array_equal(y, [4, 9])

    def test_idx_rejects_malformed_counts_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            ip, lp = Path(directory)/"images", Path(directory)/"labels"
            ip.write_bytes(struct.pack(">IIII", 2051, 2, 2, 2)+bytes(8))
            lp.write_bytes(struct.pack(">II", 2049, 1)+bytes(1))
            with self.assertRaises(ValueError):
                load_idx(ip, lp)
            lp.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                load_idx(ip, lp)


if __name__ == "__main__":
    unittest.main()
