import tempfile
from pathlib import Path
import unittest
import numpy as np
from convforge import TinyCNN, Adam, cross_entropy, clip_gradients, shape_dataset
from examples.gradient_check import numerical_gradient


class ModelTests(unittest.TestCase):
    def test_full_model_parameter_and_pixel_gradients(self):
        model = TinyCNN(input_size=8, channels=(2, 3), hidden=5, num_classes=3,
                        dropout=0, dtype="float64", seed=9)
        x = np.random.default_rng(4).normal(size=(2, 1, 8, 8))
        targets = np.array([0, 2])
        logits, cache, _ = model.forward(x, training=True)
        _, dy = cross_entropy(logits, targets)
        dx = model.backward(dy, cache)
        grads = {name: g.copy() for name, g in model.gradients().items()}
        objective = lambda: cross_entropy(model.forward(x)[0], targets)[0]
        np.testing.assert_allclose(dx, numerical_gradient(x, objective), atol=1e-8, rtol=2e-5)
        for name, p in model.parameters().items():
            np.testing.assert_allclose(grads[name], numerical_gradient(p, objective), atol=1e-8, rtol=2e-5, err_msg=name)

    def test_sensitivity_matches_inference_logit_derivative(self):
        model = TinyCNN(input_size=8, channels=(2, 3), hidden=5, num_classes=3,
                        dtype="float64", dropout=0.4, seed=7)
        x = np.random.default_rng(13).normal(size=(1, 1, 8, 8))
        grad = model.input_gradient(x, 1)
        numeric = numerical_gradient(x, lambda: float(model.forward(x)[0][0, 1]))
        np.testing.assert_allclose(grad, numeric, atol=1e-8, rtol=2e-5)

    def test_checkpoint_exact_roundtrip(self):
        model = TinyCNN()
        x, _ = shape_dataset(1, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            model.save(path)
            restored = TinyCNN.load(path)
            np.testing.assert_array_equal(model.predict_proba(x), restored.predict_proba(x))

    def test_trace_shapes_and_parameter_count(self):
        model = TinyCNN()
        x = np.ones((2, 1, 24, 24), np.float32)
        logits, _, trace = model.forward(x, trace=True)
        self.assertEqual(trace["conv1"].shape, (2, 8, 24, 24))
        self.assertEqual(trace["pool2"].shape, (2, 16, 6, 6))
        self.assertEqual(logits.shape, (2, 6))
        self.assertEqual(model.parameter_count, 19910)

    def test_wrong_model_shape_rejected(self):
        with self.assertRaises(ValueError):
            TinyCNN().forward(np.zeros((1, 1, 28, 28), np.float32))

    def test_small_training_reduces_loss(self):
        x, y = shape_dataset(8, seed=12, noise=0.01)
        model = TinyCNN(dropout=0, seed=42)
        initial = cross_entropy(model.forward(x)[0], y)[0]
        optimizer = Adam(model.parameters())
        for _ in range(35):
            _, _, grads = model.loss_and_grads(x, y)
            clip_gradients(grads)
            optimizer.step(grads)
        final = cross_entropy(model.forward(x)[0], y)[0]
        self.assertLess(final, initial*0.3)
        self.assertGreater(float(np.mean(model.forward(x)[0].argmax(-1) == y)), 0.9)


class OptimizerTests(unittest.TestCase):
    def test_cross_entropy_stability_and_derivative(self):
        logits = np.array([[1000., 999., 998.], [-1000., -1002., -999.]])
        targets = np.array([1, 2])
        loss, grad = cross_entropy(logits, targets)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(grad, numerical_gradient(logits, lambda: cross_entropy(logits, targets)[0]), atol=1e-7)

    def test_adam_first_update_and_invalid_gradient_atomicity(self):
        p = np.array([2.], dtype=np.float64)
        optimizer = Adam({"x": p}, lr=0.1)
        optimizer.step({"x": np.array([1.])})
        self.assertAlmostEqual(p[0], 2-0.1/(1+1e-8), places=12)
        with self.assertRaises(ValueError):
            optimizer.step({"x": np.array([np.nan])})
        self.assertEqual(optimizer.step_count, 1)

    def test_global_gradient_clipping(self):
        gradients = {"a": np.array([3.]), "b": np.array([4.])}
        self.assertEqual(clip_gradients(gradients, 2), 5)
        self.assertAlmostEqual(float(np.sqrt(sum((g*g).sum() for g in gradients.values()))), 2)


if __name__ == "__main__":
    unittest.main()
