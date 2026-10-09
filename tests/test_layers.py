import unittest
import numpy as np
from convforge import Dense, ReLU, Flatten, MaxPool2D, GlobalAveragePool2D, Dropout
from examples.gradient_check import numerical_gradient


class LayerTests(unittest.TestCase):
    def test_dense_all_gradients(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(2, 4))
        layer = Dense(4, 3, dtype=np.float64)
        out, cache = layer.forward(x, training=True)
        dy = rng.normal(size=out.shape)
        dx = layer.backward(dy, cache)
        objective = lambda: float(np.sum(layer.forward(x)[0]*dy))
        for parameter, grad in ((x, dx), (layer.params["W"], layer.grads["W"]), (layer.params["b"], layer.grads["b"])):
            np.testing.assert_allclose(grad, numerical_gradient(parameter, objective), rtol=1e-5, atol=1e-8)

    def test_relu_and_zero_derivative(self):
        x = np.array([[-2., 0., 3.]])
        layer = ReLU()
        out, cache = layer.forward(x, training=True)
        np.testing.assert_array_equal(out, [[0, 0, 3]])
        np.testing.assert_array_equal(layer.backward(np.ones_like(x), cache), [[0, 0, 1]])

    def test_flatten_preserves_order_and_backward(self):
        x = np.arange(48, dtype=np.float64).reshape(2, 2, 3, 4)
        layer = Flatten()
        out, cache = layer.forward(x, training=True)
        self.assertEqual(out.shape, (2, 24))
        np.testing.assert_array_equal(layer.backward(out, cache), x)

    def test_pool_hand_computed_forward_backward(self):
        x = np.array([[[[1., 5., 3., 4.], [2., 3., 8., 1.], [3., 2., 1., 9.], [7., 1., 4., 2.]]]])
        layer = MaxPool2D(2)
        out, cache = layer.forward(x, training=True)
        np.testing.assert_array_equal(out, [[[[5, 8], [7, 9]]]])
        dx = layer.backward(np.ones_like(out), cache)
        expected = np.zeros_like(x)
        expected[0, 0, [0, 1, 3, 2], [1, 2, 0, 3]] = 1
        np.testing.assert_array_equal(dx, expected)

    def test_overlapping_pool_gradient(self):
        x = np.random.default_rng(8).normal(size=(1, 2, 4, 5))
        layer = MaxPool2D(3, stride=1)
        out, cache = layer.forward(x, training=True)
        dy = np.random.default_rng(9).normal(size=out.shape)
        dx = layer.backward(dy, cache)
        objective = lambda: float(np.sum(layer.forward(x)[0]*dy))
        np.testing.assert_allclose(dx, numerical_gradient(x, objective), atol=1e-8)

    def test_pool_ties_use_first_row_major_winner(self):
        layer = MaxPool2D(2)
        out, cache = layer.forward(np.ones((1, 1, 2, 2)), training=True)
        np.testing.assert_array_equal(layer.backward(np.ones_like(out), cache), [[[[1, 0], [0, 0]]]])

    def test_pool_border_not_covered_has_zero_gradient(self):
        layer = MaxPool2D(2)
        out, cache = layer.forward(np.ones((1, 1, 5, 5)), training=True)
        dx = layer.backward(np.ones_like(out), cache)
        np.testing.assert_array_equal(dx[:, :, -1], 0)
        np.testing.assert_array_equal(dx[:, :, :, -1], 0)

    def test_global_average_pool_gradients(self):
        x = np.random.default_rng(2).normal(size=(1, 2, 3, 4))
        layer = GlobalAveragePool2D()
        out, cache = layer.forward(x, training=True)
        dy = np.array([[2., -3.]])
        dx = layer.backward(dy, cache)
        objective = lambda: float(np.sum(layer.forward(x)[0]*dy))
        np.testing.assert_allclose(dx, numerical_gradient(x, objective), atol=1e-8)

    def test_dropout_cache_derivative_and_inference(self):
        layer = Dropout(0.25, seed=2)
        x = np.ones((2, 10), np.float32)
        out, cache = layer.forward(x, training=True)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(layer.backward(np.ones_like(x), cache), out)
        inference, saved = layer.forward(x)
        np.testing.assert_array_equal(inference, x)
        self.assertIsNone(saved)

    def test_dropout_seed_and_mean_preservation(self):
        x = np.ones((100, 100), np.float64)
        a = Dropout(0.3, seed=7).forward(x, training=True)[0]
        b = Dropout(0.3, seed=7).forward(x, training=True)[0]
        np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(float(a.mean()), 1, delta=0.03)

    def test_upstream_shape_is_checked(self):
        layer = Dense(3, 2, dtype=np.float64)
        _, saved = layer.forward(np.ones((1, 3)), training=True)
        with self.assertRaises(ValueError):
            layer.backward(np.ones((2, 2)), saved)


if __name__ == "__main__":
    unittest.main()
