import unittest
import numpy as np
from convforge import Conv2D, im2col, col2im, naive_conv2d
from examples.gradient_check import check


class ConvolutionTests(unittest.TestCase):
    def test_independent_scalar_reference(self):
        rng = np.random.default_rng(6)
        for dtype in (np.float32, np.float64):
            x = rng.normal(size=(2, 2, 5, 7)).astype(dtype)
            for kernel, stride, pad, dilation in (((3, 2), 1, "valid", 1),
                                                  ((2, 3), (2, 1), "same", (1, 2)),
                                                  ((3, 3), 2, (1, 2), 1)):
                with self.subTest(dtype=dtype, kernel=kernel, stride=stride, padding=pad):
                    layer = Conv2D(2, 3, kernel, stride=stride, padding=pad,
                                   dilation=dilation, dtype=dtype)
                    layer.params["b"][:] = 0.25
                    actual = layer.forward(x)[0]
                    expected = naive_conv2d(x, layer.params["W"], layer.params["b"],
                                            stride=stride, padding=pad, dilation=dilation)
                    tol = 2e-6 if dtype == np.float32 else 1e-12
                    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)

    def test_cross_correlation_does_not_flip_kernel(self):
        layer = Conv2D(1, 1, 2, padding="valid", dtype=np.float64)
        layer.params["W"][0, 0] = np.array([[1., 2.], [3., 4.]])
        x = np.arange(1., 10.).reshape(1, 1, 3, 3)
        expected = np.array([[[[37., 47.], [67., 77.]]]])
        np.testing.assert_array_equal(layer.forward(x)[0], expected)

    def test_im2col_col2im_adjoint(self):
        rng = np.random.default_rng(9)
        for stride, padding, dilation in ((1, 0, 1), (2, "same", 2), ((1, 2), (2, 1), (2, 1))):
            x = rng.normal(size=(2, 2, 6, 7))
            columns, spec = im2col(x, (2, 3), stride=stride, padding=padding, dilation=dilation)
            g = rng.normal(size=columns.shape)
            self.assertAlmostEqual(float(np.sum(columns*g)), float(np.sum(x*col2im(g, spec))), places=10)

    def test_overlap_add_counts(self):
        x = np.zeros((1, 1, 3, 3))
        columns, spec = im2col(x, 3, padding="same")
        counts = col2im(np.ones_like(columns), spec)
        np.testing.assert_array_equal(counts[0, 0], [[4, 6, 4], [6, 9, 6], [4, 6, 4]])

    def test_same_padding_output_is_ceil_with_asymmetric_padding(self):
        x = np.zeros((1, 1, 5, 6))
        _, spec = im2col(x, 2, stride=2, padding="same")
        self.assertEqual(spec.output_shape, (3, 3))
        self.assertEqual(spec.padding, (0, 1, 0, 0))

    def test_convolution_numerical_gradients(self):
        report = check()
        self.assertTrue(report["passed"])
        self.assertLess(max(v["max_absolute_error"] for v in report["errors"].values()), 2e-8)

    def test_backward_dtype_is_preserved(self):
        layer = Conv2D(1, 2, dtype=np.float32)
        x = np.ones((1, 1, 4, 4), np.float32)
        out, saved = layer.forward(x, training=True)
        dx = layer.backward(np.ones_like(out), saved)
        self.assertEqual(dx.dtype, np.float32)
        self.assertTrue(all(g.dtype == np.float32 for g in layer.grads.values()))

    def test_invalid_geometry_and_dtype(self):
        x = np.ones((1, 1, 3, 3))
        for kwargs in ({"stride": 0}, {"dilation": -1}, {"padding": "mystery"}):
            with self.assertRaises(ValueError):
                im2col(x, 3, **kwargs)
        with self.assertRaises(ValueError):
            im2col(x, 5)
        with self.assertRaises(TypeError):
            im2col(x.astype(int), 2)
        with self.assertRaises(ValueError):
            Conv2D((1, 2), 3)


if __name__ == "__main__":
    unittest.main()
