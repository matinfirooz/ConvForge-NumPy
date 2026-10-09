"""A complete hand-computable convolution, ReLU, pool, and backward example."""
import numpy as np
from convforge import Conv2D, ReLU, MaxPool2D, im2col


def main():
    np.set_printoptions(precision=3, suppress=True)
    x = np.arange(1., 17.).reshape(1, 1, 4, 4)
    conv = Conv2D(1, 1, 3, padding="valid", dtype=np.float64)
    conv.params["W"][0, 0] = np.array([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
    conv.params["b"][0] = 10
    columns, _ = im2col(x, 3)
    y, saved_conv = conv.forward(x, training=True)
    relu, pool = ReLU(), MaxPool2D(2)
    activated, saved_relu = relu.forward(y, training=True)
    pooled, saved_pool = pool.forward(activated, training=True)
    print("1. Input:\n", x[0, 0])
    print("2. Shared kernel:\n", conv.params["W"][0, 0], "\nBias:", conv.params["b"])
    print("3. im2col: every row is one 3x3 patch:\n", columns)
    print("4. Cross-correlation:\n", y[0, 0])
    print("5. ReLU then 2x2 max pool:\n", pooled[0, 0])
    gradient = pool.backward(np.ones_like(pooled), saved_pool)
    gradient = relu.backward(gradient, saved_relu)
    dx = conv.backward(gradient, saved_conv)
    print("6. Gradient of sum(pooled) wrt input:\n", dx[0, 0])
    print("7. Kernel gradient:\n", conv.grads["W"][0, 0])
    print("Pooling ties choose the first row-major winner; forward uses cross-correlation.")


if __name__ == "__main__":
    main()
