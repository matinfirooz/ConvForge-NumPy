"""Explicit forward/backward layers. Caches belong to the caller; no autograd."""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .ops import pair, floating, im2col, col2im


def _width(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _gradient(dy, expected, dtype):
    floating(dy)
    if dy.shape != expected or dy.dtype != dtype:
        raise ValueError("upstream gradient must match output shape and input dtype")


class Conv2D:
    """NCHW cross-correlation with OIHW weights and He initialization.

    Backward overwrites this layer's parameter gradients. Weight tying and
    implicit gradient accumulation are not provided. Do not update weights
    until all backward calls using the current weights have finished.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, *, stride=1,
                 padding="same", dilation=1, seed=42, dtype=np.float32):
        self.in_channels = _width(in_channels, "in_channels")
        self.out_channels = _width(out_channels, "out_channels")
        self.kernel = pair(kernel_size, "kernel_size")
        self.stride, self.dilation = pair(stride, "stride"), pair(dilation, "dilation")
        self.padding = padding
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("use float32 or float64")
        rng = np.random.default_rng(seed)
        fan_in = self.in_channels*np.prod(self.kernel)
        shape = (self.out_channels, self.in_channels, *self.kernel)
        self.params = {"W": (rng.normal(size=shape)*np.sqrt(2/fan_in)).astype(dtype),
                       "b": np.zeros(self.out_channels, dtype=dtype)}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    def forward(self, x, *, training=False):
        floating(x, 4)
        if x.shape[1] != self.in_channels or x.dtype != self.params["W"].dtype:
            raise ValueError("input channels/dtype do not match the convolution")
        columns, spec = im2col(x, self.kernel, stride=self.stride,
                              padding=self.padding, dilation=self.dilation)
        flat = columns @ self.params["W"].reshape(self.out_channels, -1).T + self.params["b"]
        oh, ow = spec.output_shape
        output = flat.reshape(x.shape[0], oh, ow, self.out_channels).transpose(0, 3, 1, 2)
        return output, (columns, spec) if training else None

    def backward(self, dy, cache):
        columns, spec = cache
        _gradient(dy, (spec.input_shape[0], self.out_channels, *spec.output_shape), columns.dtype)
        flat = dy.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grads["W"][...] = (flat.T @ columns).reshape(self.params["W"].shape)
        self.grads["b"][...] = flat.sum(axis=0)
        return col2im(flat @ self.params["W"].reshape(self.out_channels, -1), spec)


class Dense:
    def __init__(self, in_features, out_features, *, seed=42, dtype=np.float32):
        in_features, out_features = _width(in_features, "in_features"), _width(out_features, "out_features")
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("use float32 or float64")
        rng = np.random.default_rng(seed)
        self.params = {"W": (rng.normal(size=(in_features, out_features))*np.sqrt(2/in_features)).astype(dtype),
                       "b": np.zeros(out_features, dtype=dtype)}
        self.grads = {name: np.zeros_like(value) for name, value in self.params.items()}

    def forward(self, x, *, training=False):
        floating(x, 2)
        if x.shape[-1] != self.params["W"].shape[0] or x.dtype != self.params["W"].dtype:
            raise ValueError("input features/dtype do not match Dense")
        return x @ self.params["W"] + self.params["b"], x if training else None

    def backward(self, dy, cache):
        _gradient(dy, (cache.shape[0], self.params["W"].shape[1]), cache.dtype)
        self.grads["W"][...] = cache.T @ dy
        self.grads["b"][...] = dy.sum(axis=0)
        return dy @ self.params["W"].T


class ReLU:
    def forward(self, x, *, training=False):
        floating(x)
        return np.maximum(x, 0), (x > 0, x.dtype) if training else None

    def backward(self, dy, cache):
        mask, dtype = cache
        _gradient(dy, mask.shape, dtype)
        return dy*mask  # Defined derivative at zero: zero.


class Flatten:
    def forward(self, x, *, training=False):
        floating(x)
        return x.reshape(x.shape[0], -1), (x.shape, x.dtype) if training else None

    def backward(self, dy, cache):
        shape, dtype = cache
        _gradient(dy, (shape[0], int(np.prod(shape[1:]))), dtype)
        return dy.reshape(shape)


class MaxPool2D:
    """Valid pooling; first row-major maximum wins ties; overlaps add in backward."""
    def __init__(self, kernel_size=2, *, stride=None):
        self.kernel = pair(kernel_size, "kernel_size")
        self.stride = self.kernel if stride is None else pair(stride, "stride")

    def forward(self, x, *, training=False):
        floating(x, 4)
        kh, kw = self.kernel
        if kh > x.shape[-2] or kw > x.shape[-1]:
            raise ValueError("pooling kernel exceeds input")
        windows = sliding_window_view(x, (kh, kw), axis=(-2, -1))
        windows = windows[:, :, ::self.stride[0], ::self.stride[1]]
        shape = windows.shape[:4]
        flat = windows.reshape(*shape, kh*kw)
        winners = flat.argmax(axis=-1)
        output = np.take_along_axis(flat, winners[..., None], axis=-1)[..., 0]
        return output, (x.shape, winners, x.dtype) if training else None

    def backward(self, dy, cache):
        shape, winners, dtype = cache
        _gradient(dy, winners.shape, dtype)
        dx = np.zeros(shape, dtype=dy.dtype)
        oh, ow = dy.shape[-2:]
        sh, sw = self.stride
        for ky in range(self.kernel[0]):
            for kx in range(self.kernel[1]):
                dx[:, :, ky:ky+oh*sh:sh, kx:kx+ow*sw:sw] += dy*(winners == ky*self.kernel[1]+kx)
        return dx


class GlobalAveragePool2D:
    def forward(self, x, *, training=False):
        floating(x, 4)
        return x.mean(axis=(-2, -1)), (x.shape, x.dtype) if training else None

    def backward(self, dy, cache):
        shape, dtype = cache
        _gradient(dy, shape[:2], dtype)
        return np.broadcast_to(dy[:, :, None, None]/(shape[-2]*shape[-1]), shape).copy()


class Dropout:
    """Inverted dropout; inference is identity. RNG is owned by this layer."""
    def __init__(self, rate=0.1, *, seed=42):
        if not np.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError("dropout rate must be in [0, 1)")
        self.rate, self.rng = float(rate), np.random.default_rng(seed)

    def forward(self, x, *, training=False):
        floating(x)
        if not training:
            return x, None
        mask = (self.rng.random(x.shape) >= self.rate).astype(x.dtype)/(1-self.rate)
        return x*mask, mask

    def backward(self, dy, cache):
        _gradient(dy, cache.shape, cache.dtype)
        return dy*cache
