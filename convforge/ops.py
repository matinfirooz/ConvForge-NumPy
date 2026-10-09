"""NCHW cross-correlation primitives: im2col and its overlap-adding adjoint."""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pair(value, name, *, allow_zero=False):
    values = (value, value) if isinstance(value, (int, np.integer)) else value
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must be an integer or a pair")
    for item in values:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(f"{name} must contain integers")
        if item < (0 if allow_zero else 1):
            raise ValueError(f"{name} is out of range")
    return tuple(int(v) for v in values)


def floating(x, ndim=None):
    if not isinstance(x, np.ndarray):
        raise TypeError("input must be a NumPy array")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"input must have {ndim} dimensions")
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("use float32 or float64 arrays")
    if min(x.shape, default=0) <= 0 or not np.isfinite(x).all():
        raise ValueError("input dimensions must be nonzero and values finite")


def geometry(height, width, kernel, stride, dilation, padding):
    kh, kw = kernel
    sh, sw = stride
    dh, dw = dilation
    eh, ew = (kh-1)*dh+1, (kw-1)*dw+1
    if isinstance(padding, str):
        if padding == "same":
            oh, ow = (height+sh-1)//sh, (width+sw-1)//sw
            ph, pw = max(0, (oh-1)*sh+eh-height), max(0, (ow-1)*sw+ew-width)
            pads = (ph//2, ph-ph//2, pw//2, pw-pw//2)
        elif padding == "valid":
            pads = (0, 0, 0, 0)
        else:
            raise ValueError("padding must be 'same', 'valid', or a nonnegative integer/pair")
    else:
        ph, pw = pair(padding, "padding", allow_zero=True)
        pads = (ph, ph, pw, pw)
    top, bottom, left, right = pads
    oh = (height+top+bottom-eh)//sh+1
    ow = (width+left+right-ew)//sw+1
    if oh <= 0 or ow <= 0:
        raise ValueError("effective kernel is larger than the padded input")
    return (oh, ow), pads


@dataclass(frozen=True)
class ColumnSpec:
    input_shape: tuple
    kernel: tuple
    stride: tuple
    dilation: tuple
    padding: tuple
    output_shape: tuple


def im2col(x, kernel_size, *, stride=1, padding=0, dilation=1):
    """Return a (N*OH*OW, C*KH*KW) patch matrix and its geometry.

    The sliding-window view is read-only. The final matrix reshape can make a
    copy; this transparent CPU implementation does not claim constant memory.
    'same' padding uses ceil(H/stride), including asymmetric padding if needed.
    """
    floating(x, 4)
    kernel, stride, dilation = (pair(v, n) for v, n in
                                ((kernel_size, "kernel_size"), (stride, "stride"), (dilation, "dilation")))
    (oh, ow), pads = geometry(*x.shape[-2:], kernel, stride, dilation, padding)
    top, bottom, left, right = pads
    padded = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    kh, kw = kernel
    dh, dw = dilation
    windows = sliding_window_view(padded, ((kh-1)*dh+1, (kw-1)*dw+1), axis=(-2, -1))
    windows = windows[:, :, ::stride[0], ::stride[1], ::dh, ::dw]
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(x.shape[0]*oh*ow, -1)
    return columns, ColumnSpec(x.shape, kernel, stride, dilation, pads, (oh, ow))


def col2im(columns, spec):
    """Apply im2col's adjoint: scatter-add, NOT a mathematical inverse."""
    floating(columns, 2)
    n, c, h, w = spec.input_shape
    kh, kw = spec.kernel
    oh, ow = spec.output_shape
    if columns.shape != (n*oh*ow, c*kh*kw):
        raise ValueError("columns do not match their ColumnSpec")
    top, bottom, left, right = spec.padding
    result = np.zeros((n, c, h+top+bottom, w+left+right), dtype=columns.dtype)
    patches = columns.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    sh, sw = spec.stride
    dh, dw = spec.dilation
    # Each slice has unique indices; successive kernel offsets add overlaps.
    for i in range(kh):
        for j in range(kw):
            ys, xs = i*dh, j*dw
            result[:, :, ys:ys+oh*sh:sh, xs:xs+ow*sw:sw] += patches[:, :, i, j]
    return result[:, :, top:top+h, left:left+w]


def naive_conv2d(x, weight, bias=None, *, stride=1, padding=0, dilation=1):
    """Independent scalar-loop forward reference, useful for learning/testing."""
    floating(x, 4)
    floating(weight, 4)
    if x.dtype != weight.dtype or x.shape[1] != weight.shape[1]:
        raise ValueError("weight channels and dtype must match input")
    kernel, stride, dilation = weight.shape[-2:], pair(stride, "stride"), pair(dilation, "dilation")
    (oh, ow), pads = geometry(*x.shape[-2:], kernel, stride, dilation, padding)
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != x.dtype):
        raise ValueError("bias shape/dtype mismatch")
    top, bottom, left, right = pads
    padded = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
    result = np.empty((x.shape[0], weight.shape[0], oh, ow), dtype=x.dtype)
    for n in range(x.shape[0]):
        for out in range(weight.shape[0]):
            for y in range(oh):
                for xx in range(ow):
                    total = 0.0 if bias is None else float(bias[out])
                    for c in range(x.shape[1]):
                        for ky in range(kernel[0]):
                            for kx in range(kernel[1]):
                                total += float(padded[n, c, y*stride[0]+ky*dilation[0],
                                                      xx*stride[1]+kx*dilation[1]]) * float(weight[out, c, ky, kx])
                    result[n, out, y, xx] = total
    return result
