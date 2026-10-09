"""Offline randomized shape images and a strict loader for local MNIST IDX files."""
from pathlib import Path
import gzip
import struct
import numpy as np

CLASS_NAMES = ("circle", "square", "triangle", "plus", "cross", "stripes")


def shape_dataset(per_class=300, *, seed=42, size=24, dtype=np.float32, noise=0.065):
    """Balanced filled silhouettes with varied position, size, orientation, and noise.

    No downloaded data or image library. Splits must use separate seeds.
    Rotation is restricted to +/-15 degrees so plus/cross remain distinct.
    This is a generated teaching dataset, not a real-world vision benchmark.
    """
    if not isinstance(per_class, int) or per_class <= 0 or not isinstance(size, int) or size < 8:
        raise ValueError("per_class must be positive and size >=8")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be nonnegative and finite")
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("use float32 or float64")
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1, 1, size)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    images, targets = [], []
    for label in range(len(CLASS_NAMES)):
        for _ in range(per_class):
            angle = rng.uniform(-np.pi/12, np.pi/12)
            dx, dy = rng.uniform(-0.18, 0.18, 2)
            radius = rng.uniform(0.42, 0.69)
            u = ((xx-dx)*np.cos(angle)+(yy-dy)*np.sin(angle))/radius
            v = (-(xx-dx)*np.sin(angle)+(yy-dy)*np.cos(angle))/radius
            if label == 0:
                foreground = u*u+v*v <= 1
            elif label == 1:
                foreground = (np.abs(u) <= 0.9) & (np.abs(v) <= 0.9)
            elif label == 2:
                foreground = (v >= -1) & (v <= 0.8) & (np.abs(u) <= (v+1)/1.8)
            elif label in (3, 4):
                if label == 4:
                    u, v = (u+v)/np.sqrt(2), (u-v)/np.sqrt(2)
                foreground = ((np.abs(u) <= 0.24) & (np.abs(v) <= 1)) | ((np.abs(v) <= 0.24) & (np.abs(u) <= 1))
            else:
                foreground = (np.minimum.reduce([np.abs(u-shift) for shift in (-0.62, 0, 0.62)]) <= 0.13) & (np.abs(v) <= 1)
            # Separable [1,2,1]/4 blur softens pixel-grid edges using NumPy alone.
            f = foreground.astype(np.float64)
            f = (np.pad(f, ((1,1),(0,0)))[0:size]+2*f+np.pad(f, ((1,1),(0,0)))[2:size+2])/4
            f = (np.pad(f, ((0,0),(1,1)))[:, 0:size]+2*f+np.pad(f, ((0,0),(1,1)))[:, 2:size+2])/4
            background, brightness = rng.uniform(0, 0.1), rng.uniform(0.7, 1.0)
            image = np.clip(background+brightness*f+rng.normal(0, noise, f.shape), 0, 1)
            images.append(image.astype(dtype))
            targets.append(label)
    images = np.stack(images)[:, None]
    targets = np.array(targets, dtype=np.int64)
    order = rng.permutation(len(targets))
    return images[order], targets[order]


def batches(x, y, *, batch_size=64, rng=None):
    if len(x) != len(y) or len(x) == 0 or batch_size <= 0:
        raise ValueError("invalid dataset or batch size")
    order = np.arange(len(y)) if rng is None else rng.permutation(len(y))
    for start in range(0, len(y), batch_size):
        selection = order[start:start+batch_size]
        yield x[selection], y[selection]


def _bytes(path):
    path = Path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def load_idx(images_path, labels_path, *, dtype=np.float32):
    """Read uncompressed or gzip MNIST-style unsigned-byte IDX files from disk.

    No downloading is attempted. Header/count/length mismatches are rejected.
    Return NCHW pixels normalized to [0,1] and int64 labels.
    """
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("use float32 or float64")
    image_bytes, label_bytes = _bytes(images_path), _bytes(labels_path)
    if len(image_bytes) < 16 or len(label_bytes) < 8:
        raise ValueError("truncated IDX header")
    magic, count, height, width = struct.unpack(">IIII", image_bytes[:16])
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if magic != 2051 or label_magic != 2049 or count != label_count or min(count, height, width) <= 0:
        raise ValueError("invalid IDX magic, dimensions, or sample counts")
    if len(image_bytes) != 16+count*height*width or len(label_bytes) != 8+count:
        raise ValueError("IDX byte length mismatch")
    images = np.frombuffer(image_bytes, dtype=np.uint8, offset=16).reshape(count, 1, height, width).astype(dtype)/255
    labels = np.frombuffer(label_bytes, dtype=np.uint8, offset=8).astype(np.int64)
    return images, labels
