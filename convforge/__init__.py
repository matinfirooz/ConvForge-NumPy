"""ConvForge: transparent NumPy convolution, backpropagation, and learning."""
from .layers import Conv2D, Dense, ReLU, MaxPool2D, GlobalAveragePool2D, Flatten, Dropout
from .ops import im2col, col2im, naive_conv2d
from .optim import Adam, cross_entropy, softmax, clip_gradients
from .model import TinyCNN
from .data import shape_dataset, load_idx, batches, CLASS_NAMES

__version__ = "1.0.0"
__all__ = ["Conv2D", "Dense", "ReLU", "MaxPool2D", "GlobalAveragePool2D", "Flatten", "Dropout",
           "im2col", "col2im", "naive_conv2d", "Adam", "cross_entropy", "softmax",
           "clip_gradients", "TinyCNN", "shape_dataset", "load_idx", "batches", "CLASS_NAMES"]
