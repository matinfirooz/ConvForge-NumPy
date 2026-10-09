"""Create and execute a tutorial notebook with JSON, stdlib, and NumPy only."""
import contextlib
import io
import json
from pathlib import Path
import platform

CELLS = [
    ("markdown", """# ConvForge: CNNs with nothing hidden

**Built for [@matinfirooz](https://github.com/matinfirooz).**

This executed tutorial follows a convolution from individual patches to a
trained classifier, analytical gradients, feature maps, and pixel sensitivity.
All saved numerical outputs are actual runs. Tensor math uses only NumPy.
"""),
    ("code", """from pathlib import Path
import sys, json
root = Path.cwd()
if not (root / "convforge").is_dir():
    root = root.parent
if not (root / "convforge").is_dir():
    raise RuntimeError("Open from the repository root or notebooks directory")
sys.path.insert(0, str(root))
import numpy as np
from convforge import Conv2D, ReLU, MaxPool2D, im2col, col2im, TinyCNN, shape_dataset, CLASS_NAMES, cross_entropy
np.set_printoptions(precision=4, suppress=True)
print("NumPy version:", np.__version__)
"""),
    ("markdown", r"""## 1. Slide one shared kernel over local patches

The layer uses NCHW inputs and OIHW kernels. Its operation is cross-correlation
without a spatial kernel flip. This simple example adds a bias of 10:

$$Y_{y,x}=b+\sum_{i,j}W_{i,j}X_{y+i,x+j}.$$
"""),
    ("code", """x = np.arange(1., 17.).reshape(1, 1, 4, 4)
conv = Conv2D(1, 1, 3, padding="valid", dtype=np.float64)
conv.params["W"][0, 0] = np.array([[1., 0., -1.], [1., 0., -1.], [1., 0., -1.]])
conv.params["b"][0] = 10
out, cache = conv.forward(x, training=True)
print("Input:\\n", x[0, 0])
print("Kernel:\\n", conv.params["W"][0, 0])
print("Output:\\n", out[0, 0])
print("Top-left dot product plus bias:", np.sum(x[0, 0, :3, :3]*conv.params["W"][0, 0])+10)
"""),
    ("markdown", r"""## 2. im2col turns patches into a matrix multiplication

Each row stores one patch. Flattening the shared filter makes the convolution
a matrix product. The explicit patch matrix can occupy extra memory.

$$Y_f=CW_f^T+b.$$
"""),
    ("code", """columns, spec = im2col(x, 3)
print("Patch rows:\\n", columns)
flat_result = columns @ conv.params["W"].reshape(1, -1).T + conv.params["b"]
np.testing.assert_array_equal(flat_result.reshape(2, 2), out[0, 0])
print("Matrix-product agreement: exact")
"""),
    ("markdown", """## 3. Add all overlapping input-gradient contributions

`col2im` is im2col's adjoint, not its inverse. A pixel covered by several
windows receives several gradient contributions.
"""),
    ("code", """z = np.zeros((1, 1, 3, 3), dtype=np.float64)
patches, geometry = im2col(z, 3, padding="same")
counts = col2im(np.ones_like(patches), geometry)
print("Coverage counts:\\n", counts[0, 0])
dx = conv.backward(np.ones_like(out), cache)
print("Gradient of sum(conv output) wrt input:\\n", dx[0, 0])
print("Kernel gradient:\\n", conv.grads["W"][0, 0])
"""),
    ("markdown", """## 4. Verify the derivative independently

Finite differences perturb each input, kernel, and bias value. The tested
case includes two input channels, three output channels, stride, dilation,
and asymmetric same padding.
"""),
    ("code", """from examples.gradient_check import check
print(json.dumps(check(), indent=2))
"""),
    ("markdown", """## 5. Pooling selects a location; backward follows it

Max pooling routes gradients to winning pixels. Ties choose the first
row-major winner. ReLU uses a zero derivative at zero.
"""),
    ("code", """relu = ReLU()
activated, relu_cache = relu.forward(out, training=True)
pool = MaxPool2D(2)
pooled, pool_cache = pool.forward(activated, training=True)
print("Pooled output:", pooled)
pool_gradient = pool.backward(np.ones_like(pooled), pool_cache)
print("Pool gradient:\\n", pool_gradient[0, 0])
print("ReLU gradient:\\n", relu.backward(pool_gradient, relu_cache)[0, 0])
"""),
    ("markdown", """## 6. A real trained model, with every intermediate tensor accessible

The included CNN learned six randomized silhouette classes. It has 19,910
parameters. This is a synthetic image-classification experiment, not an MNIST
accuracy result. Dropout is disabled for inference.
"""),
    ("code", """model = TinyCNN.load(root / "docs/results/shape_cnn.npz")
images, labels = shape_dataset(2, seed=2026)
logits, _, activations = model.forward(images[:1], trace=True)
probabilities = model.predict_proba(images[:1])[0]
print("Parameters:", model.parameter_count)
print("True class:", CLASS_NAMES[int(labels[0])])
print("Predicted class:", CLASS_NAMES[int(probabilities.argmax())])
print("Probabilities:", dict(zip(CLASS_NAMES, probabilities.round(4))))
for name, tensor in activations.items():
    print(name, tensor.shape)
"""),
    ("markdown", """## 7. Pixel sensitivity is an analytical input gradient

The absolute gradient shows local sensitivity of the selected logit to pixel
changes. It does not provide a causal explanation or calibrated confidence.
"""),
    ("code", """selected_class = int(probabilities.argmax())
sensitivity = model.input_gradient(images[:1], selected_class)
print("Pixel-gradient shape:", sensitivity.shape)
print("Pixel-gradient absolute maximum:", float(np.abs(sensitivity).max()))
print("Pixel-gradient L2 norm:", float(np.linalg.norm(sensitivity)))
test = json.loads((root / "docs/results/eval_report.json").read_text())
print("Held-out generated test:", test["correct"], "/", test["examples"], "accuracy", test["accuracy"])
"""),
    ("markdown", """## 8. Train, inspect, and draw

From the repository root:

```bash
python -m examples.train_shapes
python -m examples.evaluate
python -m examples.serve_lab
```

Open the localhost URL printed by the last command. Custom drawing predictions
and pixel gradients run in NumPy on the Python server. The HTML file also has
an offline gallery of precomputed model outputs.

- [Model feature maps](../docs/assets/feature_maps.svg)
- [Training curves](../docs/assets/training.svg)
- [Full forward/backward math guide](../docs/MATH.md)
- [CNN laboratory](../docs/cnn_lab.html)

Notebook editors are optional; all scripts require only NumPy and the Python
standard library. The saved checkpoint contains inference weights and model
configuration, not optimizer state for exact training resumption.
"""),
]


def build():
    namespace = {"__name__": "__notebook__"}
    cells, count = [], 0
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            count += 1
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(compile(source, f"notebook_cell_{count}", "exec"), namespace)
            text = capture.getvalue()
            cell.update(execution_count=count, outputs=[{"output_type": "stream", "name": "stdout",
                                                         "text": text.splitlines(keepends=True)}] if text else [])
        cells.append(cell)
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()}},
        "nbformat": 4, "nbformat_minor": 4}
    path = Path("notebooks/CNN_From_Scratch.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1)+"\n", encoding="utf-8")
    print(f"Created {path}: {len(cells)} cells, {count} executed code cells.")


if __name__ == "__main__":
    build()
