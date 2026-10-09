"""Finite-difference the actual convolution input, kernel, and bias gradients."""
import json
from pathlib import Path
import numpy as np
from convforge import Conv2D


def numerical_gradient(array, objective, epsilon=1e-6):
    numeric = np.empty_like(array)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original+epsilon
        plus = objective()
        array[index] = original-epsilon
        minus = objective()
        array[index] = original
        numeric[index] = (plus-minus)/(2*epsilon)
    return numeric


def check(seed=7, epsilon=1e-6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, 2, 4, 5))
    layer = Conv2D(2, 3, (2, 3), stride=(2, 1), padding="same",
                   dilation=(1, 2), seed=seed, dtype=np.float64)
    output, cache = layer.forward(x, training=True)
    upstream = rng.normal(size=output.shape)
    dx = layer.backward(upstream, cache)
    objective = lambda: float(np.sum(layer.forward(x)[0]*upstream))
    errors = {}
    for name, parameter, grad in [("input", x, dx), ("kernel", layer.params["W"], layer.grads["W"]),
                                  ("bias", layer.params["b"], layer.grads["b"])]:
        numeric = numerical_gradient(parameter, objective, epsilon)
        errors[name] = {"max_absolute_error": float(np.max(np.abs(numeric-grad))),
                        "relative_l2_error": float(np.linalg.norm(numeric-grad)/max(np.linalg.norm(numeric)+np.linalg.norm(grad), 1e-12))}
        np.testing.assert_allclose(grad, numeric, rtol=2e-5, atol=2e-8)
    return {"passed": True, "dtype": "float64", "seed": seed, "epsilon": epsilon,
            "input_shape": list(x.shape), "kernel_shape": list(layer.params["W"].shape),
            "stride": [2, 1], "dilation": [1, 2], "padding": "same", "errors": errors}


def main():
    report = check()
    path = Path("docs/results/gradient_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
