"""Small sequential CNN with explicit caches, inspection, and safe checkpoints."""
from pathlib import Path
import json
import numpy as np
from .layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Dropout
from .optim import cross_entropy, softmax
from .ops import floating


class TinyCNN:
    def __init__(self, *, input_size=24, num_classes=6, channels=(8, 16), hidden=32,
                 dropout=0.1, seed=42, dtype="float32", class_names=None):
        if not isinstance(input_size, int) or input_size < 4:
            raise ValueError("input_size must be an integer >=4")
        if len(channels) != 2:
            raise ValueError("provide two convolution channel counts")
        dtype = np.dtype(dtype)
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        if len(class_names) != num_classes or not all(isinstance(n, str) for n in class_names):
            raise ValueError("class_names must contain one string per class")
        self.config = {"input_size": input_size, "num_classes": num_classes,
                       "channels": list(channels), "hidden": hidden, "dropout": dropout,
                       "seed": seed, "dtype": str(dtype), "class_names": list(class_names)}
        flat_features = channels[1]*(input_size//4)**2
        self.layers = [
            ("conv1", Conv2D(1, channels[0], 3, seed=seed, dtype=dtype)),
            ("relu1", ReLU()), ("pool1", MaxPool2D(2)),
            ("conv2", Conv2D(channels[0], channels[1], 3, seed=seed+1, dtype=dtype)),
            ("relu2", ReLU()), ("pool2", MaxPool2D(2)), ("flatten", Flatten()),
            ("dense1", Dense(flat_features, hidden, seed=seed+2, dtype=dtype)),
            ("relu3", ReLU()), ("dropout", Dropout(dropout, seed=seed+3)),
            ("classifier", Dense(hidden, num_classes, seed=seed+4, dtype=dtype)),
        ]

    def parameters(self):
        return {f"{name}.{key}": value for name, layer in self.layers
                for key, value in getattr(layer, "params", {}).items()}

    def gradients(self):
        return {f"{name}.{key}": value for name, layer in self.layers
                for key, value in getattr(layer, "grads", {}).items()}

    @property
    def parameter_count(self):
        return sum(p.size for p in self.parameters().values())

    def forward(self, x, *, training=False, trace=False):
        floating(x, 4)
        expected = (1, self.config["input_size"], self.config["input_size"])
        if x.shape[1:] != expected or x.dtype != np.dtype(self.config["dtype"]):
            raise ValueError(f"expected (N,{expected}) and dtype {self.config['dtype']}")
        caches, activations = [], {"input": x} if trace else None
        for name, layer in self.layers:
            x, cache = layer.forward(x, training=training)
            if training:
                caches.append(cache)
            if trace:
                activations[name] = x
        return x, tuple(caches) if training else None, activations

    def backward(self, d_logits, caches):
        if caches is None or len(caches) != len(self.layers):
            raise ValueError("use caches from forward(training=True)")
        gradient = d_logits
        for (_, layer), cache in zip(reversed(self.layers), reversed(caches)):
            gradient = layer.backward(gradient, cache)
        return gradient

    def loss_and_grads(self, x, targets):
        logits, caches, _ = self.forward(x, training=True)
        loss, d_logits = cross_entropy(logits, targets)
        self.backward(d_logits, caches)
        return loss, float(np.mean(logits.argmax(axis=-1) == targets)), self.gradients()

    def predict_proba(self, x):
        return softmax(self.forward(x)[0])

    def input_gradient(self, x, class_index):
        """Gradient of a chosen class logit wrt pixels; dropout is disabled.

        This is sensitivity, not proof of a causal explanation. Parameter
        gradients are overwritten as a side effect; weights are unchanged.
        """
        if not isinstance(class_index, (int, np.integer)) or not 0 <= class_index < self.config["num_classes"]:
            raise ValueError("class_index out of range")
        gradient_caches, output = [], x
        # Request backward caches while explicitly bypassing stochastic dropout.
        for _, layer in self.layers:
            if isinstance(layer, Dropout):
                cache = np.ones_like(output)
            else:
                output, cache = layer.forward(output, training=True)
            gradient_caches.append(cache)
        upstream = np.zeros_like(output)
        upstream[:, class_index] = 1
        return self.backward(upstream, tuple(gradient_caches))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.parameters(), config=np.array(json.dumps(self.config)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            model = cls(**json.loads(str(stored["config"])))
            for name, parameter in model.parameters().items():
                value = stored[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype or not np.isfinite(value).all():
                    raise ValueError(f"invalid checkpoint parameter: {name}")
                parameter[...] = value
        return model
