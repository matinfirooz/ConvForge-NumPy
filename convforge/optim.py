"""Stable classification losses, global gradient clipping, and Adam."""
import numpy as np
from .ops import floating


def softmax(logits):
    floating(logits, 2)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp/exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    if not isinstance(targets, np.ndarray) or targets.shape != (logits.shape[0],):
        raise ValueError("targets must have shape (batch,)")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must be integers")
    probabilities = softmax(logits)
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("target out of range")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    loss = np.mean(np.log(np.exp(shifted).sum(axis=-1))-shifted[np.arange(len(targets)), targets])
    gradient = probabilities.copy()
    gradient[np.arange(len(targets)), targets] -= 1
    return float(loss), gradient/len(targets)


def clip_gradients(grads, max_norm=5.0):
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be positive and finite")
    if not all(np.isfinite(g).all() for g in grads.values()):
        raise ValueError("gradients must be finite")
    norm = float(np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in grads.values())))
    if norm > max_norm:
        for gradient in grads.values():
            gradient *= max_norm/(norm+1e-12)
    return norm


class Adam:
    def __init__(self, params, *, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("learning rate and epsilon must be positive and finite")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0,1)")
        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps
        self.m = {name: np.zeros_like(p) for name, p in params.items()}
        self.v = {name: np.zeros_like(p) for name, p in params.items()}
        self.step_count = 0

    def step(self, grads):
        if grads.keys() != self.params.keys():
            raise ValueError("gradient keys must match parameters")
        for name, g in grads.items():
            if g.shape != self.params[name].shape or g.dtype != self.params[name].dtype:
                raise ValueError(f"gradient shape/dtype mismatch: {name}")
            if not np.isfinite(g).all():
                raise ValueError("gradient contains nonfinite values")
        self.step_count += 1
        for name, p in self.params.items():
            g = grads[name]
            self.m[name] *= self.beta1
            self.m[name] += (1-self.beta1)*g
            self.v[name] *= self.beta2
            self.v[name] += (1-self.beta2)*g*g
            m = self.m[name]/(1-self.beta1**self.step_count)
            v = self.v[name]/(1-self.beta2**self.step_count)
            p -= self.lr*m/(np.sqrt(v)+self.eps)
