"""Evaluate a saved shape CNN on a separate generated test sample."""
import argparse
import json
from pathlib import Path
import numpy as np
from convforge import TinyCNN, shape_dataset, CLASS_NAMES
from examples.train_shapes import evaluate as score
from convforge.visuals import confusion_figure


def evaluate(*, checkpoint="docs/results/shape_cnn.npz", per_class=100,
             seed=2026, noise=0.065, output="docs/results/eval_report.json"):
    model = TinyCNN.load(checkpoint)
    if model.config["num_classes"] != len(CLASS_NAMES):
        raise ValueError("this command evaluates the six-class shape checkpoint")
    x, y = shape_dataset(per_class, seed=seed, noise=noise,
                         size=model.config["input_size"], dtype=np.dtype(model.config["dtype"]))
    loss, accuracy, predicted = score(model, x, y)
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    report = {"task": "synthetic silhouette classification", "test_seed": seed,
              "examples": len(y), "noise_standard_deviation": noise,
              "checkpoint": str(checkpoint), "accuracy": accuracy,
              "correct": int(np.sum(y == predicted)), "cross_entropy": loss,
              "class_names": list(CLASS_NAMES), "confusion_matrix": matrix.tolist(),
              "per_class_accuracy": (matrix.diagonal()/matrix.sum(axis=1)).tolist(),
              "note": "Independent random test images. One teaching checkpoint; not an MNIST or real-world accuracy claim."}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    confusion_figure(matrix, CLASS_NAMES, path.parent.parent/"assets"/"confusion.svg")
    print(f"Held-out generated test: {report['correct']}/{len(y)}, accuracy {accuracy:.2%}, cross entropy {loss:.5f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="docs/results/shape_cnn.npz")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise", type=float, default=0.065)
    parser.add_argument("--output", default="docs/results/eval_report.json")
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
