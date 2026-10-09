"""Train a real CNN on generated shape images using analytical gradients."""
import argparse
import csv
import json
import platform
from pathlib import Path
import time
import numpy as np
from convforge import TinyCNN, Adam, cross_entropy, clip_gradients, shape_dataset, batches, CLASS_NAMES


def evaluate(model, x, y):
    total_loss, correct, predictions = 0.0, 0, []
    for bx, by in batches(x, y, batch_size=128):
        logits = model.forward(bx)[0]
        loss, _ = cross_entropy(logits, by)
        predicted = logits.argmax(axis=-1)
        correct += int(np.sum(predicted == by))
        total_loss += loss*len(by)
        predictions.append(predicted)
    return total_loss/len(y), correct/len(y), np.concatenate(predictions)


def train(*, epochs=12, per_class=300, batch_size=64, seed=42,
          output="docs/results"):
    if epochs <= 0 or per_class <= 0 or batch_size <= 0:
        raise ValueError("epochs, per_class, batch_size must be positive")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    train_x, train_y = shape_dataset(per_class, seed=seed)
    val_x, val_y = shape_dataset(60, seed=1337)
    model = TinyCNN(seed=seed, class_names=CLASS_NAMES)
    initial_parameters = {name: p.copy() for name, p in model.parameters().items()}
    optimizer = Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed+1)
    history = []
    started = time.perf_counter()
    loss, accuracy, _ = evaluate(model, val_x, val_y)
    history.append({"epoch": 0, "train_loss": None, "train_accuracy": None,
                    "val_loss": loss, "val_accuracy": accuracy})
    print(f"epoch 00 | validation loss {loss:.4f} | validation accuracy {accuracy:.2%}", flush=True)
    for epoch in range(1, epochs+1):
        total_loss, train_correct = 0.0, 0.0
        for bx, by in batches(train_x, train_y, batch_size=batch_size, rng=rng):
            batch_loss, batch_accuracy, grads = model.loss_and_grads(bx, by)
            clip_gradients(grads, 5.0)
            optimizer.step(grads)
            total_loss += batch_loss*len(by)
            train_correct += batch_accuracy*len(by)
        loss, accuracy, _ = evaluate(model, val_x, val_y)
        history.append({"epoch": epoch, "train_loss": total_loss/len(train_y),
                        "train_accuracy": train_correct/len(train_y),
                        "val_loss": loss, "val_accuracy": accuracy})
        print(f"epoch {epoch:02d} | train loss {history[-1]['train_loss']:.4f} | validation loss {loss:.4f} | validation accuracy {accuracy:.2%}", flush=True)
    elapsed = time.perf_counter()-started
    model.save(destination/"shape_cnn.npz")
    restored = TinyCNN.load(destination/"shape_cnn.npz")
    np.testing.assert_array_equal(model.predict_proba(val_x[:8]), restored.predict_proba(val_x[:8]))
    report = {"task": "generated six-class silhouette classification", "seed": seed,
              "validation_seed": 1337, "epochs": epochs, "train_examples": len(train_y),
              "validation_examples": len(val_y), "batch_size": batch_size,
              "parameters": model.parameter_count, "optimizer_steps": optimizer.step_count,
              "model_config": model.config, "history": history, "initial": history[0],
              "final": history[-1], "elapsed_seconds": elapsed,
              "python": platform.python_version(), "numpy": np.__version__,
              "checkpoint_reload_exact": True,
              "parameter_change_l2": {name: float(np.linalg.norm(p-initial_parameters[name])) for name, p in model.parameters().items()},
              "note": "Balanced synthetic images with random transformations/noise. Fixed validation monitors training only. Last-epoch checkpoint; single-seed teaching experiment."}
    (destination/"train_report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (destination/"training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved {destination}; {model.parameter_count:,} parameters; {elapsed:.2f} seconds.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/results")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
