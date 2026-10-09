"""Train the same NumPy CNN from user-provided local MNIST IDX files."""
import argparse
import json
from pathlib import Path
import numpy as np
from convforge import TinyCNN, Adam, clip_gradients, load_idx, batches
from examples.train_shapes import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="local train-images-idx3-ubyte[.gz]")
    parser.add_argument("--labels", required=True, help="local train-labels-idx1-ubyte[.gz]")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=12000, help="number used including a 10% validation holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/mnist")
    args = parser.parse_args()
    if args.epochs <= 0 or args.limit < 20:
        parser.error("epochs must be positive and limit >=20")
    x, y = load_idx(args.images, args.labels)
    if x.shape[-2:] != (28, 28) or (y > 9).any():
        raise ValueError("expected MNIST 28x28 images and digit labels 0..9")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y))[:args.limit]
    x, y = x[order], y[order]
    if len(y) < 20:
        raise ValueError("need at least 20 examples for a train/validation split")
    split = int(len(y)*0.9)
    train_x, train_y, val_x, val_y = x[:split], y[:split], x[split:], y[split:]
    model = TinyCNN(input_size=28, num_classes=10, seed=args.seed, class_names=[str(i) for i in range(10)])
    optimizer = Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(1, args.epochs+1):
        for bx, by in batches(train_x, train_y, rng=rng):
            _, _, grads = model.loss_and_grads(bx, by)
            clip_gradients(grads)
            optimizer.step(grads)
        loss, accuracy, _ = evaluate(model, val_x, val_y)
        history.append({"epoch": epoch, "validation_loss": loss, "validation_accuracy": accuracy})
        print(f"epoch {epoch} | validation accuracy {accuracy:.2%}", flush=True)
    destination = Path(args.output)
    model.save(destination/"mnist_cnn.npz")
    (destination/"report.json").write_text(json.dumps({"history": history,
        "train_examples": len(train_y), "validation_examples": len(val_y), "seed": args.seed,
        "note": "Local IDX data, random train/validation split; no official test set evaluated."}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
