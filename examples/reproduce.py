"""One command for tests, training, held-out evaluation, gradients, and visuals."""
import json
import os
from pathlib import Path
import platform
import sys
import unittest


def main():
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    if not result.wasSuccessful():
        sys.exit(1)
    from examples.train_shapes import train
    from examples.evaluate import evaluate
    from examples.gradient_check import main as gradients
    from examples.build_showcase import build as showcase
    from examples.build_notebook import build as notebook
    train()
    evaluate()
    gradients()
    showcase()
    notebook()
    report = {"tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "successful": result.wasSuccessful(), "python": platform.python_version(),
              "command": "python -m examples.reproduce"}
    (root/"docs/results/verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Reproduction complete. Run python -m examples.serve_lab to use the live drawing lab.")


if __name__ == "__main__":
    main()
