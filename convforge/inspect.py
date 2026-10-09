"""Export actual model activations and class-logit pixel sensitivity."""
import numpy as np
from .optim import softmax
from .visuals import normalized


def inspect_image(model, image):
    size = model.config["input_size"]
    image = np.asarray(image, dtype=model.config["dtype"])
    if image.shape != (size, size) or not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
        raise ValueError(f"pixels must be a finite {size}x{size} array in [0,1]")
    x = image[None, None]
    logits, _, trace = model.forward(x, trace=True)
    probabilities = softmax(logits)[0]
    prediction = int(probabilities.argmax())
    sensitivity = model.input_gradient(x, prediction)[0, 0]
    return {"input": image.tolist(), "logits": logits[0].tolist(),
            "probabilities": probabilities.tolist(), "prediction": prediction,
            "prediction_name": model.config["class_names"][prediction],
            "conv1_maps": [normalized(a).tolist() for a in trace["relu1"][0]],
            "conv2_maps": [normalized(a).tolist() for a in trace["relu2"][0]],
            "sensitivity": normalized(sensitivity, absolute=True).tolist(),
            "shapes": {name: list(a.shape) for name, a in trace.items()},
            "normalization_note": "Each displayed map independently min-max scaled. Sensitivity is absolute class-logit gradient, not a causal explanation."}
