"""Generate readable, exact SVGs and a portable CNN laboratory HTML file."""
import json
from pathlib import Path
import numpy as np
from convforge import TinyCNN, shape_dataset, CLASS_NAMES
from convforge.inspect import inspect_image
from convforge.visuals import SVG, hero, curves, image, normalized, INK, MUTED, MINT, ORANGE


def build():
    docs = Path("docs")
    assets = docs/"assets"
    assets.mkdir(parents=True, exist_ok=True)
    model = TinyCNN.load(docs/"results/shape_cnn.npz")
    history = json.loads((docs/"results/train_report.json").read_text())["history"]
    hero(assets/"hero.svg")
    curves(history, assets/"training.svg")
    x, y = shape_dataset(12, seed=2026)
    samples = []
    for label in range(6):
        index = int(np.flatnonzero(y == label)[0])
        record = inspect_image(model, x[index, 0])
        record["true_label"] = CLASS_NAMES[label]
        samples.append(record)
    data = {"class_names": list(CLASS_NAMES), "size": model.config["input_size"],
            "parameters": model.parameter_count, "samples": samples,
            "filters": [normalized(f[0]).tolist() for f in model.layers[0][1].params["W"]],
            "note": "First generated example of each class from a separate seed; predictions and maps computed by the trained NumPy CNN."}
    (docs/"results/lab_data.json").write_text(json.dumps(data, separators=(",", ":"))+"\n", encoding="utf-8")
    template = (docs/"lab_template.html").read_text(encoding="utf-8")
    (docs/"cnn_lab.html").write_text(template.replace("__LAB_DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    gallery_figure(samples, assets/"dataset.svg")
    feature_figure(samples[2], assets/"feature_maps.svg")
    filters_figure(data["filters"], assets/"filters.svg")
    print("Generated CNN laboratory and five SVG showcase assets.")
    return data


def gallery_figure(samples, path):
    svg = SVG(1150, 288, "Six generated shape classes with actual CNN predictions")
    svg.text(32, 41, "Six shapes. Thousands of variations.", 25, INK, 700)
    svg.text(32, 70, "Random position, scale, rotation, brightness, and pixel noise · generated entirely in NumPy", 13, MUTED)
    for i, record in enumerate(samples):
        x = 37+i*185
        image(svg, np.array(record["input"]), x, 94, 137)
        svg.text(x+68, 251, record["true_label"], 13, INK, 600, "middle")
        svg.text(x+68, 273, "pred: "+record["prediction_name"], 11, MINT, 400, "middle")
    svg.save(path)


def feature_figure(record, path):
    svg = SVG(1120, 388, "Actual convolutional feature maps and input pixel sensitivity")
    svg.text(32, 40, "Follow an image through learned filters", 25, INK, 700)
    svg.text(32, 70, "Per-map display scaling · outputs come from the included trained checkpoint", 13, MUTED)
    image(svg, np.array(record["input"]), 35, 119, 172)
    svg.text(35, 105, "INPUT", 12, MINT, 700)
    svg.text(35, 318, record["true_label"], 16, INK, 600)
    svg.text(35, 342, "prediction: "+record["prediction_name"], 12, MUTED)
    svg.text(247, 105, "CONV 1 + ReLU", 12, MINT, 700)
    for i in range(4):
        xx, yy = 247+(i%2)*111, 119+(i//2)*111
        image(svg, np.array(record["conv1_maps"][i]), xx, yy, 98, palette="mint")
    svg.text(518, 105, "CONV 2 + ReLU", 12, MINT, 700)
    for i in range(4):
        xx, yy = 518+(i%2)*111, 119+(i//2)*111
        image(svg, np.array(record["conv2_maps"][i]), xx, yy, 98, palette="mint")
    svg.text(832, 105, "PIXEL SENSITIVITY", 12, ORANGE, 700)
    image(svg, np.array(record["sensitivity"]), 832, 119, 172, palette="warm")
    svg.text(832, 315, "|d class logit / d pixel|", 12, MUTED)
    svg.text(32, 370, "Sensitivity measures local model response; it does not establish a causal explanation.", 12, MUTED)
    svg.save(path)


def filters_figure(filters, path):
    svg = SVG(1110, 242, "Eight learned 3 by 3 first-layer convolution kernels")
    svg.text(32, 40, "The filters are learned, not hand-coded", 24, INK, 700)
    svg.text(32, 68, "First convolution layer · eight 3 × 3 kernels · independently scaled for display", 13, MUTED)
    for i, array in enumerate(filters):
        x = 35+i*132
        image(svg, np.array(array), x, 97, 95, palette="mint")
        svg.text(x+47, 219, f"filter {i}", 12, MUTED, 400, "middle")
    svg.save(path)


if __name__ == "__main__":
    build()
