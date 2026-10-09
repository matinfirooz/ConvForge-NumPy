"""Precise SVG figures using the standard library; no plotting dependency."""
from html import escape
from pathlib import Path
import numpy as np

BG, PANEL, INK, MUTED = "#0a101c", "#151f30", "#edf2fa", "#98aac0"
MINT, ORANGE, PURPLE = "#69e3bf", "#ffb478", "#beaaff"


class SVG:
    def __init__(self, width, height, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
                      f'<title>{escape(title)}</title>', f'<rect width="{width}" height="{height}" rx="18" fill="{BG}"/>']

    def rect(self, x, y, width, height, fill=PANEL, radius=7, **extra):
        attrs = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in extra.items())
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" {attrs}/>')

    def text(self, x, y, value, size=16, fill=INK, weight=400, anchor="start", **extra):
        attrs = " ".join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in extra.items())
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" {attrs}>{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=MUTED, width=1):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')

    def curve(self, points, color=MINT):
        values = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polyline points="{values}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts)+"\n</svg>\n", encoding="utf-8")


def normalized(array, *, absolute=False):
    a = np.abs(array) if absolute else np.asarray(array)
    low, high = float(a.min()), float(a.max())
    return np.zeros_like(a) if high == low else (a-low)/(high-low)


def image(svg, array, x, y, size, *, palette="gray"):
    a = np.clip(array, 0, 1)
    height, width = a.shape
    cell = size/max(height, width)
    for i in range(height):
        for j in range(width):
            value = float(a[i, j])
            if palette == "mint":
                low, high = (15, 29, 43), (105, 227, 191)
            elif palette == "warm":
                low, high = (20, 25, 40), (255, 180, 120)
            else:
                low, high = (12, 18, 30), (235, 242, 250)
            fill = "#"+"".join(f"{round(l+(h-l)*value):02x}" for l, h in zip(low, high))
            svg.rect(x+j*cell, y+i*cell, cell+0.02, cell+0.02, fill, 0)


def hero(path):
    svg = SVG(1280, 485, "ConvForge: a trainable CNN implemented entirely in NumPy")
    svg.text(48, 46, "MATINFIROOZ / CONVFORGE-NUMPY", 14, MINT, 700, letter_spacing=2)
    svg.text(45, 118, "A CNN. Every operation exposed.", 44, INK, 700)
    svg.text(48, 158, "Pixels → learned filters → feature maps → predictions → manual gradients", 19, MUTED)
    titles = [("INPUT", "1 × 24 × 24", "noisy silhouettes"),
              ("CONV + POOL", "8 × 12 × 12", "learn local patterns"),
              ("CONV + POOL", "16 × 6 × 6", "combine features"),
              ("CLASSIFIER", "32 → 6", "choose a shape")]
    for i, (title, shape, caption) in enumerate(titles):
        x = 48+i*305
        svg.rect(x, 209, 277, 143)
        svg.text(x+18, 237, f"0{i+1} / {title}", 13, MINT, 700)
        svg.text(x+18, 282, shape, 24, INK, 700)
        svg.text(x+18, 321, caption, 15, MUTED)
        if i < 3:
            svg.line(x+280, 279, x+301, 279, MINT, 2)
            svg.curve([(x+295, 274), (x+301, 279), (x+295, 284)], MINT)
    svg.text(48, 406, "NUMPY ONLY", 14, MINT, 700)
    svg.text(267, 406, "MANUAL BACKPROP", 14, ORANGE, 700)
    svg.text(560, 406, "LIVE DRAWING LAB", 14, PURPLE, 700)
    svg.text(898, 406, "19,910 PARAMETERS", 14, MINT, 700)
    svg.text(48, 452, "Train it. Trace it. Differentiate it. See what its filters learn.", 16, MUTED)
    svg.save(path)


def curves(history, path):
    svg = SVG(1100, 415, "Measured CNN validation accuracy and cross entropy during training")
    svg.text(34, 41, "Pixels become a decision rule", 25, INK, 700)
    svg.text(34, 68, "Generated shapes · fixed validation set · analytical gradients · last-epoch checkpoint", 14, MUTED)
    for field, title, x0, color, ymax in (("val_accuracy", "Validation accuracy", 66, MINT, 1.0),
                                        ("val_loss", "Validation cross entropy", 620, ORANGE,
                                         max(h["val_loss"] for h in history)*1.1)):
        svg.text(x0, 109, title, 17, INK, 600)
        top, bottom, width = 133, 337, 410
        for ratio in (0, 0.25, 0.5, 0.75, 1):
            yy = bottom-(bottom-top)*ratio
            svg.line(x0, yy, x0+width, yy, "#28394c")
            svg.text(x0-10, yy+4, f"{ratio*ymax:.2f}", 11, MUTED, anchor="end")
        last = max(h["epoch"] for h in history)
        svg.curve([(x0+width*h["epoch"]/last, bottom-(bottom-top)*h[field]/ymax) for h in history], color)
        for ratio in (0, 0.5, 1):
            svg.text(x0+width*ratio, 366, int(last*ratio), 12, MUTED, anchor="middle")
        svg.text(x0+width/2, 396, "epochs", 12, MUTED, anchor="middle")
    svg.save(path)


def confusion_figure(matrix, labels, path):
    svg = SVG(850, 545, "Held-out synthetic test confusion matrix for the trained NumPy CNN")
    svg.text(32, 42, "A held-out test, class by class", 25, INK, 700)
    svg.text(32, 73, "Rows = true class · columns = predicted class · independent test seed", 14, MUTED)
    startx, starty, cell = 190, 124, 60
    maximum = max(int(matrix.max()), 1)
    for j, label in enumerate(labels):
        svg.text(startx+j*cell+28, starty-14, label, 11, MUTED, anchor="middle")
    for i, label in enumerate(labels):
        svg.text(startx-14, starty+i*cell+35, label, 12, MUTED, anchor="end")
        for j in range(len(labels)):
            value = int(matrix[i, j])
            ratio = value/maximum
            fill = "#"+"".join(f"{round(l+(h-l)*ratio):02x}" for l, h in zip((18, 33, 47), (105, 227, 191)))
            svg.rect(startx+j*cell, starty+i*cell, cell-3, cell-3, fill, 5)
            svg.text(startx+j*cell+28, starty+i*cell+35, value, 17, BG if ratio > 0.5 else INK, 600, "middle")
    svg.text(32, 520, f"Correct: {int(np.trace(matrix))}/{int(matrix.sum())} · accuracy: {np.trace(matrix)/matrix.sum():.2%}", 16, MINT, 600)
    svg.save(path)
