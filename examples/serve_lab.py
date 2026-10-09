"""Serve the drawing lab locally; all CNN inference and gradients use NumPy."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
from convforge import TinyCNN
from convforge.inspect import inspect_image


def make_server(*, port=8765, checkpoint="docs/results/shape_cnn.npz"):
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(checkpoint)
    checkpoint = checkpoint if checkpoint.is_absolute() else root/checkpoint
    model = TinyCNN.load(checkpoint)
    if (model.config["input_size"] != 24 or model.config["num_classes"] != 6
            or model.config["class_names"] != ["circle", "square", "triangle", "plus", "cross", "stripes"]
            or model.config["channels"] != [8, 16]):
        raise ValueError("this drawing lab expects the included 24x24 six-class shape model")
    source = (root/"docs/cnn_lab.html").read_text(encoding="utf-8")
    pattern = r'(<script id="lab-data" type="application/json">)(.*?)(</script>)'
    match = re.search(pattern, source, re.DOTALL)
    if match is None:
        raise ValueError("laboratory data missing; run python -m examples.build_showcase")
    data = json.loads(match.group(2))
    refreshed = []
    for sample in data["samples"]:
        result = inspect_image(model, sample["input"])
        result["true_label"] = sample["true_label"]
        refreshed.append(result)
    data["samples"], data["parameters"] = refreshed, model.parameter_count
    # A retrained checkpoint can differ from the packaged gallery. In live
    # mode, refresh gallery predictions/maps with the checkpoint loaded now.
    source = source[:match.start(2)]+json.dumps(data, separators=(",", ":"))+source[match.end(2):]
    document = source.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json; charset=utf-8"):
            body = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, document, "text/html; charset=utf-8")
            elif self.path == "/api/metadata":
                self.send(200, {"size": 24, "class_names": model.config["class_names"],
                                "parameters": model.parameter_count, "backend": "NumPy"})
            else:
                self.send(404, {"error": "Unknown path"})

        def do_POST(self):
            if self.path != "/api/predict":
                self.send(404, {"error": "Unknown path"})
                return
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 128*1024:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(count))
                result = inspect_image(model, payload["pixels"])
            except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
                self.send(400, {"error": str(error)})
                return
            self.send(200, result)

        def log_message(self, format, *args):
            pass

    # Local host only. Sequential handling avoids races while sensitivity updates
    # reusable gradient buffers. Inference itself never changes model weights.
    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default="docs/results/shape_cnn.npz")
    server = make_server(**vars(parser.parse_args()))
    print(f"Open http://127.0.0.1:{server.server_port} · Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
