"""Local web application for interactive follicle detection and measurement."""

import sys
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_pipeline  # noqa: E402
from vtk_io import read_vtk  # noqa: E402


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
MODEL_PATH = ROOT / "data" / "final_ovary_model.joblib"
RESULTS = {}


def get_model():
    if not hasattr(get_model, "model"):
        get_model.model = joblib.load(MODEL_PATH)
    return get_model.model


def serialise_measurements(measurements):
    return [
        {**measurement, "centroid_zyx": list(measurement["centroid_zyx"])}
        for measurement in measurements
    ]


def slice_payload(volume, labels, index):
    image = volume[index].astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return {
        "index": index,
        "image": np.rint(image * 255).astype(np.uint8).tolist(),
        "labels": labels[index].astype(np.uint16).tolist(),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyse")
def analyse():
    uploaded = request.files.get("volume")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a VTK volume before analysing."}), 400
    if not uploaded.filename.lower().endswith(".vtk"):
        return jsonify({"error": "The current reader expects a legacy .vtk volume."}), 400

    try:
        volume, spacing, _ = read_vtk(uploaded.read())
        result = run_pipeline(volume, spacing, get_model())
    except Exception as error:
        app.logger.exception("Pipeline failed")
        return jsonify({"error": f"Analysis failed: {error}"}), 422

    result_id = uuid.uuid4().hex
    RESULTS[result_id] = {"volume": volume, "labels": result["follicle_labels"]}
    measurements = serialise_measurements(result["measurements"])
    return jsonify({
        "result_id": result_id,
        "shape": list(volume.shape),
        "spacing": list(spacing),
        "summary": result["summary"],
        "measurements": measurements,
        "first_slice": slice_payload(volume, result["follicle_labels"], volume.shape[0] // 2),
    })


@app.get("/api/results/<result_id>/slice/<int:index>")
def get_slice(result_id, index):
    result = RESULTS.get(result_id)
    if result is None:
        return jsonify({"error": "This analysis is no longer available."}), 404
    if index < 0 or index >= result["volume"].shape[0]:
        return jsonify({"error": "Slice index is outside the volume."}), 400
    return jsonify(slice_payload(result["volume"], result["labels"], index))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)