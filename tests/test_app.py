import io

import numpy as np

import app
from vtk_io import write_vtk_labels


def test_homepage_and_validation_errors():
    client = app.app.test_client()

    assert client.get("/").status_code == 200
    models = client.get("/api/models").get_json()
    assert set(models) == {"classical", "deep_learning"}
    assert client.post("/api/analyse").status_code == 400
    response = client.post(
        "/api/analyse",
        data={"volume": (io.BytesIO(b"not vtk"), "volume.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_analyse_returns_summary_and_slice(monkeypatch, tmp_path):
    volume = np.zeros((2, 3, 4), dtype=np.uint8)
    labels = np.zeros_like(volume)
    labels[0, 1, 1] = 1
    vtk_path = tmp_path / "volume.vtk"
    write_vtk_labels(volume, (1.0, 1.0, 1.0), vtk_path)

    monkeypatch.setattr(app, "get_model", lambda: object())
    monkeypatch.setattr(app, "run_pipeline", lambda volume, spacing, model: {
        "follicle_labels": labels,
        "measurements": [{
            "follicle_id": 1,
            "volume_mm3": 1.0,
            "equivalent_diameter_mm": 1.24,
            "centroid_zyx": (0.0, 1.0, 1.0),
            "maturity": "immature",
        }],
        "summary": {
            "total_follicle_count": 1,
            "optimal_for_retrieval_count": 0,
            "immature_count": 1,
            "post_mature_count": 0,
        },
    })

    client = app.app.test_client()
    with vtk_path.open("rb") as volume_file:
        response = client.post(
            "/api/analyse",
            data={"volume": (volume_file, "volume.vtk")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["total_follicle_count"] == 1
    assert payload["measurements"][0]["centroid_zyx"] == [0.0, 1.0, 1.0]
    slice_response = client.get(f"/api/results/{payload['result_id']}/slice/0")
    assert slice_response.status_code == 200
    assert slice_response.get_json()["labels"][1][1] == 1
