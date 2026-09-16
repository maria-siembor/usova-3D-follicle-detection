from pipeline import classify_maturity, measure_follicles, summarize_patient
import numpy as np


def test_classify_maturity_boundaries():
    assert classify_maturity(15.99) == "immature"
    assert classify_maturity(16.0) == "optimal_for_retrieval"
    assert classify_maturity(22.0) == "optimal_for_retrieval"
    assert classify_maturity(22.01) == "post_mature"


def test_measurements_use_voxel_volume_and_centroid():
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 2
    measurements = measure_follicles(labels, voxel_vol_mm3=8.0)

    assert [item["follicle_id"] for item in measurements] == [1, 2]
    assert all(item["volume_mm3"] == 8.0 for item in measurements)
    assert measurements[0]["centroid_zyx"] == (0.0, 0.0, 0.0)
    assert measurements[1]["centroid_zyx"] == (1.0, 1.0, 1.0)


def test_summary_counts_each_maturity_class():
    measurements = [
        {"maturity": "immature"},
        {"maturity": "optimal_for_retrieval"},
        {"maturity": "post_mature"},
        {"maturity": "optimal_for_retrieval"},
    ]

    assert summarize_patient(measurements) == {
        "total_follicle_count": 4,
        "optimal_for_retrieval_count": 2,
        "immature_count": 1,
        "post_mature_count": 1,
    }
