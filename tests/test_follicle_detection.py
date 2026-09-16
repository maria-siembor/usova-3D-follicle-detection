import numpy as np

from follicle_detection import detect_follicle_candidates


def test_detection_respects_ovary_roi_and_minimum_size():
    volume = np.full((9, 9, 9), 100, dtype=np.uint8)
    volume[2:5, 2:5, 2:5] = 0
    volume[6, 6, 6] = 0
    roi = np.ones_like(volume, dtype=bool)
    roi[2:5, 2:5, 2:5] = True
    roi[6, 6, 6] = False

    labels = detect_follicle_candidates(
        volume, roi, voxel_vol_mm3=1.0, smooth_sigma=0, open_iterations=0,
        min_diameter_mm=2.0,
    )

    assert labels.max() == 1
    assert labels[3, 3, 3] == 1
    assert labels[6, 6, 6] == 0