"""Follicle candidate detection: threshold within ovary ROI, morphological
opening to break speckle-noise bridges between adjacent follicles,
minimum-size filter. No maximum-size filter (genuine large cysts exist
in this data, both experts agreed on structures up to ~140mm diameter,
see RESULTS_LOG.md).

Parameter grid explored during development (informal, on all 16
volumes, DO NOT use these numbers as the final reported result, that
would be parameter selection on the test set):
  threshold=20, no smoothing, no opening: sens=0.637, prec=0.265
  threshold=20, opening=2 only: sens=0.817, prec=0.343
  threshold=20, smoothing=1.0 + opening=2: sens=0.727, prec=0.505 (best F1 so far)
  extent-based shape filtering: rejected, incorrectly excludes real
    spherical follicles (a perfect sphere only fills ~52% of its
    bounding box, so 'extent' can't distinguish round-but-small-extent
    real follicles from irregular false positives)

This module implements the candidate detection function only. The
correct final parameter choice must come from nested cross-validation
(see nested_cv.py), not from picking whichever config scored best
across all 16 volumes at once.
"""

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops


def detect_follicle_candidates(volume, ovary_roi, voxel_vol_mm3,
                                 threshold=20, smooth_sigma=1.0, open_iterations=2,
                                 min_diameter_mm=2.0):
    vol = volume.astype(np.float32)
    if smooth_sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma=smooth_sigma)

    candidates = (vol < threshold) & ovary_roi

    if open_iterations > 0:
        candidates = ndimage.binary_opening(candidates, iterations=open_iterations)

    labels, n = ndimage.label(candidates)
    min_voxels = (4/3 * np.pi * (min_diameter_mm/2)**3) / voxel_vol_mm3

    keep_labels = [r.label for r in regionprops(labels) if r.area >= min_voxels]

    filtered = np.isin(labels, keep_labels) * labels
    filtered_relabeled, _ = ndimage.label(filtered > 0)
    return filtered_relabeled