"""Follicle candidate detection: threshold within ovary ROI, morphological
opening to break speckle-noise bridges between adjacent follicles,
minimum-size filter. No maximum-size filter."""

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
