"""Multiscale intensity/edge/texture feature extraction for per-voxel
ovary classification. Features: raw intensity, Gaussian-smoothed
intensity, gradient magnitude, Laplacian of Gaussian, local variance,
each at several scales, plus a positional feature (distance from volume
center), following the standard multiscale RF-segmentation recipe.

The positional feature was added after finding that a purely
intensity/texture-based classifier had low precision (0.38-0.48 on 2/3
volumes tested): it cannot distinguish "dark tissue near the ovary's
typical central location" from "dark tissue elsewhere in the frame"
(e.g. acoustic shadowing near the frame edge), since none of the
intensity/texture features encode voxel position."""

import numpy as np
from scipy import ndimage


def extract_features(volume, sigmas=(1, 2, 4, 8)):
    vol = volume.astype(np.float32)
    features = [vol]

    for sigma in sigmas:
        smoothed = ndimage.gaussian_filter(vol, sigma=sigma)
        features.append(smoothed)

        grad = ndimage.gaussian_gradient_magnitude(vol, sigma=sigma)
        features.append(grad)

        log = ndimage.gaussian_laplace(vol, sigma=sigma)
        features.append(log)

        local_mean = ndimage.uniform_filter(vol, size=int(sigma * 2 + 1))
        local_sqmean = ndimage.uniform_filter(vol**2, size=int(sigma * 2 + 1))
        local_var = np.clip(local_sqmean - local_mean**2, 0, None)
        features.append(local_var)

    # Positional feature: normalized distance from volume center.
    # Normalized per-axis by that axis's own size, since volumes vary
    # in dimensions (confirmed earlier: shapes range from
    # (199,91,101) to (247,159,187) across volumes).
    z, y, x = np.indices(vol.shape).astype(np.float32)
    dz, dy, dx = vol.shape
    z_norm = (z - dz / 2) / (dz / 2)
    y_norm = (y - dy / 2) / (dy / 2)
    x_norm = (x - dx / 2) / (dx / 2)
    dist_from_center = np.sqrt(z_norm**2 + y_norm**2 + x_norm**2)
    features.append(dist_from_center)

    return np.stack(features, axis=-1)