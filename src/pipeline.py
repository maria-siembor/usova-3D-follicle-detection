"""End-to-end pipeline: raw volume -> segmented ovary -> detected
follicles -> measurements -> IVF retrieval-suitability classification.

Clinical grounding for the size classification (see references below):
no single universal threshold exists in the literature, "definitive
clinical consensus... is yet to be determined" (Nature Communications,
2025). Different sources and patient populations cite different ranges:
  - General/most commonly cited: lead follicles >=17-18mm as the
    standard trigger criterion; 16-22mm cited as the range most likely
    to yield a mature oocyte at retrieval (CNY Fertility; PMC5930292)
  - Diminished ovarian reserve patients: 15-17mm optimal (J Ovarian Res, 2025)
  - Poor responders (POSEIDON group 3/4): 18-24mm optimal (PMC12406972)
Default here uses the general 16-22mm range, but this is a parameter,
not a hardcoded clinical claim, adjust per the intended patient
population / clinic protocol.
"""

import numpy as np
import joblib
from scipy import ndimage
from skimage.measure import regionprops

from features import extract_features
from ovary_cv import postprocess as postprocess_ovary
from follicle_detection import detect_follicle_candidates

FOLLICLE_CONFIG = {'threshold': 18, 'smooth_sigma': 1.0, 'open_iterations': 2}

# Default retrieval-size window (mm), see docstring above for sourcing
DEFAULT_MIN_RETRIEVAL_MM = 16.0
DEFAULT_MAX_RETRIEVAL_MM = 22.0


def train_final_ovary_model(vol_ids, n_per_class=20000, n_estimators=100, max_depth=15):
    from sklearn.ensemble import RandomForestClassifier
    from ovary_cv import sample_balanced

    X_train, y_train = [], []
    for vid in vol_ids:
        feats = np.load(f'data/features/vol{vid}_feats.npy').astype(np.float32)
        labels = np.load(f'data/features/vol{vid}_ovary.npy')
        X, y = sample_balanced(feats, labels, n_per_class=n_per_class)
        X_train.append(X)
        y_train.append(y)
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def segment_ovary(clf, volume):
    feats = extract_features(volume)
    shape = volume.shape
    pred = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(shape)
    return postprocess_ovary(pred)


def classify_maturity(diameter_mm, min_retrieval_mm=DEFAULT_MIN_RETRIEVAL_MM,
                       max_retrieval_mm=DEFAULT_MAX_RETRIEVAL_MM):
    if diameter_mm < min_retrieval_mm:
        return 'immature'
    elif diameter_mm <= max_retrieval_mm:
        return 'optimal_for_retrieval'
    else:
        return 'post_mature'


def measure_follicles(follicle_labels, voxel_vol_mm3,
                       min_retrieval_mm=DEFAULT_MIN_RETRIEVAL_MM,
                       max_retrieval_mm=DEFAULT_MAX_RETRIEVAL_MM):
    measurements = []
    for region in regionprops(follicle_labels):
        volume_mm3 = region.area * voxel_vol_mm3
        equiv_diameter_mm = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
        zc, yc, xc = region.centroid
        maturity = classify_maturity(equiv_diameter_mm, min_retrieval_mm, max_retrieval_mm)
        measurements.append({
            'follicle_id': int(region.label),
            'volume_mm3': float(volume_mm3),
            'equivalent_diameter_mm': float(equiv_diameter_mm),
            'centroid_zyx': (float(zc), float(yc), float(xc)),
            'maturity': maturity,
        })
    return measurements


def summarize_patient(measurements):
    """The clinically-relevant summary: total count, and how many
    fall in the optimal retrieval window, mirroring how AFC (antral
    follicle count) and lead-follicle counts are reported clinically."""
    total = len(measurements)
    n_optimal = sum(1 for m in measurements if m['maturity'] == 'optimal_for_retrieval')
    n_immature = sum(1 for m in measurements if m['maturity'] == 'immature')
    n_post_mature = sum(1 for m in measurements if m['maturity'] == 'post_mature')
    return {
        'total_follicle_count': total,
        'optimal_for_retrieval_count': n_optimal,
        'immature_count': n_immature,
        'post_mature_count': n_post_mature,
    }


def run_pipeline(volume, spacing, ovary_clf, follicle_config=None,
                  min_retrieval_mm=DEFAULT_MIN_RETRIEVAL_MM,
                  max_retrieval_mm=DEFAULT_MAX_RETRIEVAL_MM,
                  ovary_erosion_iterations=5):
    follicle_config = follicle_config or FOLLICLE_CONFIG
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    ovary_mask = segment_ovary(ovary_clf, volume)
    ovary_roi = ovary_mask.astype(bool)
    if ovary_erosion_iterations > 0:
        from scipy import ndimage as _ndi
        ovary_roi = _ndi.binary_erosion(ovary_roi, iterations=ovary_erosion_iterations)
    follicle_labels = detect_follicle_candidates(
        volume, ovary_roi, voxel_vol_mm3, **follicle_config
    )
    measurements = measure_follicles(follicle_labels, voxel_vol_mm3,
                                       min_retrieval_mm, max_retrieval_mm)
    summary = summarize_patient(measurements)

    return {
        'ovary_mask': ovary_mask,
        'follicle_labels': follicle_labels,
        'measurements': measurements,
        'summary': summary,
    }