"""Python port of the official USOVA3D matching algorithm (czEvaluate.m /
AgreedBothExperts.m), so our validation is directly comparable to the
published evaluation protocol. Matches detected (R) regions against
ground-truth (F) follicles using rho1 (recall-like) x rho2
(precision-like) as the pairing score, greedy best-match-first."""

import numpy as np


def match_follicles(detected_labels, ground_truth_labels):
    """
    Args:
        detected_labels: labeled volume, our algorithm's output (0=background)
        ground_truth_labels: labeled volume, expert annotation (0=background)

    Returns:
        dict with: matches (list of (det_label, gt_label, rho1, rho2)),
        unmatched_detected (false positives), unmatched_gt (missed follicles)
    """
    det_ids = np.unique(detected_labels)
    det_ids = det_ids[det_ids > 0]
    gt_ids = np.unique(ground_truth_labels)
    gt_ids = gt_ids[gt_ids > 0]

    if len(det_ids) == 0 or len(gt_ids) == 0:
        return {
            'matches': [],
            'unmatched_detected': list(det_ids),
            'unmatched_gt': list(gt_ids),
        }

    gt_areas = {g: (ground_truth_labels == g).sum() for g in gt_ids}
    det_areas = {d: (detected_labels == d).sum() for d in det_ids}

    # score matrix: rows = detected, cols = ground truth
    scores = np.zeros((len(det_ids), len(gt_ids)))
    rho1_mat = np.zeros_like(scores)
    rho2_mat = np.zeros_like(scores)

    for i, d in enumerate(det_ids):
        det_mask = detected_labels == d
        overlapping_gt = np.unique(ground_truth_labels[det_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        for g in overlapping_gt:
            j = np.where(gt_ids == g)[0][0]
            intersection = (det_mask & (ground_truth_labels == g)).sum()
            rho1 = intersection / gt_areas[g]   # coverage of the true follicle
            rho2 = intersection / det_areas[d]  # correctness of the detection
            rho1_mat[i, j] = rho1
            rho2_mat[i, j] = rho2
            scores[i, j] = rho1 * rho2

    matches = []
    matched_det = set()
    matched_gt = set()

    while np.any(scores > 0):
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        i, j = idx
        if scores[i, j] <= 0:
            break
        matches.append((det_ids[i], gt_ids[j], rho1_mat[i, j], rho2_mat[i, j]))
        matched_det.add(det_ids[i])
        matched_gt.add(gt_ids[j])
        scores[i, :] = 0
        scores[:, j] = 0

    unmatched_detected = [d for d in det_ids if d not in matched_det]
    unmatched_gt = [g for g in gt_ids if g not in matched_gt]

    return {
        'matches': matches,
        'unmatched_detected': unmatched_detected,
        'unmatched_gt': unmatched_gt,
    }


def summarize_matching(result):
    n_matched = len(result['matches'])
    n_missed = len(result['unmatched_gt'])
    n_false_pos = len(result['unmatched_detected'])
    n_true = n_matched + n_missed

    sensitivity = n_matched / n_true if n_true > 0 else None
    precision = n_matched / (n_matched + n_false_pos) if (n_matched + n_false_pos) > 0 else None
    mean_rho1 = np.mean([m[2] for m in result['matches']]) if n_matched > 0 else None
    mean_rho2 = np.mean([m[3] for m in result['matches']]) if n_matched > 0 else None

    return {
        'n_true_follicles': n_true,
        'n_matched': n_matched,
        'n_missed': n_missed,
        'n_false_positives': n_false_pos,
        'sensitivity': sensitivity,
        'precision': precision,
        'mean_rho1': mean_rho1,
        'mean_rho2': mean_rho2,
    }