"""Nested cross-validation for follicle detection parameters."""

import os
import json
import numpy as np
import sys
sys.path.append('src')

from vtk_io import read_vtk
from follicle_matching import match_follicles, summarize_matching
from follicle_detection import detect_follicle_candidates

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

CONFIG_GRID = [
    {'threshold': t, 'smooth_sigma': s, 'open_iterations': o}
    for t in [18, 22]
    for s in [0.5, 1.0]
    for o in [1, 2]
]


def load_volume_data(vid):
    vol, spacing, _ = read_vtk(f'data/Training_Set_1/vol{vid}.vtk')
    ovary_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r1.vtk')
    ovary_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r2.vtk')
    foll_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_f_r1.vtk')
    foll_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_f_r2.vtk')

    ovary_roi = (ovary_r1 > 0) & (ovary_r2 > 0)
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

    match_r1r2 = match_follicles(foll_r1, foll_r2)
    agreed_gt_ids = set(m[1] for m in match_r1r2['matches'])
    gt_consensus = np.isin(foll_r2, list(agreed_gt_ids)) * foll_r2

    return {'vol': vol, 'ovary_roi': ovary_roi, 'voxel_vol_mm3': voxel_vol_mm3,
            'gt_consensus': gt_consensus}


def evaluate_config(data, config):
    filtered = detect_follicle_candidates(
        data['vol'], data['ovary_roi'], data['voxel_vol_mm3'], **config
    )
    result = match_follicles(filtered, data['gt_consensus'])
    summary = summarize_matching(result)
    sens = summary['sensitivity'] or 0
    prec = summary['precision'] or 0
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0
    return sens, prec, f1


def run_nested_cv(results_path='data/follicle_nested_cv_results.json'):
    print("Loading and precomputing all 16 volumes...")
    all_data = {vid: load_volume_data(vid) for vid in VOL_IDS}
    print("Done.\n")

    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r['vol_id'] for r in results}
        print(f"Resuming, {len(done_ids)} folds already done: {done_ids}")
    else:
        results = []
        done_ids = set()

    for test_vid in VOL_IDS:
        if test_vid in done_ids:
            continue

        train_ids = [v for v in VOL_IDS if v != test_vid]

        best_config = None
        best_mean_f1 = -1
        for config in CONFIG_GRID:
            f1_scores = [evaluate_config(all_data[vid], config)[2] for vid in train_ids]
            mean_f1 = np.mean(f1_scores)
            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_config = config

        sens, prec, f1 = evaluate_config(all_data[test_vid], best_config)

        result = {
            'vol_id': test_vid,
            'selected_config': best_config,
            'train_mean_f1': best_mean_f1,
            'test_sensitivity': sens,
            'test_precision': prec,
            'test_f1': f1,
        }
        results.append(result)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"vol{test_vid}: selected {best_config} (train F1={best_mean_f1:.3f}) "
              f"-> test sens={sens:.3f} prec={prec:.3f} F1={f1:.3f} "
              f"[{len(results)}/{len(VOL_IDS)} done]")

    return results
