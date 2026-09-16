import os
import json
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

from ovary_cv import sample_balanced, postprocess as postprocess_ovary, dice_iou
from follicle_matching import match_follicles, summarize_matching
from follicle_detection import detect_follicle_candidates
from vtk_io import read_vtk

FOLLICLE_CONFIG = {'threshold': 18, 'smooth_sigma': 1.0, 'open_iterations': 2}


def run_full_pipeline_cv(vol_ids, results_path='data/full_pipeline_cv_results.json',
                          n_per_class=20000, n_estimators=100, max_depth=15):
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r['vol_id'] for r in results}
        print(f"Resuming, {len(done_ids)} folds already done")
    else:
        results = []
        done_ids = set()

    for test_vid in vol_ids:
        if test_vid in done_ids:
            continue

        train_ids = [v for v in vol_ids if v != test_vid]

        X_train, y_train = [], []
        for vid in train_ids:
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

        test_feats = np.load(f'data/features/vol{test_vid}_feats.npy').astype(np.float32)
        test_ovary_gt = np.load(f'data/features/vol{test_vid}_ovary.npy')
        shape = test_ovary_gt.shape
        pred_ovary = clf.predict(test_feats.reshape(-1, test_feats.shape[-1])).reshape(shape)
        pred_ovary_clean = postprocess_ovary(pred_ovary)

        ovary_dice, ovary_iou, _, _ = dice_iou(pred_ovary_clean, test_ovary_gt)

        vol, spacing, _ = read_vtk(f'data/Training_Set_1/vol{test_vid}.vtk')
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

        # erosion=5, selected via quick check: sens unchanged (0.696),
        # precision improved 0.310->0.517, see RESULTS_LOG.md
        eroded_roi = ndimage.binary_erosion(pred_ovary_clean.astype(bool), iterations=5)
        follicle_labels = detect_follicle_candidates(
            vol, eroded_roi, voxel_vol_mm3, **FOLLICLE_CONFIG
        )

        foll_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{test_vid}_f_r1.vtk')
        foll_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{test_vid}_f_r2.vtk')
        match_r1r2 = match_follicles(foll_r1, foll_r2)
        agreed_gt_ids = set(m[1] for m in match_r1r2['matches'])
        gt_consensus = np.isin(foll_r2, list(agreed_gt_ids)) * foll_r2

        result = match_follicles(follicle_labels, gt_consensus)
        summary = summarize_matching(result)
        sens = summary['sensitivity'] or 0
        prec = summary['precision'] or 0
        f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0

        record = {
            'vol_id': test_vid,
            'ovary_dice': ovary_dice, 'ovary_iou': ovary_iou,
            'follicle_sensitivity': sens, 'follicle_precision': prec, 'follicle_f1': f1,
        }
        results.append(record)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"vol{test_vid}: ovary_dice={ovary_dice:.3f} | "
              f"follicle sens={sens:.3f} prec={prec:.3f} F1={f1:.3f} "
              f"[{len(results)}/{len(vol_ids)} done]")

    return results
