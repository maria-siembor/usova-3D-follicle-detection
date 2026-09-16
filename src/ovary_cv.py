"""Leave-one-volume-out cross-validation for the ovary segmentation
Random Forest classifier."""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage


def dice_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    dice = 2 * intersection / (pred.sum() + gt.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    sensitivity = intersection / (gt.sum() + 1e-8)
    precision = intersection / (pred.sum() + 1e-8)
    return float(dice), float(iou), float(sensitivity), float(precision)


def sample_balanced(feats, labels, n_per_class=20000, rng=None):
    rng = rng or np.random.RandomState(42)
    pos_idx = np.argwhere(labels == 1)
    neg_idx = np.argwhere(labels == 0)

    n_pos = min(n_per_class, len(pos_idx))
    n_neg = min(n_per_class, len(neg_idx))

    pos_sample = pos_idx[rng.choice(len(pos_idx), n_pos, replace=False)]
    neg_sample = neg_idx[rng.choice(len(neg_idx), n_neg, replace=False)]

    def gather(idx_arr):
        return feats[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]], \
               labels[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]]

    Xp, yp = gather(pos_sample)
    Xn, yn = gather(neg_sample)
    return np.vstack([Xp, Xn]), np.concatenate([yp, yn])


def postprocess(pred_labels):
    labeled, n = ndimage.label(pred_labels)
    if n == 0:
        return pred_labels
    sizes = ndimage.sum(pred_labels, labeled, range(1, n + 1))
    largest = np.argmax(sizes) + 1
    cleaned = (labeled == largest)
    cleaned = ndimage.binary_fill_holes(cleaned)
    return cleaned.astype(np.uint8)


def run_loo_cv(vol_ids, n_per_class=20000, n_estimators=100, max_depth=15,
               results_path='data/ovary_cv_results.json'):
    # resume from existing results if this was interrupted before
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r['vol_id'] for r in results}
        print(f"Resuming, {len(done_ids)} folds already complete: {done_ids}")
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
        test_labels = np.load(f'data/features/vol{test_vid}_ovary.npy')

        shape = test_labels.shape
        flat_feats = test_feats.reshape(-1, test_feats.shape[-1])
        pred_flat = clf.predict(flat_feats)
        pred = pred_flat.reshape(shape)

        pred_clean = postprocess(pred)

        dice_raw, iou_raw, sens_raw, prec_raw = dice_iou(pred, test_labels)
        dice_clean, iou_clean, sens_clean, prec_clean = dice_iou(pred_clean, test_labels)

        result = {
            'vol_id': test_vid,
            'dice_raw': dice_raw, 'iou_raw': iou_raw,
            'dice_clean': dice_clean, 'iou_clean': iou_clean,
            'sensitivity_clean': sens_clean, 'precision_clean': prec_clean,
        }
        results.append(result)

        # save after every fold, not just at the end
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"vol{test_vid}: Dice(raw)={dice_raw:.3f} Dice(clean)={dice_clean:.3f} "
              f"IoU(clean)={iou_clean:.3f} Sens={sens_clean:.3f} Prec={prec_clean:.3f} "
              f"[{len(results)}/{len(vol_ids)} folds done]")

    return results
