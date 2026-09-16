import sys
import joblib
import numpy as np
from scipy import ndimage
sys.path.append('src')

from vtk_io import read_vtk
from ovary_cv import postprocess as postprocess_ovary
from follicle_detection import detect_follicle_candidates
from follicle_matching import match_follicles, summarize_matching

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]
FOLLICLE_CONFIG = {'threshold': 18, 'smooth_sigma': 1.0, 'open_iterations': 2}

clf = joblib.load('data/final_ovary_model.joblib')

for erosion in [3, 4, 5, 6]:
    all_sens, all_prec = [], []
    for vid in VOL_IDS:
        feats = np.load(f'data/features/vol{vid}_feats.npy').astype(np.float32)
        shape = feats.shape[:-1]
        pred_ovary = clf.predict(feats.reshape(-1, feats.shape[-1])).reshape(shape)
        ovary_mask = postprocess_ovary(pred_ovary).astype(bool)

        if erosion > 0:
            ovary_mask = ndimage.binary_erosion(ovary_mask, iterations=erosion)

        vol, spacing, _ = read_vtk(f'data/Training_Set_1/vol{vid}.vtk')
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

        follicle_labels = detect_follicle_candidates(vol, ovary_mask, voxel_vol_mm3, **FOLLICLE_CONFIG)

        foll_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_f_r1.vtk')
        foll_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_f_r2.vtk')
        match_r1r2 = match_follicles(foll_r1, foll_r2)
        agreed_gt_ids = set(m[1] for m in match_r1r2['matches'])
        gt_consensus = np.isin(foll_r2, list(agreed_gt_ids)) * foll_r2

        result = match_follicles(follicle_labels, gt_consensus)
        summary = summarize_matching(result)
        all_sens.append(summary['sensitivity'] or 0)
        all_prec.append(summary['precision'] or 0)

    mean_sens, mean_prec = np.mean(all_sens), np.mean(all_prec)
    f1 = 2*mean_sens*mean_prec/(mean_sens+mean_prec) if (mean_sens+mean_prec)>0 else 0
    print(f"erosion={erosion}: sens={mean_sens:.3f}, prec={mean_prec:.3f}, F1={f1:.3f}")
