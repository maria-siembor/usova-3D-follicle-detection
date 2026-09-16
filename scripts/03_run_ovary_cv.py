"""Run the full leave-one-volume-out cross-validation for ovary
segmentation."""

import sys
import time
sys.path.append('src')

from ovary_cv import run_loo_cv
import numpy as np

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

start = time.time()
results = run_loo_cv(
    VOL_IDS,
    n_per_class=20000,
    n_estimators=100,
    max_depth=15,
)
print(f"\nTotal CV time: {time.time()-start:.0f}s")

dices = [r['dice_clean'] for r in results]
ious = [r['iou_clean'] for r in results]
sens = [r['sensitivity_clean'] for r in results]
prec = [r['precision_clean'] for r in results]

print(f"\nMean Dice: {np.mean(dices):.3f} +/- {np.std(dices):.3f}")
print(f"Mean IoU: {np.mean(ious):.3f} +/- {np.std(ious):.3f}")
print(f"Mean Sensitivity: {np.mean(sens):.3f} +/- {np.std(sens):.3f}")
print(f"Mean Precision: {np.mean(prec):.3f} +/- {np.std(prec):.3f}")
