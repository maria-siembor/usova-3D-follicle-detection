"""Honest end-to-end evaluation: predicted ovary (not ground truth) ->
follicle detection -> match against ground truth follicles. This is
the real answer to 'how well does the whole pipeline work together'.
Expect a similar runtime to the original ovary CV (~25-30 min locally),
since this retrains an ovary model per fold. Progress saves per fold,
safe to interrupt/resume."""

import sys
sys.path.append('src')
from full_pipeline_cv import run_full_pipeline_cv
import numpy as np

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

results = run_full_pipeline_cv(VOL_IDS)

ovary_dice = [r['ovary_dice'] for r in results]
sens = [r['follicle_sensitivity'] for r in results]
prec = [r['follicle_precision'] for r in results]
f1 = [r['follicle_f1'] for r in results]

print(f"\n=== Full pipeline results (n={len(results)}) ===")
print(f"Ovary Dice: {np.mean(ovary_dice):.3f} +/- {np.std(ovary_dice):.3f}")
print(f"Follicle Sensitivity: {np.mean(sens):.3f} +/- {np.std(sens):.3f}")
print(f"Follicle Precision: {np.mean(prec):.3f} +/- {np.std(prec):.3f}")
print(f"Follicle F1: {np.mean(f1):.3f} +/- {np.std(f1):.3f}")
print(f"\n(Compare to ground-truth-ROI follicle F1 of 0.537, "
      f"the gap shows the real cost of imperfect ovary segmentation)")