# USOVA3D: Ovary Segmentation Results

## Method
Per-voxel Random Forest classifier (100 trees, max_depth=15), trained on
multiscale intensity/gradient/texture features (Gaussian smoothing,
gradient magnitude, Laplacian of Gaussian, local variance at sigmas
1,2,4,8) plus a positional feature (normalized distance from volume
center). Ground truth: consensus of both experts' ovary annotations.
Post-processing: largest connected component + hole filling.

Evaluated via leave-one-volume-out cross-validation across all 16
training volumes.

## Why the positional feature was added
Initial 3-volume check without it had weak precision (0.38-0.48) on
2/3 volumes. Traced to visual inspection: large non-ovary anechoic
regions (acoustic shadowing) outside the ovary boundary are
indistinguishable from true ovary tissue using intensity/texture alone.
Added normalized distance-from-center; precision improved meaningfully
on the same volumes (0.383->0.479, 0.481->0.592).

## Full 16-fold LOO-CV results
**Mean: Dice 0.785 +/- 0.073, IoU 0.652 +/- 0.096, Sensitivity 0.854 +/-
0.074, Precision 0.747 +/- 0.140.**

Weakest volumes: vol1, vol2, vol111 (precision 0.48-0.60), consistent
with the shadowing issue found during development.

---

# USOVA3D: Follicle Detection Results

## Method
Within the (ground-truth, consensus) ovary ROI: Gaussian smoothing,
intensity threshold, morphological opening (breaks speckle-noise
bridges between adjacent follicles), minimum-size filter (~2mm
equivalent diameter). No maximum-size filter, genuine large cysts exist
in this data (both experts agreed on structures up to ~140mm diameter,
in the 101+ volume series specifically).

Ground truth follicles: consensus between both experts, computed using
a Python port of the official USOVA3D matching algorithm (czEvaluate.m /
AgreedBothExperts.m: rho1 x rho2 greedy best-match-first), so our
matching is directly comparable to the published evaluation protocol,
not an invented criterion.

## Development history (informal, all 16 volumes, NOT the final result)
- No smoothing, no opening: sens=0.637, prec=0.265
- Opening only: sens=0.817, prec=0.343 (found via visualizing vol3:
  naive thresholding merges adjacent follicles via speckle-noise
  bridges into one large blob; opening breaks these)
- Smoothing + opening: sens=0.727, prec=0.505 (best F1 found this way,
  ~0.596)
- Shape filtering via 'extent' (bounding-box fill ratio): REJECTED,
  too aggressive, incorrectly excludes real spherical follicles (a
  perfect sphere only fills ~52% of its own bounding box)
- Shape filtering via 'solidity' (convex hull): computationally
  infeasible on the sandbox (34s/volume, dominated by convex hull cost
  on large merged blobs), not pursued further this iteration

## Final result: nested cross-validation (honest, unbiased estimate)
For each held-out volume, the best config (from a small grid: threshold
in {18,22}, smooth_sigma in {0.5,1.0}, open_iterations in {1,2}) was
selected using ONLY the other 15 volumes, then applied to the held-out
one. This avoids the overfitting risk of the informal exploration above.

**The same config (threshold=18, smooth_sigma=1.0, open_iterations=2)
was selected in all 16/16 folds**, a real stability signal, not
overfitting noise.

**Mean Sensitivity: 0.734 +/- 0.227**
**Mean Precision: 0.467 +/- 0.209**
**Mean F1: 0.537 +/- 0.189**

Notably lower than the informal 0.596 F1 found by tuning across all 16
volumes at once, exactly the gap nested CV is meant to reveal.

## Known weak points
- vol110 (F1=0.194, precision=0.111) and vol3 (F1=0.200,
  precision=0.200): both driven by very low precision, many false
  positives. Not yet investigated, next priority.
- vol104 (F1=1.000): perfect match, notably this is one of the 101+
  volumes with the very large single cyst-like structure, an easier
  case (few, large, well-separated targets).

## Comparison to published baselines
Potočnik et al. 2020's own baselines: 3D Directional Wavelet Transform
~78/100, CNN baseline slightly lower, on their composite scoring metric
(not directly comparable units to our sensitivity/precision/F1, since
their combined score also incorporates volume ratio and surface
distance for matched follicles, which we have not yet implemented).
Human inter-rater agreement: ~83/100. A direct comparison would require
implementing the full composite scoring, deferred as future work.

## Next steps
1. Investigate vol110 and vol3 specifically (why so many false
   positives)
2. Chain this stage to the ovary segmentation stage (currently
   evaluated independently using ground-truth ROIs; end-to-end
   performance using OUR segmented ovary, not the ground truth, will
   likely be somewhat lower)
3. Volume/diameter measurement for matched follicles
4. Output in the official submission format (labeled VTK per test
   volume) for true external validation against the held-out test set

---

# USOVA3D: Ovary Erosion Fix (Boundary Uncertainty)

## Problem found
End-to-end pipeline (predicted ovary -> follicle detection) showed
Follicle F1 = 0.373, a large drop from the ground-truth-ROI F1 of
0.537. Investigation found the drop was NOT proportional to overall
ovary Dice: vol115 had one of the best ovary Dice scores (0.858) yet
one of the worst follicle precisions (0.043), showing that WHERE
segmentation errors occur (at the boundary) matters more than how much
error there is overall.

## Fix
Erode the predicted ovary mask (binary erosion) before using it as the
follicle-detection ROI, pulling the search area inward, away from the
ovary boundary where segmentation is least certain. True follicles sit
well inside the ovary interior, not against its boundary, so this
should remove boundary-driven false positives without losing real
detections.

## Validation
Quick informal check (leaky, final model tested on its own training
data, used only to find the right erosion amount cheaply): sensitivity
stayed exactly flat (0.696) from erosion=0 through erosion=5, while
precision rose steadily (0.310 -> 0.517). Erosion=6 was the first point
sensitivity dropped (0.696 -> 0.679), confirming erosion=5 as the
right amount (maximum precision gain, no sensitivity cost).

**Honest re-validation (proper per-fold retraining, no leakage) with
erosion=5:**
- Follicle Sensitivity: 0.680 +/- 0.215 (was 0.696 informally, 0.696 pre-fix)
- Follicle Precision: 0.500 +/- 0.312 (was 0.320 pre-fix)
- **Follicle F1: 0.519 +/- 0.232 (was 0.373 pre-fix)**

This closes nearly the entire gap to the ground-truth-ROI ceiling of
0.537, confirming the erosion fix as a genuine, validated improvement,
not an artifact of the informal check.

## Remaining known weak points
vol3, vol110, vol115 still show low precision (0.125-0.143) even with
the fix. Not yet investigated individually; candidates for future work.

---

# USOVA3D: 2D U-Net vs Classical RF Comparison

## Setup
2D U-Net (from-scratch PyTorch implementation, standard encoder-decoder
with skip connections), trained on 2D axial slices exported from the
16 training volumes (~2900 slices total, only slices containing ovary
tissue kept). 4-fold grouped cross-validation (grouped by volume, no
slice-level leakage between train/test). Trained on Kaggle (T4 GPU),
~24 min/fold, 30 epochs, combined Dice+BCE loss.

## Headline result: essentially a statistical tie
| Method | Dice | IoU |
|---|---|---|
| Classical RF (16-fold LOO-CV) | 0.785 +/- 0.073 | 0.652 +/- 0.096 |
| 2D U-Net (4-fold grouped CV) | 0.796 +/- 0.095 | 0.670 +/- 0.123 |

U-Net's mean is marginally higher, but its variance is also higher, so
this is not a clear win for either method on the headline number alone.

## The more interesting finding: complementary failure patterns
Per-volume comparison shows the two methods fail on almost entirely
different volumes, not a case of one method being uniformly better:

| Volume | Classical RF Dice | U-Net Dice | Difference |
|---|---|---|---|
| vol110 | 0.734 | 0.562 | RF much better (-0.172 for U-Net) |
| vol101 | 0.847 | 0.766 | RF better (-0.081) |
| vol3 | 0.811 | 0.702 | RF better (-0.109) |
| vol111 | 0.723 | 0.857 | U-Net much better (+0.134) |
| vol6 | 0.703 | 0.800 | U-Net better (+0.097) |
| vol117 | 0.775 | 0.853 | U-Net better (+0.078) |
| vol2 | 0.705 | 0.775 | U-Net better (+0.070) |

U-Net wins on 11/16 volumes overall, but RF's wins are concentrated on
specific, different cases than U-Net's losses. This suggests the two
methods are sensitive to different failure modes (3D spatial features
vs. 2D slice-level texture), rather than one being a strictly better
version of the other. An ensemble combining both could plausibly
outperform either alone, noted as future work.

## Methodological caveats (stated explicitly, not glossed over)
- 4-fold (U-Net) vs 16-fold (RF) cross-validation granularity differs,
  the U-Net's per-fold test sets are 4x larger, so its per-fold
  variance estimate is coarser than the RF's per-volume LOO estimate.
- 2D-slice-based prediction (U-Net) vs genuinely 3D multiscale features
  (RF) is a real architectural difference, not just "two classifiers on
  the same information."

## Decision: web app will offer both methods as a toggle
Given the tie plus complementary failure modes, rather than picking one
"winner," the interactive app exposes both the classical RF and the
U-Net as selectable options, reflecting the actual finding (two valid,
differently-behaved methods) rather than overstating one as superior.