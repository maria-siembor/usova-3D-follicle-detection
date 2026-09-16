"""Run the full nested cross-validation for follicle detection.
Expect real runtime here, roughly 2000 config-evaluations total across
all folds. Progress saves after every outer fold (data/follicle_nested_cv_results.json),
safe to interrupt (Ctrl+C) and rerun, it resumes automatically."""

import sys
sys.path.append('src')
from follicle_nested_cv import run_nested_cv
import numpy as np

results = run_nested_cv()

sens = [r['test_sensitivity'] for r in results]
prec = [r['test_precision'] for r in results]
f1 = [r['test_f1'] for r in results]

print(f"\n=== Final nested CV results (n={len(results)} folds) ===")
print(f"Mean Sensitivity: {np.mean(sens):.3f} +/- {np.std(sens):.3f}")
print(f"Mean Precision: {np.mean(prec):.3f} +/- {np.std(prec):.3f}")
print(f"Mean F1: {np.mean(f1):.3f} +/- {np.std(f1):.3f}")

# which configs got selected most often, useful to know if one config dominates
from collections import Counter
config_strs = [str(r['selected_config']) for r in results]
print("\nConfig selection frequency:")
for cfg, count in Counter(config_strs).most_common():
    print(f"  {count}x: {cfg}")