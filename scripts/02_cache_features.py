"""Extract multiscale features + consensus ovary labels for all 16
training volumes, cached to data/features/ so the cross-validation
step doesn't recompute them on every run."""

import sys
import os
import time
sys.path.append('src')

from vtk_io import read_vtk
from features import extract_features
import numpy as np

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

os.makedirs('data/features', exist_ok=True)

start_all = time.time()
for vid in VOL_IDS:
    out_feats = f'data/features/vol{vid}_feats.npy'
    out_ovary = f'data/features/vol{vid}_ovary.npy'
    if os.path.exists(out_feats) and os.path.exists(out_ovary):
        print(f"vol{vid}: already cached, skipping")
        continue

    vol, spacing, _ = read_vtk(f'data/Training_Set_1/vol{vid}.vtk')
    ovary_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r1.vtk')
    ovary_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r2.vtk')

    feats = extract_features(vol)
    ovary_consensus = ((ovary_r1 > 0) & (ovary_r2 > 0)).astype(np.uint8)

    np.save(out_feats, feats.astype(np.float16))
    np.save(out_ovary, ovary_consensus)

    print(f"vol{vid}: cached ({time.time()-start_all:.0f}s elapsed)")

print(f"\nTotal time: {time.time()-start_all:.0f}s")