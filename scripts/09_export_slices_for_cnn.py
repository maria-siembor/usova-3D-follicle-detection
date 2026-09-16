import sys
import os
import numpy as np
sys.path.append('src')

from vtk_io import read_vtk

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

os.makedirs('data/cnn_slices', exist_ok=True)

for vid in VOL_IDS:
    vol, spacing, _ = read_vtk(f'data/Training_Set_1/vol{vid}.vtk')
    ovary_r1, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r1.vtk')
    ovary_r2, _, _ = read_vtk(f'data/Training_Set_1/vol{vid}_o_r2.vtk')

    ovary_consensus = ((ovary_r1 > 0) & (ovary_r2 > 0)).astype(np.uint8)

    # only keep slices that actually contain ovary tissue in at least one
    # of the two experts' masks, empty slices add training noise with no signal
    has_ovary = (ovary_r1 > 0) | (ovary_r2 > 0)
    slice_has_content = has_ovary.any(axis=(1, 2))

    slices = vol[slice_has_content].astype(np.uint8)
    masks = ovary_consensus[slice_has_content].astype(np.uint8)

    np.savez_compressed(
        f'data/cnn_slices/vol{vid}_slices.npz',
        images=slices, masks=masks, vol_id=vid, spacing=spacing
    )
    print(f"vol{vid}: {len(slices)} slices exported")

print("\nDone. Upload the data/cnn_slices/ folder as a Kaggle Dataset next.")
