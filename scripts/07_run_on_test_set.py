"""Run the full deployment pipeline on all 19 held-out test volumes:
segment ovary, detect follicles, measure, classify by IVF retrieval
suitability. Saves per-volume measurement CSVs, a labeled VTK per
volume (official USOVA3D submission format), and a summary CSV across
all patients. Progress saves per volume, resumable."""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
sys.path.append('src')

from vtk_io import read_vtk, write_vtk_labels
from pipeline import run_pipeline

TEST_VOL_IDS = [7, 8, 9, 11, 12, 13, 14, 15, 103, 105, 107, 108, 112, 113, 114, 116, 118, 120, 121]

os.makedirs('outputs/test_set_results', exist_ok=True)
os.makedirs('outputs/test_set_segmented', exist_ok=True)

clf = joblib.load('data/final_ovary_model.joblib')

results_path = 'outputs/test_set_summary.json'
if os.path.exists(results_path):
    with open(results_path) as f:
        all_summaries = json.load(f)
    done_ids = {r['vol_id'] for r in all_summaries}
    print(f"Resuming, {len(done_ids)} volumes already done")
else:
    all_summaries = []
    done_ids = set()

for vid in TEST_VOL_IDS:
    if vid in done_ids:
        continue

    vol, spacing, _ = read_vtk(f'data/Test_Set_1/vol{vid}.vtk')
    result = run_pipeline(vol, spacing, clf)

    # per-patient measurements CSV
    meas_df = pd.DataFrame(result['measurements'])
    meas_df.to_csv(f'outputs/test_set_results/vol{vid}_measurements.csv', index=False)

    # official-format labeled VTK output (for potential submission to the
    # actual USOVA3D evaluation, per Description_Data_Structures_For_Evaluation.pdf)
    write_vtk_labels(result['follicle_labels'].astype(np.uint8), spacing,
                      f'outputs/test_set_segmented/vol{vid}_seg_01.vtk',
                      dataset_name=f'vol{vid} automated segmentation')

    summary = {'vol_id': vid, **result['summary']}
    all_summaries.append(summary)

    with open(results_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"vol{vid}: {summary['total_follicle_count']} follicles total, "
          f"{summary['optimal_for_retrieval_count']} optimal for retrieval "
          f"[{len(all_summaries)}/{len(TEST_VOL_IDS)} done]")

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv('outputs/test_set_patient_summary.csv', index=False)
print(f"\nSaved patient-level summary to outputs/test_set_patient_summary.csv")
print(summary_df.describe())