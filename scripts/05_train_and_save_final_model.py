"""Train the final ovary segmentation model on all 16 training volumes
and save it to disk."""

import sys
import time
import joblib
sys.path.append('src')

from pipeline import train_final_ovary_model

VOL_IDS = [1, 2, 3, 4, 5, 6, 101, 102, 104, 106, 109, 110, 111, 115, 117, 119]

print("Training final ovary model on all 16 volumes...")
t0 = time.time()
clf = train_final_ovary_model(VOL_IDS)
print(f"Done in {time.time()-t0:.0f}s")

joblib.dump(clf, 'data/final_ovary_model.joblib')
print("Saved to data/final_ovary_model.joblib")
