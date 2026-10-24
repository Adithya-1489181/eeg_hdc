# Test data

Small `.edf` clips for exercising the web UI, trimmed from the project's
raw dataset (`data/raw/...`) so they stay small (2–8 MB each) instead of
shipping full hour-long recordings.

Each seizure clip is +/- 60s around the annotated seizure window.
Non-seizure clips are 3-minute segments from recordings with zero
annotated seizures.

| Folder | Source | Class |
|---|---|---|
| `focal_aware/` | SienaScalp PN14 (`PN14-1.edf`) | Focal Aware |
| `focal_impaired_awareness/` | SienaScalp PN00 (`PN00-1.edf`) | Focal Impaired Awareness |
| `focal_to_bilateral_tonic_clonic/` | SienaScalp PN10 (`PN10-1.edf`) | Focal to Bilateral Tonic-Clonic |
| `non_seizure/` | CHB-MIT chb01/chb02 (`chb01_01.edf`, `chb02_01.edf`) | No seizure |

Subtype labels come from `data/harmonized/windows/manifest.json`
(`subtype_by_patient`). Seizure timings come from each patient's
`Seizures-list-PNxx.txt` / `*-summary.txt` in `data/raw/`.

Verified to load cleanly through `inference.py`'s `preprocess_edf`.
