# NeuroScan — Web UI for EEG Seizure Analysis

A professional, premium web application that lets you upload a raw `.edf`
EEG file and get:

- A clean **oscilloscope-style visualization** of the 18-channel bipolar EEG signal
  (time-stamped, labeled axes, interactive zoom/pan).
- **Seizure localization** — detected ictal regions are highlighted on the trace.
- **Subtype classification** (CNN v4 dual-embedding model) shown below the signal,
  with per-class probability bars and a per-event breakdown.

Built with Flask (backend), Plotly.js (visualization), and the project's existing
CNN v4 pipeline.

## Design

- **Light poster palette:** warm paper background, near-black ink text, muted
  mustard-yellow accent, and vermilion red reserved for seizure markings. No
  gradients, no neon.
- **Signal viewer:** each EEG channel is drawn in its own distinct color, and
  detected seizure regions are marked in a single bright vermilion band.
- **Typography:** Manrope (UI) + DM Mono (data / labels).
- **UI/UX:** clear view states (upload → processing → results → error),
  drag-and-drop with keyboard support, interactive plot with zoom/pan/reset,
  and responsive layout.

## Architecture (CNN v4)

- **Input:** 10 s windows @ 256 Hz, 18 bipolar channels, per-channel z-scored.
- **Feature extraction:** two frozen CRNN backbones
  (`phase_a_backbone_detector.pt` + `finetuned_backbone.pt`) whose 128-d
  embeddings are concatenated into a 256-d vector.
- **Classifier:** `HistGradientBoostingClassifier`
  (`max_depth=4, lr=0.08, max_iter=200, l2=0.5`), trained with SMOTE
  (`focal_aware ×3, focal_to_bilateral_tonic_clonic ×5`).
- **Seizure detection:** the Phase A detached backbone's binary head, applied over
  4 s windows, thresholded, and filtered to require a minimum event duration.

## File layout

```
web/
├── app.py            # Flask server (routes: /, /api/status, /api/analyze)
├── inference.py      # Model architecture + EDF preprocessing + prediction pipeline
├── train_export.py   # Re-trains & saves checkpoints/crnn_hierarchical_smote_v4/classifier.pkl
├── requirements.txt
├── templates/index.html
├── static/css/style.css
└── static/js/app.js
```

## Setup

```bash
cd web
# using the project virtual environment
../venv/Scripts/python -m pip install -r requirements.txt
../venv/Scripts/python -m pip install flask
```

## Train / export the classifier (one-time)

The trained classifier is saved as
`checkpoints/crnn_hierarchical_smote_v4/classifier.pkl`. If it is missing
(or you want to re-export it), run:

```bash
../venv/Scripts/python train_export.py
```

This extracts dual-embeddings from the harmonized stage-2 windows, SMOTE-resamples,
trains the `HistGradientBoostingClassifier`, and writes `classifier.pkl`
(expected test accuracy ≈ 0.90–0.91).

## Run

```bash
../venv/Scripts/python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Usage

1. Drag-and-drop (or browse) a `.edf` file.
2. Select the recording's **mains frequency** (50 Hz for EU/Asia, 60 Hz for
   US/Americas) and a detection **threshold**.
3. NeuroScan preprocesses the signal, runs seizure detection, classifies the
   subtype, and renders the annotated EEG trace.

## Notes

- Both **CHB-MIT** (already bipolar) and **Siena Scalp** (monopolar, auto-derived
  to the canonical double-banana bipolar montage) `.edf` files are supported.
- `classifier.pkl` is the only artifact that must be present for full subtype
  classification. If it is absent, seizure detection still works and subtype
  shows as "Unknown".
