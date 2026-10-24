import json
import pickle
import warnings
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT_ROOT / "checkpoints"

CANONICAL_18 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ", "CZ-PZ",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
]

LEFT_TEMPORAL = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
LEFT_PARASAGITTAL = ["FP1-F3", "F3-C3", "C3-P3", "P3-O1"]
MIDLINE = ["FZ-CZ", "CZ-PZ"]
RIGHT_PARASAGITTAL = ["FP2-F4", "F4-C4", "C4-P4", "P4-O2"]
RIGHT_TEMPORAL = ["FP2-F8", "F8-T8", "T8-P8", "P8-O2"]

N_CH = len(CANONICAL_18)
TARGET_SFREQ = 256.0
BP_LOW, BP_HIGH = 0.5, 45.0

STAGE1_WINDOW_S = 4.0
STAGE1_N = int(round(STAGE1_WINDOW_S * TARGET_SFREQ))

STAGE2_WINDOW_S = 10.0
STAGE2_N = int(round(STAGE2_WINDOW_S * TARGET_SFREQ))
STAGE2_STRIDE_S = 2.0

CNN_WIDTH = 32
RNN_HIDDEN = 64
RNN_LAYERS = 1
CNN_DROPOUT = 0.1
RNN_DROPOUT = 0.2

CLASS_NAMES = [
    "Focal Aware",
    "Focal Impaired Awareness",
    "Focal to Bilateral Tonic-Clonic",
]
CLASS_KEYS = [
    "focal_aware",
    "focal_impaired_awareness",
    "focal_to_bilateral_tonic_clonic",
]

BIPOLAR_MAP = {
    "FP1-F7": ("Fp1", "F7"),
    "F7-T7": ("F7", "T3"),
    "T7-P7": ("T3", "T5"),
    "P7-O1": ("T5", "O1"),
    "FP1-F3": ("Fp1", "F3"),
    "F3-C3": ("F3", "C3"),
    "C3-P3": ("C3", "P3"),
    "P3-O1": ("P3", "O1"),
    "FZ-CZ": ("Fz", "Cz"),
    "CZ-PZ": ("Cz", "Pz"),
    "FP2-F4": ("Fp2", "F4"),
    "F4-C4": ("F4", "C4"),
    "C4-P4": ("C4", "P4"),
    "P4-O2": ("P4", "O2"),
    "FP2-F8": ("Fp2", "F8"),
    "F8-T8": ("F8", "T4"),
    "T8-P8": ("T4", "T6"),
    "P8-O2": ("T6", "O2"),
}

def _norm_name(ch):
    c = ch.strip()
    for p in ("EEG ", "EKG ", "ECG ", "EEG"):
        if c.upper().startswith(p.upper()):
            c = c[len(p):].strip()
    for s in ("-REF", "-LE", "-ref", "-le"):
        if c.endswith(s):
            c = c[: -len(s)].strip()
    return c


def _is_non_eeg(ch):
    return ch.upper() in {"1", "2", "MK", "EKG", "ECG", "EKG1", "EKG2",
                           "VNS", "-", "--", ".", "EVENTS", "STI 014",
                           "POLYPHY", "DC01", "DC02", "DC03", "DC04"}


class CRNNBackbone(nn.Module):
    def __init__(self, n_channels=18, cnn_width=CNN_WIDTH,
                 rnn_hidden=RNN_HIDDEN, rnn_layers=RNN_LAYERS,
                 cnn_dropout=CNN_DROPOUT, rnn_dropout=RNN_DROPOUT):
        super().__init__()

        def block(c_in, c_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, stride=stride,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(cnn_dropout),
            )

        self.conv = nn.ModuleList([
            block(n_channels, cnn_width, 63, 2),
            block(cnn_width, cnn_width * 2, 31, 2),
            block(cnn_width * 2, cnn_width * 4, 15, 2),
            block(cnn_width * 4, cnn_width * 4, 7, 2),
        ])
        self.rnn = nn.LSTM(
            cnn_width * 4, rnn_hidden, num_layers=rnn_layers,
            batch_first=True, bidirectional=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.Tanh(),
            nn.Linear(rnn_hidden, 1),
        )
        self.out_dim = rnn_hidden * 2

    def forward(self, x):
        f = x
        for block in self.conv:
            f = block(f)
        f = f.transpose(1, 2)
        seq, _ = self.rnn(f)
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), seq).squeeze(1)


class HierarchicalCRNN(nn.Module):
    def __init__(self, n_channels=N_CH, n_subtypes=3, cnn_width=CNN_WIDTH,
                 rnn_hidden=RNN_HIDDEN, rnn_layers=RNN_LAYERS,
                 cnn_dropout=CNN_DROPOUT, rnn_dropout=RNN_DROPOUT,
                 head_dropout=0.35):
        super().__init__()
        self.backbone = CRNNBackbone(n_channels, cnn_width, rnn_hidden,
                                     rnn_layers, cnn_dropout, rnn_dropout)
        d = self.backbone.out_dim

        def head(n_out):
            return nn.Sequential(
                nn.Linear(d, d // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(d // 2, n_out),
            )

        self.detect_head = head(1)
        self.subtype_head = head(n_subtypes)


class SeizureAnalyzer:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.detect_model = None
        self.frozen_backbone = None
        self.finetuned_backbone = None
        self.classifier = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        phase_a_ckpt = (
            CHECKPOINTS / "crnn_hierarchical"
            / "phase_a_backbone_detector.pt"
        )
        finetuned_ckpt = (
            CHECKPOINTS / "crnn_hierarchical_smote_v2"
            / "finetuned_backbone.pt"
        )
        classifier_pkl = (
            CHECKPOINTS / "crnn_hierarchical_smote_v4"
            / "classifier.pkl"
        )

        if not phase_a_ckpt.exists():
            raise FileNotFoundError(
                f"Phase A checkpoint not found: {phase_a_ckpt}"
            )
        if not finetuned_ckpt.exists():
            raise FileNotFoundError(
                f"Fine-tuned backbone not found: {finetuned_ckpt}"
            )

        ckpt_a = torch.load(
            phase_a_ckpt, map_location=self.device, weights_only=False
        )
        full_model = HierarchicalCRNN(
            N_CH, 3, CNN_WIDTH, RNN_HIDDEN, RNN_LAYERS,
            CNN_DROPOUT, RNN_DROPOUT, head_dropout=0.35,
        ).to(self.device)
        full_model.load_state_dict(ckpt_a["model_state"])
        full_model.eval()

        self.detect_model = full_model
        self.frozen_backbone = full_model.backbone
        for p in self.frozen_backbone.parameters():
            p.requires_grad = False

        ckpt_ft = torch.load(
            finetuned_ckpt, map_location=self.device, weights_only=False
        )
        ft_model = HierarchicalCRNN(
            N_CH, 3, CNN_WIDTH, RNN_HIDDEN, RNN_LAYERS,
            CNN_DROPOUT, RNN_DROPOUT, head_dropout=0.5,
        ).to(self.device)
        ft_model.load_state_dict(ckpt_ft["model_state"])
        self.finetuned_backbone = ft_model.backbone
        for p in self.finetuned_backbone.parameters():
            p.requires_grad = False
        del ft_model

        if classifier_pkl.exists():
            with open(classifier_pkl, "rb") as f:
                self.classifier = pickle.load(f)

        self._loaded = True

    def preprocess_edf(self, edf_path, line_freq=50.0):
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        ren = {}
        for ch in raw.ch_names:
            c = _norm_name(ch)
            if c and c != ch and c not in ren.values():
                ren[ch] = c
        if ren:
            raw.rename_channels(ren)

        drop = [c for c in raw.ch_names if _is_non_eeg(c)]
        if drop:
            raw.drop_channels(drop)

        has_bipolar = any("-" in c for c in raw.ch_names)

        if has_bipolar:
            raw = self._resolve_bipolar(raw)
        else:
            raw = self._derive_bipolar(raw)

        missing = [c for c in CANONICAL_18 if c not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Cannot find required channels: {missing}. "
                f"Available: {raw.ch_names}"
            )

        raw.pick_channels(CANONICAL_18, ordered=True)

        sfreq = float(raw.info["sfreq"])
        notches = [line_freq * k for k in (1, 2, 3) if line_freq * k < sfreq / 2]
        if notches:
            raw.notch_filter(freqs=notches, verbose=False)
        raw.filter(
            BP_LOW, BP_HIGH, method="fir", phase="zero",
            fir_design="firwin", verbose=False,
        )
        if not np.isclose(raw.info["sfreq"], TARGET_SFREQ):
            raw.resample(TARGET_SFREQ, verbose=False)

        signal = raw.get_data().astype(np.float32)
        del raw
        return signal

    @staticmethod
    def _classify_duplicate(raw, label):
        copies = [
            c for c in raw.ch_names if c == label or c.startswith(label + "-")
        ]
        if len(copies) < 2:
            return "unknown", copies
        d = raw.get_data(picks=copies[:2])
        if np.std(d[0]) < 1e-15 and np.std(d[1]) < 1e-15:
            return "placeholder", copies
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(d[0], d[1])[0, 1]
        if not np.isfinite(corr):
            return "placeholder", copies
        return ("true_duplicate" if corr > 0.999 else "ambiguous"), copies

    def _resolve_duplicates(self, raw):
        from collections import Counter

        base = [
            n.rsplit("-", 1)[0] if n.rsplit("-", 1)[-1].isdigit() else n
            for n in raw.ch_names
        ]
        dupes = [label for label, n in Counter(base).items() if n > 1]
        for label in dupes:
            if label not in raw.ch_names and not any(
                c.startswith(label + "-") for c in raw.ch_names
            ):
                continue
            cat, copies = self._classify_duplicate(raw, label)
            if cat == "ambiguous":
                raise ValueError(
                    f"ambiguous duplicate '{label}' "
                    f"(montage changed mid-recording)"
                )
            if cat == "placeholder":
                raw.drop_channels(copies)
            elif cat == "true_duplicate":
                raw.drop_channels(copies[1:])
                if copies[0] != label:
                    raw.rename_channels({copies[0]: label})
        return raw

    def _resolve_bipolar(self, raw):
        raw = self._resolve_duplicates(raw)

        rename_map = {}
        existing = {c for c in raw.ch_names}
        for canon in CANONICAL_18:
            if canon in existing:
                continue
            for ch in raw.ch_names:
                if ch.upper() == canon.upper():
                    rename_map[ch] = canon
                    break
        if rename_map:
            raw.rename_channels(rename_map)

        keep = [c for c in raw.ch_names if c in CANONICAL_18]
        if len(keep) < N_CH:
            found = {c.upper(): c for c in keep}
            for canon in CANONICAL_18:
                if canon.upper() not in found:
                    for ch in raw.ch_names:
                        if ch.upper() == canon.upper() and ch not in keep:
                            keep.append(ch)
                            break
            raw.pick(keep)
            raw.rename_channels({ch: ch for ch in keep})

        return raw

    def _derive_bipolar(self, raw):
        by_upper = {}
        for ch in raw.ch_names:
            by_upper.setdefault(ch.upper(), ch)

        resolve = lambda e: by_upper.get(e.upper())

        missing = [
            n for n in CANONICAL_18
            if any(resolve(e) is None for e in BIPOLAR_MAP[n])
        ]
        if missing:
            raise ValueError(
                f"cannot derive {missing}; have {sorted(raw.ch_names)}"
            )

        cache = {ch: raw.get_data(picks=[ch])[0] for ch in set(raw.ch_names)}
        data = np.stack([
            cache[resolve(BIPOLAR_MAP[n][0])]
            - cache[resolve(BIPOLAR_MAP[n][1])]
            for n in CANONICAL_18
        ])
        info = mne.create_info(
            CANONICAL_18, sfreq=float(raw.info["sfreq"]), ch_types="eeg"
        )
        raw_new = mne.io.RawArray(data, info, verbose=False)
        return raw_new

    def _windows_from_signal(self, signal, window_n, stride_n):
        n_samples = signal.shape[1]
        windows = []
        starts = []
        for start in range(0, n_samples - window_n + 1, stride_n):
            w = signal[:, start: start + window_n].astype(np.float32)
            mu = w.mean(axis=1, keepdims=True)
            sd = w.std(axis=1, keepdims=True) + 1e-6
            w = (w - mu) / sd
            windows.append(w)
            starts.append(start)
        return (
            np.stack(windows) if windows else np.empty((0, N_CH, window_n)),
            starts,
        )

    @torch.no_grad()
    def _detect_batch(self, windows):
        if len(windows) == 0:
            return np.array([])
        x = torch.from_numpy(windows).to(self.device)
        logits = self.detect_model.detect_head(
            self.frozen_backbone(x)
        ).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    @torch.no_grad()
    def _embed_dual(self, windows):
        if len(windows) == 0:
            return np.empty((0, 256))
        x = torch.from_numpy(windows).to(self.device)
        e1 = self.frozen_backbone(x)
        e2 = self.finetuned_backbone(x)
        return torch.cat([e1, e2], dim=1).cpu().numpy()

    def detect_seizures(self, signal, threshold=0.5, batch_size=256,
                        min_event_s=8.0):
        stride_n = STAGE1_N
        windows, starts = self._windows_from_signal(
            signal, STAGE1_N, stride_n
        )
        if len(windows) == 0:
            return []

        all_probs = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i: i + batch_size]
            probs = self._detect_batch(batch)
            all_probs.append(probs)
        all_probs = np.concatenate(all_probs)

        seizure_mask = all_probs > threshold
        events = []
        in_event = False
        ev_start = 0

        for i, is_seiz in enumerate(seizure_mask):
            if is_seiz and not in_event:
                in_event = True
                ev_start = starts[i]
            elif not is_seiz and in_event:
                in_event = False
                ev_end = starts[i - 1] + STAGE1_N
                if (ev_end - ev_start) >= min_event_s * TARGET_SFREQ:
                    events.append(
                        (ev_start / TARGET_SFREQ, ev_end / TARGET_SFREQ)
                    )

        if in_event:
            ev_end = starts[-1] + STAGE1_N
            if (ev_end - ev_start) >= min_event_s * TARGET_SFREQ:
                events.append((ev_start / TARGET_SFREQ, ev_end / TARGET_SFREQ))

        merged = []
        for s, e in events:
            if merged and s - merged[-1][1] < 2.0:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        return merged

    def classify_subtypes(self, signal, seizure_events, batch_size=128):
        if self.classifier is None:
            return [
                {
                    "start_s": s,
                    "end_s": e,
                    "duration_s": e - s,
                    "predicted_class": "Unknown",
                    "confidence": 0.0,
                    "probabilities": {k: 0.0 for k in CLASS_KEYS},
                }
                for s, e in seizure_events
            ]

        results = []
        for ev_start_s, ev_end_s in seizure_events:
            ev_start = int(ev_start_s * TARGET_SFREQ)
            ev_end = int(ev_end_s * TARGET_SFREQ)

            stride_n = int(STAGE2_STRIDE_S * TARGET_SFREQ)
            windows = []
            window_starts = []

            for s in range(ev_start, ev_end - STAGE2_N + 1, stride_n):
                w = signal[:, s: s + STAGE2_N].astype(np.float32)
                mu = w.mean(axis=1, keepdims=True)
                sd = w.std(axis=1, keepdims=True) + 1e-6
                w = (w - mu) / sd
                windows.append(w)
                window_starts.append(s)

            if not windows:
                duration = ev_end_s - ev_start_s
                windows_arr = signal[:, ev_start: ev_start + STAGE2_N]
                if windows_arr.shape[1] < STAGE2_N:
                    windows_arr = np.pad(
                        windows_arr,
                        ((0, 0), (0, STAGE2_N - windows_arr.shape[1])),
                        mode="edge",
                    )
                mu = windows_arr.mean(axis=1, keepdims=True)
                sd = windows_arr.std(axis=1, keepdims=True) + 1e-6
                windows_arr = (
                    (windows_arr - mu) / sd
                ).astype(np.float32)
                windows = [windows_arr]
                window_starts = [ev_start]

            windows_np = np.stack(windows)

            all_embs = []
            for i in range(0, len(windows_np), batch_size):
                batch = windows_np[i: i + batch_size]
                embs = self._embed_dual(batch)
                all_embs.append(embs)
            all_embs = np.concatenate(all_embs)

            preds = self.classifier.predict(all_embs)
            probas = self.classifier.predict_proba(all_embs)

            unique, counts = np.unique(preds, return_counts=True)
            majority_idx = unique[np.argmax(counts)]
            majority_conf = counts.max() / counts.sum()

            mean_probas = probas.mean(axis=0)

            results.append({
                "start_s": float(ev_start_s),
                "end_s": float(ev_end_s),
                "duration_s": float(ev_end_s - ev_start_s),
                "predicted_class": CLASS_NAMES[majority_idx],
                "predicted_class_key": CLASS_KEYS[majority_idx],
                "confidence": float(majority_conf),
                "probabilities": {
                    CLASS_KEYS[i]: float(mean_probas[i])
                    for i in range(len(CLASS_KEYS))
                },
            })

        return results

    def analyze(self, edf_path, line_freq=50.0, detection_threshold=0.5,
                min_event_s=8.0):
        self.load_models()

        signal = self.preprocess_edf(edf_path, line_freq)

        duration_s = signal.shape[1] / TARGET_SFREQ
        n_channels = signal.shape[0]

        seizure_events = self.detect_seizures(
            signal, detection_threshold, min_event_s=min_event_s
        )

        classifications = []
        if seizure_events:
            classifications = self.classify_subtypes(signal, seizure_events)

        downsample_factor = 4
        ds_sfreq = TARGET_SFREQ / downsample_factor
        traces = []
        for ch in range(n_channels):
            ch_data = signal[ch, ::downsample_factor] * 1e6
            traces.append(ch_data.tolist())

        seizure_regions = [
            {
                "start_s": c["start_s"],
                "end_s": c["end_s"],
                "duration_s": c["duration_s"],
                "classification": c["predicted_class"],
                "confidence": c["confidence"],
                "probabilities": c["probabilities"],
            }
            for c in classifications
        ]

        overall_class = None
        overall_conf = 0.0
        if classifications:
            best = max(classifications, key=lambda x: x["confidence"])
            overall_class = best["predicted_class"]
            overall_conf = best["confidence"]

        return {
            "file_info": {
                "name": Path(edf_path).name,
                "duration_s": round(duration_s, 1),
                "n_channels": n_channels,
                "channels": CANONICAL_18[:n_channels],
                "sfreq": TARGET_SFREQ,
            },
            "eeg_data": {
                "downsampled_sfreq": ds_sfreq,
                "traces": traces,
                "voltage_scale_uV": 1.0,
            },
            "seizure_detected": len(seizure_events) > 0,
            "n_seizure_events": len(seizure_events),
            "seizure_regions": seizure_regions,
            "summary": {
                "total_seizure_duration_s": round(
                    sum(c["duration_s"] for c in classifications), 1
                ),
                "n_seizure_events": len(seizure_events),
                "overall_classification": overall_class,
                "overall_confidence": round(overall_conf, 3),
            },
        }
