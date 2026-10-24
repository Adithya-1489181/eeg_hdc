import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import (
    CRNNBackbone, HierarchicalCRNN, CANONICAL_18, N_CH,
    CNN_WIDTH, RNN_HIDDEN, RNN_LAYERS, CNN_DROPOUT, RNN_DROPOUT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
WIN_ROOT = PROJECT_ROOT / "data" / "harmonized" / "windows"

SEED = 20260803
TEST_FRAC = 0.3
SMOTE_K = 5
SMOTE_RATIO = {"focal_aware": 3, "focal_to_bilateral_tonic_clonic": 5}


def load_manifest():
    with open(WIN_ROOT / "manifest.json") as f:
        return json.load(f)


def load_backbones(device):
    phase_a = (
        CHECKPOINTS / "crnn_hierarchical"
        / "phase_a_backbone_detector.pt"
    )
    finetuned = (
        CHECKPOINTS / "crnn_hierarchical_smote_v2"
        / "finetuned_backbone.pt"
    )

    ckpt_a = torch.load(phase_a, map_location=device, weights_only=False)
    full_a = HierarchicalCRNN(
        N_CH, 3, CNN_WIDTH, RNN_HIDDEN, RNN_LAYERS,
        CNN_DROPOUT, RNN_DROPOUT, head_dropout=0.35,
    ).to(device)
    full_a.load_state_dict(ckpt_a["model_state"])
    frozen = full_a.backbone.to(device).eval()
    for p in frozen.parameters():
        p.requires_grad = False
    del full_a

    ckpt_ft = torch.load(finetuned, map_location=device, weights_only=False)
    full_ft = HierarchicalCRNN(
        N_CH, 3, CNN_WIDTH, RNN_HIDDEN, RNN_LAYERS,
        CNN_DROPOUT, RNN_DROPOUT, head_dropout=0.5,
    ).to(device)
    full_ft.load_state_dict(ckpt_ft["model_state"])
    finetuned_bb = full_ft.backbone.to(device).eval()
    for p in finetuned_bb.parameters():
        p.requires_grad = False
    del full_ft

    return frozen, finetuned_bb


def read_window(path, start, n_timepoints):
    with h5py.File(path, "r") as f:
        sig = f["signal"]
        s = int(start)
        end = min(s + n_timepoints, sig.shape[1])
        w = sig[:, s:end].astype(np.float32)
    if w.shape[1] < n_timepoints:
        w = np.pad(
            w, ((0, 0), (0, n_timepoints - w.shape[1])), mode="edge"
        )
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True) + 1e-6
    return (w - mu) / sd


@torch.no_grad()
def embed_windows(df, frozen, finetuned_bb, n_tp, device, batch=128):
    n = len(df)
    emb = np.empty((n, 256), dtype=np.float32)

    for i in range(0, n, batch):
        chunk = df.iloc[i: i + batch]
        batch_w = np.stack([
            read_window(r.path, r.start, n_tp)
            for r in chunk.itertuples()
        ])
        x = torch.from_numpy(batch_w).to(device)
        e1 = frozen(x).cpu().numpy()
        e2 = finetuned_bb(x).cpu().numpy()
        emb[i: i + len(batch_w)] = np.concatenate([e1, e2], axis=1)

    return emb


def event_split(df, test_frac, seed):
    train_ev, test_ev = set(), set()
    for _, sub_df in df.groupby("subtype"):
        ev_sizes = (
            sub_df.groupby("event_id")
            .size()
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        total = int(ev_sizes.n.sum())
        target_test = total * test_frac
        target_train = total - target_test
        cum_t, cum_v = 0, 0
        t_ids, v_ids = [], []
        for row in ev_sizes.itertuples(index=False):
            if (target_test - cum_v) >= (target_train - cum_t):
                v_ids.append(row.event_id)
                cum_v += row.n
            else:
                t_ids.append(row.event_id)
                cum_t += row.n
        if not v_ids:
            v_ids.append(t_ids.pop())
        if not t_ids:
            t_ids.append(v_ids.pop())
        train_ev.update(t_ids)
        test_ev.update(v_ids)
    return train_ev, test_ev


def smote_resample(X, y, ratio_map, k_neighbors, seed):
    counts = pd.Series(y).value_counts()
    majority = counts.max()
    strategy = {}
    for cls, n in counts.items():
        if n >= majority:
            continue
        from inference import CLASS_NAMES
        name = CLASS_NAMES[cls]
        ratio = ratio_map.get(name, 3)
        strategy[cls] = int(min(majority, n * ratio))
    k = max(1, min(k_neighbors, int(counts.min()) - 1))
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


def main():
    print("Loading manifest...")
    manifest = load_manifest()
    class_to_idx = manifest["stage2"]["class_to_idx"]
    n_tp = int(round(manifest["stage2"]["window_s"] * manifest["sfreq"]))

    print("Loading stage2 windows...")
    s2 = pd.read_parquet(WIN_ROOT / "stage2_windows.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading backbones...")
    frozen, finetuned_bb = load_backbones(device)

    print("Splitting data...")
    train_ev, test_ev = event_split(s2, TEST_FRAC, SEED)
    train_df = s2[s2.event_id.isin(train_ev)].reset_index(drop=True)
    test_df = s2[s2.event_id.isin(test_ev)].reset_index(drop=True)
    print(f"  Train: {len(train_df)} windows | Test: {len(test_df)} windows")

    print("Extracting train embeddings...")
    emb_train = embed_windows(train_df, frozen, finetuned_bb, n_tp, device)
    y_train = train_df.subtype.map(class_to_idx).to_numpy()

    print("Extracting test embeddings...")
    emb_test = embed_windows(test_df, frozen, finetuned_bb, n_tp, device)
    y_test = test_df.subtype.map(class_to_idx).to_numpy()

    print(f"SMOTE resampling (ratio={SMOTE_RATIO})...")
    X_res, y_res = smote_resample(
        emb_train, y_train, SMOTE_RATIO, SMOTE_K, SEED
    )
    print(f"  After SMOTE: {len(X_res)} samples")

    print("Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.5,
        random_state=SEED,
    )
    clf.fit(X_res, y_res)

    train_acc = clf.score(emb_train, y_train)
    test_acc = clf.score(emb_test, y_test)
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    out_dir = CHECKPOINTS / "crnn_hierarchical_smote_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "classifier.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Classifier saved to: {out_path}")


if __name__ == "__main__":
    main()
