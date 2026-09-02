"""8-class linear probes on the representations a trained one-stage model gives MSP-PODCAST.

The model is frozen; only a fresh 8-way linear layer is fitted on the MSP Train
split. That measures how much of MSP-PODCAST's 8-class emotion structure is
already linearly available in representations learned from IEMOCAP 4-way, at
every depth of the pipeline at once:

    T1  Llama + LoRA utterance embedding      (4096)
    S1  CARE Stage I head                     (256)
    T2  text Stage II, conversation context    (2048)
    S2  audio Stage II                        (256)
    T2||S2  what Stage III actually consumes   (2304)
    h   Stage III fused hidden                 (256)

Protocol follows AdaLTM so the numbers are comparable to theirs: UAR (macro
recall), macro precision and macro-F1 with bootstrap confidence intervals, on
Test1 and Test2, with the same effective-number class weighting — without it a
probe simply predicts Neutral, which is 34 % of Train against Fear's 1.3 %.
Chance on 8 balanced classes is 12.5 % UAR; AdaLTM quotes 35.56 % macro-F1 for
vox-profile's fine-tuned WavLM-large, which is the number to read these against.

Usage:
    python -m scripts.probe_msp_8class --reps data/cache/msp_reps_nomsp_seed1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.seed import set_seed   # noqa: E402

SPLITS = {"train": "Train", "dev": "Development", "test1": "Test1", "test2": "Test2"}
PROBES = ["t1", "s1", "t2", "s2", "t2+s2", "h"]


def class_weights(labels: np.ndarray, num_classes: int = 8) -> torch.Tensor:
    """Effective-number weighting, identical to AdaLTM's utils/data/podcast.py."""
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes)
    n = len(labels)
    beta = (n - 1) / n
    effective = 1.0 - torch.pow(beta, counts)
    w = torch.where(counts > 0, (1.0 - beta) / effective, torch.zeros_like(effective.float()))
    return w / w.sum() * num_classes


def gather(reps: Dict, key: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    """Stack one representation into per-split matrices."""
    parts = {s: [] for s in SPLITS}
    labs = {s: [] for s in SPLITS}
    inv = {v: k for k, v in SPLITS.items()}
    for entry in reps.values():
        s = inv.get(entry["split"])
        if s is None:
            continue
        if "+" in key:
            vec = torch.cat([entry[k] for k in key.split("+")], dim=-1)
        else:
            vec = entry[key]
        parts[s].append(vec)
        labs[s].append(entry["label"])
    X = {s: torch.stack(v).float() for s, v in parts.items() if v}
    y = {s: np.asarray(v) for s, v in labs.items() if v}
    return X, y


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric, n_boot: int, rng) -> Tuple[float, float]:
    """Point estimate and 95 % CI half-width, resampling utterances."""
    point = metric(y_true, y_pred)
    if n_boot <= 0:
        return float(point), float("nan")
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float((hi - lo) / 2)


def fit_probe(X: Dict[str, torch.Tensor], y: Dict[str, np.ndarray], device, args) -> Dict[str, np.ndarray]:
    """Linear (optionally one-hidden-layer) 8-way probe, early stopped on dev macro-F1."""
    d = X["train"].size(1)
    mean = X["train"].mean(0, keepdim=True)
    std = X["train"].std(0, keepdim=True).clamp(min=1e-6)

    def prep(split):
        return ((X[split] - mean) / std).to(device) if args.standardize else X[split].to(device)

    Xd = {s: prep(s) for s in X}
    yd = {s: torch.tensor(y[s], dtype=torch.long, device=device) for s in y}

    if args.hidden > 0:
        probe = nn.Sequential(nn.Linear(d, args.hidden), nn.ReLU(),
                              nn.Dropout(args.dropout), nn.Linear(args.hidden, 8))
    else:
        probe = nn.Linear(d, 8)
    probe = probe.to(device)

    w = class_weights(y["train"]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = Xd["train"].size(0)

    best_dev, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(probe(Xd["train"][idx]), yd["train"][idx], weight=w).backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            dev_pred = probe(Xd["dev"]).argmax(-1).cpu().numpy()
        dev_f1 = f1_score(y["dev"], dev_pred, average="macro", zero_division=0)
        if dev_f1 > best_dev:
            best_dev, bad = dev_f1, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        return {s: probe(Xd[s]).argmax(-1).cpu().numpy() for s in Xd if s.startswith("test")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--probes", nargs="*", default=PROBES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-bootstraps", type=int, default=200)
    ap.add_argument("--standardize", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.reps, map_location="cpu", weights_only=False)
    reps, names = payload["reps"], payload["label_names"]
    print(f"{len(reps)} utterances from {payload['checkpoint']}  "
          f"(grouped_by_show={payload.get('grouped_by_show')})")

    counts: Dict[str, int] = {}
    for e in reps.values():
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    print("splits: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    rng = np.random.default_rng(args.seed)
    metrics = [("UAR", lambda t, p: recall_score(t, p, average="macro", zero_division=0)),
               ("Pre", lambda t, p: precision_score(t, p, average="macro", zero_division=0)),
               ("MaF1", lambda t, p: f1_score(t, p, average="macro", zero_division=0)),
               ("WF1", lambda t, p: f1_score(t, p, average="weighted", zero_division=0))]

    header = f"{'probe':<8}{'dim':>6}  {'split':<6}" + "".join(f"{m:>16}" for m, _ in metrics)
    print("\n" + header)
    print("-" * len(header))
    rows: List[str] = []
    for key in args.probes:
        X, y = gather(reps, key)
        if "train" not in X or "dev" not in X:
            print(f"{key:<8} skipped — need both Train and Development splits")
            continue
        preds = fit_probe(X, y, device, args)
        for split in sorted(preds):
            cells = []
            for _, fn in metrics:
                c, ci = bootstrap_ci(y[split], preds[split], fn, args.num_bootstraps, rng)
                cells.append(f"{c*100:>9.2f} ±{ci*100:>4.2f}")
            line = f"{key:<8}{X['train'].size(1):>6}  {split:<6}" + "".join(cells)
            print(line)
            rows.append(line)

    print("\nchance on 8 classes = 12.50 UAR; AdaLTM quotes 35.56 MaF1 for "
          "vox-profile's fine-tuned WavLM-large")
    print("labels:", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
