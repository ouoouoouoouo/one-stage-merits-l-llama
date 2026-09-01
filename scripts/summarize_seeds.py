"""Aggregate the per-seed result.json files a sweep leaves behind.

    python -m scripts.summarize_seeds outputs/msp
    python -m scripts.summarize_seeds outputs/msp outputs/nomsp --latex

Reports mean +/- std over seeds for every head, the form the staged baselines
are quoted in — a single run is not comparable to them.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

HEADS = ("fusion", "text_stage2", "audio_stage2", "text_stage1", "audio_stage1")

# Context rows. merits-l-llama figures are 5-seed means of the staged pipeline;
# the RoBERTa one-stage figure comes from the sibling repo one-stage-merits-l.
REFERENCE = {
    "staged Llama (5 seeds)":     {"fusion": 0.8567},
    "staged Llama + MSP pre":     {"fusion": 0.8550},
    "one-stage RoBERTa (5 seeds)": {"fusion": 0.8281, "text_stage2": 0.8025,
                                    "audio_stage2": 0.6374, "text_stage1": 0.6739,
                                    "audio_stage1": 0.5547},
}
MAIN_BASELINE = ("staged Llama (5 seeds)", 0.8567)


def load(run_dir: Path) -> List[Dict]:
    results = []
    for path in sorted(run_dir.glob("**/result.json")):
        try:
            # utf-8-sig: tolerate a BOM if a file was ever touched on Windows.
            results.append(json.loads(path.read_text(encoding="utf-8-sig")))
        except json.JSONDecodeError:
            print(f"[warn] {path} is not valid JSON, skipped")
    return results


def summarize(run_dir: Path, latex: bool) -> None:
    results = load(run_dir)
    if not results:
        print(f"{run_dir}: no result.json found — did the runs finish?")
        return

    print(f"\n{run_dir}  ({len(results)} seeds)")
    print("-" * 84)
    print(f"{'seed':>5}  {'epoch':>5}  {'val fus':>8}  " +
          "  ".join(f"{h:>12}" for h in HEADS))
    for r in sorted(results, key=lambda x: x["seed"]):
        row = "  ".join(f"{r['test'][h]['weighted_f1']:>12.4f}" for h in HEADS)
        print(f"{r['seed']:>5}  {r['best_epoch']:>5}  "
              f"{r['val_fusion_weighted_f1']:>8.4f}  {row}")

    print("-" * 84)
    stats = {}
    for h in HEADS:
        vals = [r["test"][h]["weighted_f1"] for r in results]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        stats[h] = (mean, std, max(vals))
    for label, idx in (("mean", 0), ("std", 1), ("best", 2)):
        print(f"{label:>5}  {'':>5}  {'':>8}  " +
              "  ".join(f"{stats[h][idx]:>12.4f}" for h in HEADS))

    print("-" * 84)
    for name, ref in REFERENCE.items():
        cells = [f"{ref[h]:>12.4f}" if h in ref else f"{'-':>12}" for h in HEADS]
        print(f"{name:<22}  " + "  ".join(cells))

    fus_mean, fus_std, fus_best = stats["fusion"]
    base_name, base = MAIN_BASELINE
    print(f"\nfusion mean {fus_mean:.4f} +/- {fus_std:.4f} (best {fus_best:.4f})   "
          f"vs {base_name} {base:.4f}:  {(fus_mean - base) * 100:+.2f} pp")

    if latex:
        print("\n% LaTeX row")
        print(f"One-stage Llama (ours) & {stats['text_stage1'][0]*100:.2f} & "
              f"{stats['text_stage2'][0]*100:.2f} & "
              f"{stats['audio_stage1'][0]*100:.2f} & "
              f"{stats['audio_stage2'][0]*100:.2f} & "
              f"{fus_mean*100:.2f} $\\pm$ {fus_std*100:.2f} \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    for d in args.run_dirs:
        summarize(d, args.latex)


if __name__ == "__main__":
    main()
