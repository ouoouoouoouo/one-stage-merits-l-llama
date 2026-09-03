"""Run MSP-PODCAST through a trained one-stage model and cache every stage's output.

Cross-corpus transfer check: the model was trained on IEMOCAP 4-way and is
frozen here. One forward pass yields five representations per utterance, so a
single extraction supports probing all of them:

    T1 (4096)  Llama + LoRA utterance embedding
    S1 (256)   CARE Stage I head output
    T2 (2048)  text Stage II, conversation-contextualised
    S2 (256)   audio Stage II
    h  (256)   Stage III fused hidden

Two mismatches between MSP-PODCAST and the architecture, both handled here:

  * MSP has no dialogues. Utterances are grouped into pseudo-conversations by
    podcast show (`MSP-PODCAST_<show>_<segment>`), in segment order — 4,970
    shows, median 15 segments, which is the same order of magnitude as
    IEMOCAP's ~35. With `--no-group` each utterance becomes a length-1
    conversation instead, which makes Stage II/III near-identity; that is the
    honest ablation for how much the conversation blocks contribute.
  * MSP has no CARE cache. The CARE encoder is loaded and run inline rather
    than dumped to disk first — the full 161 K-utterance cache would be ~13 GB
    and is needed only transiently.

Caveat worth carrying into any writeup: CARE was self-supervised on
MSP-PODCAST v1.11, so parts of this test set were seen (unlabelled) during its
pre-training. The text branch is clean when probing a `lambda_aux = 0` model.

Usage:
    python -m scripts.extract_msp_representations \
        --checkpoint outputs/nomsp/seed_1/best/one_stage.pt \
        --manifest /home/ouo/AdaLTM-reproduction/8class_DropTextNAN.csv \
        --audio-dir /home/ouo/dataset/MSP_Podcast/Audios \
        --care-ckpt /home/ouo/care_training/ckpts_faithful/best.pth \
        --care-repo /home/ouo/care_training/CARE/pretraining \
        --extract-script /home/ouo/care_training/care-training/scripts/extract_iemocap_care_downstream_style.py \
        --out data/cache/msp_reps_nomsp_seed1.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.one_stage_model import build_model      # noqa: E402
from src.utils.config import AttrDict                    # noqa: E402

# utils/data/podcast.py in AdaLTM, kept identical so labels line up with theirs.
LABEL_MAP = {"A": 0, "C": 1, "D": 2, "F": 3, "H": 4, "N": 5, "S": 6, "U": 7}
LABEL_NAMES = ["Anger", "Contempt", "Disgust", "Fear",
               "Happiness", "Neutral", "Sadness", "Surprise"]
_SHOW_RE = re.compile(r"MSP-PODCAST_(\d+)_(\d+)")


def load_care_extractor(extract_script: str, care_ckpt: str, care_repo: str, device: str):
    """Import care-training's CAREDownstreamExtractor from its script path.

    The script is not an importable package on the cluster, so it is loaded by
    file location. It is the reference implementation that produced the paper's
    Table IV numbers, and reusing it verbatim keeps the MSP features identical
    in kind to the cached IEMOCAP ones.
    """
    path = Path(extract_script)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Point --extract-script at care-training's "
            f"scripts/extract_iemocap_care_downstream_style.py."
        )
    spec = importlib.util.spec_from_file_location("_care_extract", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CAREDownstreamExtractor(care_ckpt, care_repo, device)


def read_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def build_conversations(df: pd.DataFrame, group_by_show: bool, max_len: int) -> List[Dict]:
    """Pseudo-conversations, in segment order within each (podcast show, split).

    Grouping is by show AND split, never across splits. 1,479 of MSP's 4,970
    shows span more than one Split_Set, covering 77,188 utterances — grouping on
    the show alone would put Train and Test1 utterances in one conversation, and
    Stage II/III contextualise across the conversation, so a test utterance's
    T2/S2/h would be computed partly from training utterances. No label leaks
    (the model is frozen here), but the representation would not be computable
    from test data alone, which is not an evaluation anyone can deploy.

    It also makes sharded and single-pass extraction produce identical output,
    so splitting the job across GPUs is purely a speed decision.
    """
    if not group_by_show:
        return [{"utts": [r.FileName], "texts": [r.text], "labels": [LABEL_MAP[r.EmoClass]],
                 "splits": [r.Split_Set]} for r in df.itertuples()]

    parsed = df["FileName"].str.extract(_SHOW_RE)
    df = df.assign(_show=parsed[0], _seg=parsed[1].astype(int))
    df = df.sort_values(["_show", "Split_Set", "_seg"])

    convs: List[Dict] = []
    for _, g in df.groupby(["_show", "Split_Set"], sort=False):
        rows = list(g.itertuples())
        # A show can run to 1020 segments; chunk so one forward stays bounded.
        for i in range(0, len(rows), max_len):
            chunk = rows[i:i + max_len]
            convs.append({
                "utts": [r.FileName for r in chunk],
                "texts": [str(r.text) for r in chunk],
                "labels": [LABEL_MAP[r.EmoClass] for r in chunk],
                "splits": [r.Split_Set for r in chunk],
            })
    return convs


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="outputs/<arm>/seed_N/best/one_stage.pt")
    ap.add_argument("--manifest", required=True, help="AdaLTM 8class_DropTextNAN.csv")
    # Only needed for the audio path; --only-t1 / --lora-adapter skip it entirely.
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--care-ckpt", default=None)
    ap.add_argument("--care-repo", default=None)
    ap.add_argument("--extract-script", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="restrict to these Split_Set values (default: all)")
    ap.add_argument("--max-conv-len", type=int, default=32,
                    help="cap on utterances per pseudo-conversation")
    ap.add_argument("--text-chunk", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--no-group", action="store_true",
                    help="one utterance per conversation (Stage II/III become near-identity)")
    ap.add_argument("--lora-adapter", default=None,
                    help="peft adapter directory to swap in for the checkpoint's own "
                         "LoRA weights, e.g. merits-l-llama's staged Stage I "
                         "outputs/iemocap_text_llama_stage1/best. --checkpoint still "
                         "supplies the architecture, so both runs are identical apart "
                         "from the adapter. Implies --only-t1.")
    ap.add_argument("--only-t1", action="store_true",
                    help="extract just the Llama utterance embedding: skips the CARE "
                         "encoder, the wav reads and Stage II/III entirely")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N conversations")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- trained model, frozen ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = AttrDict(dict(ckpt["model_cfg"]))
    print(f"Rebuilding model from {args.checkpoint} "
          f"(epoch {ckpt.get('epoch')}, seed {ckpt.get('seed')}, "
          f"val score {ckpt.get('score')})")
    model = build_model(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has tensors the model does not: {unexpected[:5]}")
    n_restored = len(ckpt["model_state_dict"])
    print(f"restored {n_restored} tensors; {len(missing)} left at their pretrained values "
          f"(the frozen Llama base)")

    only_t1 = args.only_t1
    if args.lora_adapter:
        # Task-vector comparison: same architecture, same base, same LoRA
        # subspace (r / alpha / target_modules must match), different adapter.
        # Only T1 is meaningful afterwards — the checkpoint's audio head and
        # Stage II/III belong to a different training run than this adapter.
        base = model.text_stage1.base
        if not hasattr(base, "load_adapter"):
            raise RuntimeError("--lora-adapter needs a peft-wrapped base "
                               "(model.text.use_lora must be true)")
        base.load_adapter(args.lora_adapter, adapter_name="default")
        for _, p in base.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        print(f"swapped in the LoRA adapter from {args.lora_adapter}; "
              f"extracting T1 only")
        only_t1 = True

    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_cfg.text.model_id), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- CARE encoder, frozen ----
    care = None
    if not only_t1:
        need = {"--audio-dir": args.audio_dir, "--care-ckpt": args.care_ckpt,
                "--care-repo": args.care_repo, "--extract-script": args.extract_script}
        absent = [k for k, v in need.items() if not v]
        if absent:
            raise SystemExit(f"the audio path needs {', '.join(absent)} "
                             f"(or pass --only-t1 to skip it)")
        care = load_care_extractor(args.extract_script, args.care_ckpt,
                                   args.care_repo, args.device)

    # ---- manifest ----
    df = pd.read_csv(args.manifest)
    df = df[df["EmoClass"].isin(LABEL_MAP)]
    if args.splits:
        df = df[df["Split_Set"].isin(args.splits)]
    convs = build_conversations(df, group_by_show=not args.no_group,
                                max_len=args.max_conv_len)
    if args.limit:
        convs = convs[:args.limit]
    print(f"{len(df)} utterances -> {len(convs)} "
          f"{'pseudo-conversations' if not args.no_group else 'single-utterance items'}")

    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    out: Dict[str, Dict] = {}
    n_ok = n_missing_audio = n_failed = 0

    for conv in tqdm(convs, desc="extract"):
        # --- audio: CARE features for every utterance we can find on disk ---
        if only_t1:
            # No audio path at all: every utterance survives, and the wav reads
            # (the slow part) are skipped.
            keep, feats, pooled = list(range(len(conv["utts"]))), [], []
        else:
            keep, feats, pooled = [], [], []
            for i, utt in enumerate(conv["utts"]):
                wav_path = audio_dir / utt
                if not wav_path.exists():
                    n_missing_audio += 1
                    continue
                try:
                    f, p = care.extract(read_wav(wav_path))
                except Exception as e:  # noqa: BLE001
                    print(f"\n[warn] {utt}: {e}")
                    n_failed += 1
                    continue
                keep.append(i)
                feats.append(f)
                pooled.append(p)
        if not keep:
            continue

        texts = [conv["texts"][i] for i in keep]
        enc = tok(texts, padding=True, truncation=True,
                  max_length=args.max_length, return_tensors="pt")
        K = len(keep)
        mask = torch.ones(1, K, dtype=torch.bool, device=device)

        # Reach inside the model rather than calling forward_conversation, which
        # would need labels and would not hand back the intermediate tensors.
        ids = enc["input_ids"].to(device)
        att = enc["attention_mask"].to(device)
        if args.text_chunk and K > args.text_chunk:
            t1 = torch.cat([model.text_stage1.get_features(ids[i:i + args.text_chunk],
                                                           att[i:i + args.text_chunk])
                            for i in range(0, K, args.text_chunk)], dim=0)
        else:
            t1 = model.text_stage1.get_features(ids, att)

        if only_t1:
            for j, i in enumerate(keep):
                out[conv["utts"][i]] = {
                    "t1": t1[j].half().cpu(),
                    "label": conv["labels"][i], "split": conv["splits"][i],
                }
            n_ok += len(keep)
            continue

        s1 = model.audio_stage1.get_features(
            torch.stack(feats).to(device), torch.stack(pooled).to(device))
        t2 = model.text_stage2.encode(t1.unsqueeze(0), mask)[0]
        s2 = model.audio_stage2.encode(s1.unsqueeze(0), mask)[0]
        h = model.fusion(t2.unsqueeze(0), s2.unsqueeze(0), mask)["fused_hidden"][0]

        for j, i in enumerate(keep):
            out[conv["utts"][i]] = {
                "t1": t1[j].half().cpu(), "s1": s1[j].half().cpu(),
                "t2": t2[j].half().cpu(), "s2": s2[j].half().cpu(),
                "h": h[j].half().cpu(),
                "label": conv["labels"][i], "split": conv["splits"][i],
            }
        n_ok += len(keep)

    print(f"\nextracted {n_ok} utterances | {n_missing_audio} missing audio "
          f"(v1.12 manifest against a v1.11 audio directory) | {n_failed} failed")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "reps": out,
        "label_names": LABEL_NAMES,
        # The adapter is what identifies a task-vector run; without one the
        # checkpoint identifies it.
        "checkpoint": str(args.lora_adapter or args.checkpoint),
        "architecture_from": str(args.checkpoint),
        "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
        "only_t1": only_t1,
        "grouped_by_show": not args.no_group,
        "max_conv_len": args.max_conv_len,
    }, out_path)
    print(f"saved {out_path}  ({out_path.stat().st_size / 1024**3:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
