"""Data plumbing for one-stage training.

Two streams feed the same optimizer step:

  1. `ConversationDataset` — one IEMOCAP conversation per item, carrying BOTH
     the raw transcripts (tokenized on the fly) and the cached frozen-CARE
     embeddings for every utterance in it. A conversation is the smallest unit
     that Stage II / III can consume, and the Stage I losses are computed on the
     utterances inside it, so a single batch drives all five IEMOCAP losses.

  2. `AuxTextDataset` — flat MSP-PODCAST transcripts with GPT-3.5 silver labels,
     the paper's LLM-supervised pre-training corpus. Text only: the audio side
     of MSP-PODCAST was consumed by CARE's own pre-training, which is already
     done and frozen.

The MSP stream is ~35x larger than IEMOCAP (149K utterances vs 4.3K), so it is
consumed as an endless cycle sampled every `msp_every` steps rather than as an
epoch-aligned loader — see `cycle`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
#  IEMOCAP — conversations
# ---------------------------------------------------------------------------
class ConversationDataset(Dataset):
    """One item = one conversation = (transcripts, CARE features, labels).

    Manifest columns (produced by merits-l-text's `scripts/preprocess_iemocap.py`):
        utt_id, dialogue_id, session, text, raw_emotion, label, split

    `care_features` / `care_pooled` come from `iemocap_care_downstream.pt`, the
    cache written by care-training's `extract_iemocap_care_downstream_style.py`:
        {"features": {utt_id: (13, 1536)}, "pooled": {utt_id: (768,)}}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        care_features: Dict[str, torch.Tensor],
        care_pooled: Dict[str, torch.Tensor],
        text_col: str = "text",
        label_col: str = "label",
        utt_col: str = "utt_id",
        dialogue_col: str = "dialogue_id",
    ) -> None:
        df = pd.read_csv(manifest_path)
        for col in (text_col, label_col, utt_col, dialogue_col):
            if col not in df.columns:
                raise KeyError(f"{manifest_path}: missing column `{col}`")

        self.care_features = care_features
        self.care_pooled = care_pooled

        self.dialogues: List[Dict] = []
        n_dropped = 0
        for did, group in df.groupby(dialogue_col, sort=False):
            rows = [
                (str(u), str(t), int(l))
                for u, t, l in zip(
                    group[utt_col].astype(str),
                    group[text_col].astype(str),
                    group[label_col].astype(int),
                )
                if str(u) in care_features and str(u) in care_pooled
            ]
            n_dropped += len(group) - len(rows)
            if not rows:
                continue
            self.dialogues.append({
                "dialogue_id": str(did),
                "utt_ids": [r[0] for r in rows],
                "texts": [r[1] for r in rows],
                "labels": [r[2] for r in rows],
            })
        if n_dropped:
            print(f"[ConversationDataset] {n_dropped} utterances had no CARE feature "
                  f"and were dropped ({Path(manifest_path).name}).")

    def __len__(self) -> int:
        return len(self.dialogues)

    def num_utterances(self) -> int:
        return sum(len(d["utt_ids"]) for d in self.dialogues)

    def max_conversation_length(self) -> int:
        return max((len(d["utt_ids"]) for d in self.dialogues), default=0)

    def __getitem__(self, idx: int) -> Dict:
        d = self.dialogues[idx]
        return {
            "dialogue_id": d["dialogue_id"],
            "utt_ids": d["utt_ids"],
            "texts": d["texts"],
            "care_features": torch.stack([self.care_features[u] for u in d["utt_ids"]]),
            "care_pooled": torch.stack([self.care_pooled[u] for u in d["utt_ids"]]),
            "labels": torch.tensor(d["labels"], dtype=torch.long),
        }


def make_conversation_collate(tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
    """Pad to (B, K, L): K = longest conversation, L = longest transcript.

    Padded conversation slots get `label = -100` and `mask = False`; the model
    never runs the encoder on them (see `OneStageMERITSL.forward_conversation`).
    """

    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        K = max(len(item["texts"]) for item in batch)

        flat_texts: List[str] = []
        for item in batch:
            flat_texts.extend(item["texts"])
        enc = tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        L = enc["input_ids"].size(1)

        input_ids = torch.full((B, K, L), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, K, L), dtype=torch.long)
        labels = torch.full((B, K), -100, dtype=torch.long)
        mask = torch.zeros((B, K), dtype=torch.bool)

        feat_shape = batch[0]["care_features"].shape[1:]      # (num_layers, layer_dim)
        pooled_dim = batch[0]["care_pooled"].size(-1)
        care_features = torch.zeros((B, K, *feat_shape), dtype=torch.float32)
        care_pooled = torch.zeros((B, K, pooled_dim), dtype=torch.float32)

        utt_ids: List[List[str]] = []
        dialogue_ids: List[str] = []
        cursor = 0
        for i, item in enumerate(batch):
            k = len(item["texts"])
            input_ids[i, :k] = enc["input_ids"][cursor:cursor + k]
            attention_mask[i, :k] = enc["attention_mask"][cursor:cursor + k]
            care_features[i, :k] = item["care_features"].float()
            care_pooled[i, :k] = item["care_pooled"].float()
            labels[i, :k] = item["labels"]
            mask[i, :k] = True
            utt_ids.append(item["utt_ids"])
            dialogue_ids.append(item["dialogue_id"])
            cursor += k

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "care_features": care_features,
            "care_pooled": care_pooled,
            "labels": labels,
            "mask": mask,
            "utt_ids": utt_ids,
            "dialogue_ids": dialogue_ids,
        }

    return _collate


def load_care_cache(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("features", "pooled"):
        if key not in payload:
            raise KeyError(
                f"{path}: expected keys 'features' and 'pooled' (the layout written by "
                f"care-training's extract_iemocap_care_downstream_style.py), got "
                f"{sorted(payload.keys())}."
            )
    return payload["features"], payload["pooled"]


def build_conversation_loaders(
    manifest_dir: str | Path,
    care_cache_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    eval_batch_size: int,
    max_length: int = 128,
    num_workers: int = 2,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    manifest_dir = Path(manifest_dir)
    features, pooled = load_care_cache(care_cache_path)
    sample = next(iter(features.values()))
    print(f"Loaded CARE cache: {len(features)} utterances, features {tuple(sample.shape)}, "
          f"pooled ({next(iter(pooled.values())).size(-1)},)")

    collate = make_conversation_collate(tokenizer, max_length=max_length)
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        ds = ConversationDataset(csv_path, features, pooled)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=False,
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders


# ---------------------------------------------------------------------------
#  MSP-PODCAST — flat transcripts with LLM silver labels
# ---------------------------------------------------------------------------
class AuxTextDataset(Dataset):
    """(transcript, silver label) pairs. Tokenization happens in the collate fn."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        return {"text": self.texts[idx], "label": self.labels[idx]}


def make_aux_collate(tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
    def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            [b["text"] for b in batch],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }
    return _collate


def build_aux_loaders(
    manifest_path: str | Path,
    label_map: Dict[str, int],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    eval_batch_size: int,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_length: int = 128,
    num_workers: int = 2,
    max_train_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """80/20 split of the pseudo-label CSV, matching the paper's pre-training split.

    The split is drawn with its own fixed `seed` so that changing the *training*
    seed for a multi-seed sweep does not silently move MSP utterances between
    train and val.
    """
    df = pd.read_csv(manifest_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise KeyError(
            f"{manifest_path}: missing columns {missing}. Expecting (utt_id,text,label) "
            f"as produced by merits-l-text's scripts/llm_pseudo_label_msp.py."
        )
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label_str"] = df["label"].astype(str).str.strip().str.lower()
    n_before = len(df)
    df = df[df["label_str"].isin(label_map)].reset_index(drop=True)
    if len(df) < n_before:
        print(f"[AuxTextDataset] dropped {n_before - len(df)} rows with labels outside "
              f"{sorted(label_map)}")
    df["label_int"] = df["label_str"].map(label_map).astype(int)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(len(df) * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if max_train_samples is not None:
        train_idx = train_idx[:max_train_samples]

    collate = make_aux_collate(tokenizer, max_length=max_length)
    loaders: Dict[str, DataLoader] = {}
    for split, sel, bs, shuffle in (
        ("train", train_idx, batch_size, True),
        ("val", val_idx, eval_batch_size, False),
    ):
        sub = df.iloc[sel]
        ds = AuxTextDataset(sub["text"].tolist(), sub["label_int"].tolist())
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, collate_fn=collate, drop_last=shuffle,
        )
    print(f"MSP-PODCAST silver labels: train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)}")
    return loaders


def cycle(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Endless iterator over a DataLoader, reshuffling at each pass.

    The auxiliary stream is not epoch-aligned with IEMOCAP — one IEMOCAP epoch
    is ~60 steps while MSP has ~3700 batches — so it is pulled on demand.
    """
    while True:
        for batch in loader:
            yield batch
