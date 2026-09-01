"""One-stage MERITS-L trainer.

A single optimizer driven by six cross-entropies:

    L = l_f  * L_fusion            (Stage III, the scored objective)
      + l_t2 * L_text_stage2       (conversation context, text)
      + l_a2 * L_audio_stage2      (conversation context, audio)
      + l_t1 * L_text_stage1       (utterance, text)
      + l_a1 * L_audio_stage1      (utterance, audio)
      + l_aux(t) * L_msp           (LLM silver labels, decaying)

Model selection uses the FUSION head's validation weighted-F1 only. The other
five terms are auxiliary supervision replacing "train, freeze, move on" — they
must never pick the checkpoint.

Usage:
    python -m src.train_joint --config configs/one_stage_iemocap.yaml
    python -m src.train_joint --config configs/one_stage_iemocap.yaml \
        --override seed=1 loss.lambda_aux=0.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data.joint_dataset import build_aux_loaders, build_conversation_loaders, cycle
from .models.one_stage_model import build_model
from .utils.config import AttrDict, load_config
from .utils.logging import RunLogger
from .utils.metrics import compute_metrics, detailed_report
from .utils.seed import set_seed

# Workers tokenize inside the collate fn; HF's own parallelism on top of that
# only produces a warning and contention.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The five IEMOCAP terms, in the order they are logged and reported.
STAGE_KEYS = (
    "fusion",
    "text_stage2",
    "audio_stage2",
    "text_stage1",
    "audio_stage1",
)


def _amp_dtype(name: str) -> Optional[torch.dtype]:
    name = str(name).lower()
    if name in ("none", "null", "fp32", "float32", ""):
        return None
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unknown train.amp_dtype: {name!r}")


def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    """Only the tensors that actually train: LoRA adapters + every head.

    The frozen Llama base is >99% of the parameters and is byte-identical to the
    published checkpoint, so writing it into every improvement would cost 16 GB
    per save for nothing.
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v.detach().cpu() for n, v in model.state_dict().items()
            if n in trainable_names}


def aux_scale(step: int, total_steps: int, schedule: str, final_scale: float) -> float:
    """Decay factor on lambda_aux.

    The paper pre-trains on MSP-PODCAST for 10 epochs and then never looks at it
    again. In a one-stage run the equivalent is a weight that starts at full
    strength and fades: without it, a 3-class sentiment task on 149K utterances
    keeps competing with a 4-class emotion task on 4.3K all the way to the end.
    """
    schedule = str(schedule).lower()
    if schedule in ("none", "constant"):
        return 1.0
    p = min(max(step / max(1, total_steps), 0.0), 1.0)
    if schedule == "linear":
        decay = 1.0 - p
    elif schedule == "cosine":
        decay = 0.5 * (1.0 + math.cos(math.pi * p))
    else:
        raise ValueError(f"unknown loss.aux_schedule: {schedule!r}")
    return final_scale + (1.0 - final_scale) * decay


@torch.no_grad()
def evaluate(model, loader, device, label_names, amp_dtype, text_chunk: int = 0) -> Dict[str, Dict]:
    """Metrics for every head. `fusion` is the one that selects checkpoints."""
    model.eval()
    preds: Dict[str, List[int]] = {k: [] for k in STAGE_KEYS}
    golds: Dict[str, List[int]] = {k: [] for k in STAGE_KEYS}
    losses: Dict[str, List[float]] = {k: [] for k in STAGE_KEYS}

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        care_features = batch["care_features"].to(device, non_blocking=True)
        care_pooled = batch["care_pooled"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model.forward_conversation(
                input_ids, attention_mask, care_features, care_pooled, mask, labels,
                text_chunk=text_chunk,
            )

        flat_labels = labels.reshape(-1)[out["index"]]
        for key in STAGE_KEYS:
            losses[key].append(float(out[f"loss_{key}"].item()))
            logits = out[f"logits_{key}"].float()
            if key.endswith("stage1"):
                preds[key].extend(logits.argmax(dim=-1).cpu().tolist())
                golds[key].extend(flat_labels.cpu().tolist())
            else:
                preds[key].extend(logits.argmax(dim=-1)[mask].cpu().tolist())
                golds[key].extend(labels[mask].cpu().tolist())

    results: Dict[str, Dict] = {}
    for key in STAGE_KEYS:
        m = compute_metrics(golds[key], preds[key], label_names=label_names)
        m["loss"] = float(np.mean(losses[key])) if losses[key] else float("nan")
        m["_preds"], m["_labels"] = preds[key], golds[key]
        results[key] = m
    return results


@torch.no_grad()
def evaluate_aux(model, loader, device, amp_dtype, max_batches: int) -> Dict[str, float]:
    """Sanity signal on the MSP silver-label head. Never selects a checkpoint."""
    model.eval()
    preds, golds, losses = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model.forward_aux_text(input_ids, attention_mask, labels)
        losses.append(float(out["loss"].item()))
        preds.extend(out["logits"].float().argmax(dim=-1).cpu().tolist())
        golds.extend(labels.cpu().tolist())
    m = compute_metrics(golds, preds)
    m["loss"] = float(np.mean(losses)) if losses else float("nan")
    return {k: v for k, v in m.items() if not k.startswith("_")}


def train(cfg: AttrDict) -> None:
    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", False)))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.snapshot.yaml").write_text(
        json.dumps(dict(cfg), indent=2, default=str), encoding="utf-8"
    )

    runlog = RunLogger(
        output_dir=out_dir,
        run_name=str(cfg.run_name),
        use_wandb=bool(cfg.logging.use_wandb),
        wandb_project=str(cfg.logging.wandb_project),
        wandb_config=dict(cfg),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(cfg.train.get("amp_dtype", "none"))

    # ---- data ----
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.model.text.model_id), use_fast=True)
    if tokenizer.pad_token is None:
        # Llama ships no pad token. Padded positions are masked out of the mean
        # pool and of every attention, so reusing EOS is safe — but leaving it
        # unset makes the collate fn crash on `tokenizer.pad_token_id is None`.
        tokenizer.pad_token = tokenizer.eos_token
    loaders = build_conversation_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        care_cache_path=cfg.dataset.care_cache_path,
        tokenizer=tokenizer,
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        max_length=int(cfg.dataset.get("max_length", 128)),
        num_workers=int(cfg.train.num_workers),
    )

    use_aux = bool(cfg.aux.get("enabled", True)) and float(cfg.loss.lambda_aux) > 0.0
    aux_loaders, aux_stream = None, None
    if use_aux:
        aux_loaders = build_aux_loaders(
            manifest_path=cfg.aux.manifest_path,
            label_map=dict(cfg.aux.label_map),
            tokenizer=tokenizer,
            batch_size=int(cfg.aux.batch_size),
            eval_batch_size=int(cfg.aux.eval_batch_size),
            val_fraction=float(cfg.aux.val_fraction),
            seed=int(cfg.aux.get("split_seed", 42)),
            max_length=int(cfg.dataset.get("max_length", 128)),
            num_workers=int(cfg.train.num_workers),
        )
        aux_stream = cycle(aux_loaders["train"])

    # ---- model ----
    # `num_labels` lives at the top level (the dataset owns it, not the model),
    # so it is injected into the model node — and the injected copy is what gets
    # saved with the checkpoint, so a loader can rebuild without the full config.
    model_cfg = AttrDict(dict(cfg.model))
    model_cfg.num_labels = int(cfg.num_labels)
    model = build_model(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    grad_accum = max(1, int(cfg.train.get("grad_accum_steps", 1)))
    steps_per_epoch = math.ceil(len(loaders["train"]) / grad_accum)
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    warmup_steps = int(total_steps * float(cfg.train.warmup_ratio))

    groups = model.param_groups(
        encoder_lr=float(cfg.train.encoder_lr),
        head_lr=float(cfg.train.head_lr),
        audio_lr=float(cfg.train.get("audio_lr", cfg.train.head_lr)),
        weight_decay=float(cfg.train.weight_decay),
    )
    group_names = [g["name"] for g in groups]
    optimizer = AdamW(groups)
    # The frozen 8 B base makes up >99% of `model.parameters()`; iterating it
    # every step just to clip is pure overhead, and only these tensors are ever
    # saved (see `trainable_state_dict`).
    trainable = [p for p in model.parameters() if p.requires_grad]
    text_chunk = int(cfg.train.get("text_chunk", 0))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # bfloat16 needs no loss scaling; the scaler is a no-op unless fp16 is asked for.
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype is torch.float16))
    except (AttributeError, TypeError):  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))

    train_ds = loaders["train"].dataset
    label_names = list(cfg.label_names) if "label_names" in cfg else None
    lambdas = {k: float(cfg.loss[f"lambda_{k}"]) for k in STAGE_KEYS}

    runlog.update_wandb_config({
        "params/trainable": n_params,
        "dataset/train_dialogues": len(train_ds),
        "dataset/train_utterances": train_ds.num_utterances(),
        "dataset/longest_conversation": train_ds.max_conversation_length(),
        "dataset/msp_train": len(aux_loaders["train"].dataset) if use_aux else 0,
        "train/effective_batch_dialogues": int(cfg.train.batch_size) * grad_accum,
        "schedule/steps_per_epoch": steps_per_epoch,
        "schedule/total_steps": total_steps,
        "hardware/gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })
    runlog.info("=" * 74)
    runlog.info(f"Text encoder : {cfg.model.text.model_id}  "
                f"(LoRA r={cfg.model.text.get('lora_r', 16)} "
                f"alpha={cfg.model.text.get('lora_alpha', 32)} "
                f"on {list(cfg.model.text.get('target_modules', ['q_proj', 'v_proj']))}, "
                f"pool={cfg.model.text.get('pool', 'mean')}, "
                f"base={cfg.model.text.get('base_dtype', 'bfloat16')}, "
                f"grad_ckpt={cfg.model.text.get('gradient_checkpointing', True)})")
    runlog.info(f"CARE cache   : {cfg.dataset.care_cache_path}  "
                f"({cfg.model.audio.num_layers} x {cfg.model.audio.layer_dim} "
                f"+ {cfg.model.audio.pooled_dim})")
    runlog.info(f"Dims         : T1={model.text_stage1.hidden_size} "
                f"T2={model.text_stage2.feature_dim} "
                f"S1={model.audio_stage1.hidden_dim} S2={model.audio_stage2.feature_dim}")
    runlog.info(f"Trainable    : {n_params/1e6:.1f}M")
    for split, ld in loaders.items():
        runlog.info(f"  {split:<5}: {len(ld.dataset)} dialogues / "
                    f"{ld.dataset.num_utterances()} utterances "
                    f"(longest conversation = {ld.dataset.max_conversation_length()})")
    runlog.info(f"Lambdas      : " + "  ".join(f"{k}={v}" for k, v in lambdas.items()) +
                f"  aux={cfg.loss.lambda_aux} ({cfg.loss.aux_schedule} -> "
                f"x{cfg.loss.aux_final_scale})")
    runlog.info(f"MSP aux      : {'on, every ' + str(cfg.aux.every) + ' micro-steps' if use_aux else 'OFF'}")
    runlog.info(f"Batch        : {cfg.train.batch_size} dialogues x {grad_accum} accum")
    runlog.info(f"LR           : encoder={cfg.train.encoder_lr}  "
                f"audio={cfg.train.get('audio_lr', cfg.train.head_lr)}  "
                f"head={cfg.train.head_lr}  warmup={warmup_steps}/{total_steps}")
    runlog.info(f"AMP          : {cfg.train.get('amp_dtype', 'none')}   seed={cfg.seed}")
    runlog.info("=" * 74)

    best_score, best_epoch, bad_epochs = -math.inf, -1, 0
    patience = int(cfg.train.early_stopping_patience)
    log_every = int(cfg.logging.log_every)
    aux_every = max(1, int(cfg.aux.get("every", 1))) if use_aux else 0
    best_ckpt = out_dir / "best" / "one_stage.pt"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)
    global_step = 0

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        t0 = time.time()
        running: Dict[str, List[float]] = {k: [] for k in (*STAGE_KEYS, "aux", "total")}
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            care_features = batch["care_features"].to(device, non_blocking=True)
            care_pooled = batch["care_pooled"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model.forward_conversation(
                    input_ids, attention_mask, care_features, care_pooled, mask, labels,
                    text_chunk=text_chunk,
                )
                total = sum(lambdas[k] * out[f"loss_{k}"] for k in STAGE_KEYS)
                for k in STAGE_KEYS:
                    running[k].append(float(out[f"loss_{k}"].item()))

                aux_value = 0.0
                if use_aux and micro_step % aux_every == 0:
                    aux_batch = next(aux_stream)
                    aux_out = model.forward_aux_text(
                        aux_batch["input_ids"].to(device, non_blocking=True),
                        aux_batch["attention_mask"].to(device, non_blocking=True),
                        aux_batch["labels"].to(device, non_blocking=True),
                    )
                    scale = aux_scale(global_step, total_steps,
                                      cfg.loss.aux_schedule, float(cfg.loss.aux_final_scale))
                    total = total + float(cfg.loss.lambda_aux) * scale * aux_out["loss"]
                    aux_value = float(aux_out["loss"].item())
                    running["aux"].append(aux_value)

            running["total"].append(float(total.item()))
            scaler.scale(total / grad_accum).backward()

            is_step = ((micro_step + 1) % grad_accum == 0) or (micro_step + 1 == len(loaders["train"]))
            if not is_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_every == 0:
                scalars = {f"loss_{k}": float(np.mean(running[k][-log_every * grad_accum:]))
                           for k in STAGE_KEYS}
                scalars["loss_total"] = float(np.mean(running["total"][-log_every * grad_accum:]))
                if running["aux"]:
                    scalars["loss_aux"] = float(np.mean(running["aux"][-log_every:]))
                scalars["lambda_aux_scale"] = aux_scale(
                    global_step, total_steps, cfg.loss.aux_schedule,
                    float(cfg.loss.aux_final_scale)) if use_aux else 0.0
                for gname, lr in zip(group_names, scheduler.get_last_lr()):
                    if not gname.endswith("_no_decay"):
                        scalars[f"lr_{gname}"] = lr
                runlog.log_scalars(scalars, step=global_step, prefix="train")
                pbar.set_postfix(total=f"{scalars['loss_total']:.3f}",
                                 fus=f"{scalars['loss_fusion']:.3f}")

        # ---- validation ----
        val = evaluate(model, loaders["val"], device, label_names, amp_dtype, text_chunk)
        for key in STAGE_KEYS:
            runlog.log_scalars(
                {k: v for k, v in val[key].items() if not k.startswith("_")},
                step=global_step, prefix=f"val/{key}",
            )
        if use_aux:
            runlog.log_scalars(
                evaluate_aux(model, aux_loaders["val"], device, amp_dtype,
                             int(cfg.aux.get("max_eval_batches", 40))),
                step=global_step, prefix="val/aux",
            )

        # Which CARE layers the joint objective actually leans on — the staged
        # run had no reason for these to move, so drift here is a real finding.
        lw = model.audio_stage1.get_layer_weights()
        runlog.log_scalars(
            {"argmax": int(lw.argmax()), "max": float(lw.max())},
            step=global_step, prefix="diag/care_layer_weights",
        )

        runlog.info(
            f"epoch {epoch+1:2d}  ({time.time()-t0:.0f}s)  "
            f"fusion wF1={val['fusion']['weighted_f1']:.4f}  "
            f"acc={val['fusion']['accuracy']:.4f}  |  "
            f"T2={val['text_stage2']['weighted_f1']:.4f}  "
            f"A2={val['audio_stage2']['weighted_f1']:.4f}  "
            f"T1={val['text_stage1']['weighted_f1']:.4f}  "
            f"A1={val['audio_stage1']['weighted_f1']:.4f}"
        )

        score = val["fusion"][str(cfg.train.save_best_metric)]
        # Running-best mirror: on a WandB sweep panel the raw val curve is noisy
        # enough that "did this run beat the staged baseline" is hard to read off.
        runlog.log_scalars(
            {f"best_{cfg.train.save_best_metric}": max(score, best_score),
             "best_epoch": best_epoch if score <= best_score else epoch + 1},
            step=global_step, prefix="val/fusion",
        )
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            # Trainable tensors only: LoRA adapters + every head, ~60 M params
            # (~240 MB). A full state_dict would write the frozen 8 B base to
            # disk on every improvement — 16 GB a time, for weights that are
            # already on the Hub. The saved cfg is enough for a loader to
            # rebuild the architecture and re-attach these
            # (feedback-save-checkpoint-weights).
            torch.save({
                "model_state_dict": trainable_state_dict(model),
                "state_dict_is_trainable_only": True,
                "model_cfg": dict(model_cfg),
                "train_cfg": dict(cfg.train),
                "loss_cfg": dict(cfg.loss),
                "label_names": label_names,
                "tokenizer": str(cfg.model.text.model_id),
                "epoch": epoch + 1,
                "seed": int(cfg.seed),
                "score": score,
            }, best_ckpt)
            runlog.info(f"  -> new best (val fusion {cfg.train.save_best_metric}={score:.4f}), saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                runlog.info(f"Early stopping at epoch {epoch+1} "
                            f"(no improvement for {patience} epochs).")
                break

    # ---- test with the best-val weights ----
    if "test" in loaders and best_epoch > 0:
        runlog.info(f"Reloading best checkpoint (epoch {best_epoch}) for test.")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        # strict=False: the checkpoint holds trainable tensors only, and the
        # frozen base is already in memory from `from_pretrained`.
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has tensors the model does not: {unexpected[:5]}")
        test = evaluate(model, loaders["test"], device, label_names, amp_dtype, text_chunk)
        for key in STAGE_KEYS:
            runlog.log_scalars(
                {k: v for k, v in test[key].items() if not k.startswith("_")},
                step=global_step, prefix=f"test/{key}",
            )
        runlog.info("-" * 74)
        for key in STAGE_KEYS:
            runlog.info(f"TEST {key:<13} acc={test[key]['accuracy']:.4f}  "
                        f"wF1={test[key]['weighted_f1']:.4f}  "
                        f"macroF1={test[key]['macro_f1']:.4f}")
        runlog.info("-" * 74)
        report = detailed_report(test["fusion"]["_labels"], test["fusion"]["_preds"], label_names)
        (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
        runlog.info("\n" + report)
        (out_dir / "result.json").write_text(json.dumps({
            "seed": int(cfg.seed),
            "best_epoch": best_epoch,
            "val_fusion_weighted_f1": best_score,
            "test": {k: {m: v for m, v in test[k].items() if not m.startswith("_")}
                     for k in STAGE_KEYS},
        }, indent=2), encoding="utf-8")

    runlog.info(f"DONE. best val fusion {cfg.train.save_best_metric} = {best_score:.4f} "
                f"(epoch {best_epoch}).")
    runlog.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--override", nargs="*", default=[],
                        help="dotted.key=json_value, e.g. seed=3 loss.lambda_aux=0.0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        key, _, val = ov.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        node[keys[-1]] = parsed

    train(cfg)


if __name__ == "__main__":
    main()
