"""Shape / gradient-reachability / memory check before committing to a long run.

The central claim of this project is that the fusion objective can reach back
into the Stage I encoders — something the staged cascade structurally cannot do.
That is a property of the wiring, so it is checked directly: backward on
`loss_fusion` ALONE and assert that the LoRA adapters and the CARE layer weights
receive non-zero gradients.

With an 8B text encoder there is a second thing to watch that the RoBERTa
version did not need: whether the fusion gradient reaching the audio branch is
drowned by the text branch. The test reports both magnitudes side by side.

Runs on synthetic tensors by default (no dataset needed):

    python -m scripts.smoke_test

Against the real loaders (one batch of each stream):

    python -m scripts.smoke_test --real
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.one_stage_model import build_model          # noqa: E402
from src.utils.config import AttrDict, load_config          # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def grad_norm(model, prefix: str) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return total ** 0.5


def synthetic_batch(cfg, B: int, K: int, L: int, device, vocab_size: int):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (B, K, L), generator=g)
    attention_mask = torch.ones(B, K, L, dtype=torch.long)
    care_features = torch.randn(B, K, int(cfg.audio.num_layers), int(cfg.audio.layer_dim),
                                generator=g)
    care_pooled = torch.randn(B, K, int(cfg.audio.pooled_dim), generator=g)
    labels = torch.randint(0, int(cfg.num_labels), (B, K), generator=g)
    mask = torch.ones(B, K, dtype=torch.bool)
    if K > 2:  # exercise the padding path
        mask[-1, K // 2:] = False
        labels[-1, K // 2:] = -100
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "care_features": care_features.to(device),
        "care_pooled": care_pooled.to(device),
        "labels": labels.to(device),
        "mask": mask.to(device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/one_stage_iemocap_llama.yaml")
    ap.add_argument("--model-id", default=None,
                    help="override model.text.model_id (a tiny Llama speeds this up)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--real", action="store_true", help="use the real loaders instead of noise")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--utts", type=int, default=8, help="utterances per conversation (synthetic)")
    ap.add_argument("--seq-len", type=int, default=32, help="tokens per utterance (synthetic)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model_id:
        cfg.model.text.model_id = args.model_id
    device = torch.device(args.device)

    model_cfg = AttrDict(dict(cfg.model))
    model_cfg.num_labels = int(cfg.num_labels)
    text_chunk = int(cfg.train.get("text_chunk", 0))

    print(f"device={device}  text_encoder={model_cfg.text.model_id}  text_chunk={text_chunk}")
    print("building model ...")
    model = build_model(model_cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: {n_train/1e6:.1f}M trainable / {n_all/1e9:.2f}B total   "
          f"T1={model.text_stage1.hidden_size} T2={model.text_stage2.feature_dim} "
          f"S1={model.audio_stage1.hidden_dim} S2={model.audio_stage2.feature_dim}")

    # ---- batch ----
    if args.real:
        from transformers import AutoTokenizer
        from src.data.joint_dataset import build_aux_loaders, build_conversation_loaders
        tok = AutoTokenizer.from_pretrained(str(model_cfg.text.model_id), use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        loaders = build_conversation_loaders(
            manifest_dir=cfg.dataset.manifest_dir,
            care_cache_path=cfg.dataset.care_cache_path,
            tokenizer=tok,
            batch_size=args.batch_size, eval_batch_size=args.batch_size,
            max_length=int(cfg.dataset.get("max_length", 128)),
            num_workers=0, splits=("train",),
        )
        raw = next(iter(loaders["train"]))
        batch = {k: v.to(device) for k, v in raw.items() if torch.is_tensor(v)}
        aux_loaders = build_aux_loaders(
            manifest_path=cfg.aux.manifest_path, label_map=dict(cfg.aux.label_map),
            tokenizer=tok, batch_size=int(cfg.aux.batch_size),
            eval_batch_size=int(cfg.aux.batch_size),
            val_fraction=float(cfg.aux.val_fraction), seed=int(cfg.aux.get("split_seed", 42)),
            max_length=int(cfg.dataset.get("max_length", 128)), num_workers=0,
        )
        aux_batch = {k: v.to(device) for k, v in next(iter(aux_loaders["train"])).items()}
    else:
        from transformers import AutoConfig
        vocab = AutoConfig.from_pretrained(str(model_cfg.text.model_id)).vocab_size
        batch = synthetic_batch(model_cfg, args.batch_size, args.utts, args.seq_len,
                                device, vocab)
        n_aux = int(cfg.aux.batch_size)
        aux_batch = {
            "input_ids": torch.randint(0, vocab, (n_aux, args.seq_len)).to(device),
            "attention_mask": torch.ones(n_aux, args.seq_len, dtype=torch.long).to(device),
            "labels": torch.randint(0, int(cfg.model.text.num_labels_aux),
                                    (n_aux,)).to(device),
        }

    B, K, L = batch["input_ids"].shape
    n_real = int(batch["mask"].sum())
    print(f"\nbatch: B={B} conversations, K={K} max utterances, L={L} tokens, "
          f"{n_real} real utterances\n")

    fwd = dict(batch, text_chunk=text_chunk)

    # ---- 1. forward: shapes + finite losses ----
    print("1. forward")
    model.train()
    out = model.forward_conversation(**fwd)
    check("fusion logits shape", tuple(out["logits_fusion"].shape) == (B, K, model.num_labels),
          str(tuple(out["logits_fusion"].shape)))
    check("stage1 logits are per real utterance",
          out["logits_text_stage1"].shape[0] == n_real,
          f"{out['logits_text_stage1'].shape[0]} == {n_real}")
    check("T1 is float32 (Bi-GRU must not see half precision)",
          out["logits_text_stage1"].dtype == torch.float32,
          str(out["logits_text_stage1"].dtype))
    for key in ("fusion", "text_stage2", "audio_stage2", "text_stage1", "audio_stage1"):
        v = out[f"loss_{key}"].detach()
        check(f"loss_{key} finite", bool(torch.isfinite(v)), f"{float(v):.4f}")

    # ---- 2. the claim: fusion gradient reaches Stage I ----
    print("\n2. gradient reachability (backward on loss_fusion ONLY)")
    model.zero_grad(set_to_none=True)
    out = model.forward_conversation(**fwd)
    out["loss_fusion"].backward()

    lora_norm = grad_norm(model, "text_stage1.base.")
    layerw = model.audio_stage1.weights.grad
    care_fc_norm = grad_norm(model, "audio_stage1.fc")

    check("LoRA adapters receive fusion gradient", lora_norm > 0, f"||g||={lora_norm:.3e}")
    check("CARE layer weights receive fusion gradient",
          layerw is not None and float(layerw.abs().sum()) > 0,
          "n/a" if layerw is None else f"|g|={float(layerw.abs().sum()):.3e}")
    check("CARE Stage I fc receives fusion gradient", care_fc_norm > 0, f"||g||={care_fc_norm:.3e}")
    # Routing sanity: the Stage I *heads* are off the fusion path entirely.
    check("Stage I heads get NO fusion gradient (routing is correct)",
          model.text_stage1.out_proj.weight.grad is None,
          "out_proj.grad is None")

    # Modality balance. An 8B text branch can starve the audio branch inside
    # co-attention; this is the number to watch if audio Stage I collapses.
    t_proj = grad_norm(model, "fusion.proj_text")
    a_proj = grad_norm(model, "fusion.proj_audio")
    ratio = t_proj / a_proj if a_proj > 0 else float("inf")
    print(f"  [info] fusion projection gradient  text={t_proj:.3e}  audio={a_proj:.3e}  "
          f"ratio={ratio:.2f}x")

    # ---- 3. full objective touches everything ----
    print("\n3. full weighted objective")
    model.zero_grad(set_to_none=True)
    out = model.forward_conversation(**fwd)
    total = sum(
        float(cfg.loss[f"lambda_{k}"]) * out[f"loss_{k}"]
        for k in ("fusion", "text_stage2", "audio_stage2", "text_stage1", "audio_stage1")
    )
    aux_out = model.forward_aux_text(**aux_batch)
    total = total + float(cfg.loss.lambda_aux) * aux_out["loss"]
    total.backward()

    aux_loss = aux_out["loss"].detach()
    check("aux (MSP) loss finite", bool(torch.isfinite(aux_loss)), f"{float(aux_loss):.4f}")
    check("aux head receives gradient",
          model.text_stage1.out_proj_aux.weight.grad is not None
          and float(model.text_stage1.out_proj_aux.weight.grad.abs().sum()) > 0)
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    check("every trainable parameter got a gradient", not missing,
          "ok" if not missing else f"{len(missing)} missing, e.g. {missing[:3]}")
    check("the 8B base stays frozen",
          all(not p.requires_grad for n, p in model.named_parameters()
              if "base_model" in n and "lora" not in n.lower()))

    # ---- 4. optimizer groups + memory ----
    print("\n4. optimizer + memory")
    groups = model.param_groups(encoder_lr=5e-5, head_lr=5e-5, audio_lr=1e-4,
                                weight_decay=0.01)
    families = {g["name"].replace("_no_decay", "") for g in groups}
    check("encoder / audio / head param groups exist",
          families == {"encoder", "audio", "head"}, str(sorted(families)))
    n_grouped = sum(len(g["params"]) for g in groups)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    check("all trainable tensors are in some group", n_grouped == n_total,
          f"{n_grouped}/{n_total}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  peak GPU memory: {peak:.2f} GiB "
              f"(B={B}, K={K}, L={L}, text_chunk={text_chunk})")

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print("\n" + "=" * 60)
    print(f"SMOKE TEST {'PASSED' if n_fail == 0 else 'FAILED'} "
          f"({len(CHECKS) - n_fail}/{len(CHECKS)} checks)")
    print("=" * 60)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
