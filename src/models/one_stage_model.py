"""One-stage MERITS-L (Llama) — the whole cascade as a single differentiable module.

Same experiment as `one-stage-merits-l`, with the RoBERTa-large text encoder
replaced by Llama-3.1-8B + LoRA. The staged pipeline this collapses is the one
in `merits-l-llama`, whose Stage III fusion is the best IEMOCAP result we have
(0.8567 ± 0.0139, best 0.8746).

    Llama-3.1-8B + LoRA  ->  T1 (4096)  ->  Bi-GRU 1024 x2  ->  T2 (2048)  |
                                                                           +- co-attn -> 4
    CARE-WavLM (frozen)  ->  S1 (256)   ->  Bi-GRU 128 x2   ->  S2 (256)   |
       cached (13, 1536) + (768,)

Six weighted cross-entropies replace six sequential trainings; see README.

Differences from the RoBERTa version, all forced by the encoder:
  * decoder-only, so there is no CLS token — utterance embeddings are the mean
    over non-pad tokens, matching `merits-l-llama`'s LlamaClassifier.
  * no `dense` projection between encoder and heads. RoBERTa's came from
    `RobertaClassificationHead` and was the parameter the MSP and IEMOCAP heads
    shared; here the shared parameters are the LoRA adapters themselves, which
    is both more faithful to `merits-l-llama` and 16.8 M parameters cheaper.
    T1 is the pooled hidden state directly — exactly what
    `iemocap_llama_features.pt` held in the staged run.
  * the pooled vector is cast to float32 before anything downstream: the base
    runs in bf16, and feeding half precision into the Stage II Bi-GRU was a
    documented failure in the merits-l inference release.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
#  Stage I — text (Llama-3.1-8B + LoRA)
# ---------------------------------------------------------------------------
class LlamaTextStage1(nn.Module):
    """Llama base (frozen) + LoRA adapters + one output head per label space.

    Two label spaces coexist in a one-stage run:
      * MSP-PODCAST silver labels — 3 classes (negative / neutral / positive)
      * IEMOCAP                   — 4 classes (angry / happy / sad / neutral)

    They share the encoder and its LoRA adapters and differ only in `out_proj`.
    That sharing is the entire mechanism of the auxiliary term: the MSP gradient
    has to reach the adapters that shape T1, not stop at a private head.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B",
        num_labels: int = 4,
        num_labels_aux: int = 3,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        pool: str = "mean",
        dropout: float = 0.1,
        base_dtype: str = "bfloat16",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "last"):
            raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")
        self.pool = pool

        dtype = _DTYPES[str(base_dtype)]
        self.base = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        hidden_size = self.base.config.hidden_size          # 4096 for Llama-3.1-8B
        # We never generate, and a KV cache is incompatible with gradient
        # checkpointing — transformers would disable it anyway, once per forward,
        # with a warning.
        self.base.config.use_cache = False

        if gradient_checkpointing:
            # A dialogue batch is B x K sequences (K up to 110 for IEMOCAP), so
            # the encoder sees tens to hundreds of short sequences per step.
            # use_reentrant=False is required — the reentrant variant needs an
            # input that requires grad, and token ids never do.
            self.base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.base, "enable_input_require_grads"):
                # With a frozen base + LoRA, checkpointed blocks otherwise see no
                # input requiring grad and silently drop the adapter gradients.
                self.base.enable_input_require_grads()

        if use_lora:
            from peft import LoraConfig, get_peft_model
            self.base = get_peft_model(self.base, LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(target_modules or ["q_proj", "v_proj"]),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            ))
            # Optimise the adapters in fp32. Half-precision LoRA weights under
            # six summed losses is exactly where silent NaNs come from.
            for _, p in self.base.named_parameters():
                if p.requires_grad:
                    p.data = p.data.float()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)          # IEMOCAP
        self.out_proj_aux = nn.Linear(hidden_size, num_labels_aux)  # MSP silver

        self.num_labels = num_labels
        self.num_labels_aux = num_labels_aux
        self.use_lora = use_lora

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        lengths = attention_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

    def get_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(N, L) -> (N, 4096) float32. This is T1_k, the input to text Stage II."""
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        # Everything downstream (Bi-GRU, co-attention, losses) runs in fp32.
        return self.dropout(pooled.float())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        head: str = "main",
    ) -> Dict[str, Optional[torch.Tensor]]:
        features = self.get_features(input_ids, attention_mask)
        logits = self.out_proj(features) if head == "main" else self.out_proj_aux(features)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "features": features}

    def save_adapter(self, path: str) -> None:
        """Write the LoRA adapter (~30 MB) rather than the 16 GB base."""
        self.base.save_pretrained(path)


# ---------------------------------------------------------------------------
#  Stage I — audio
# ---------------------------------------------------------------------------
class CAREStage1(nn.Module):
    """CARE's official downstream head, ported from `train_care_downstream_iemocap.py`.

        feat (B, 13, 1536) --softmax(w) over layers--> (B, 1536)
        cat with pooled (B, 768)                    -> (B, 2304)
        fc   2304 -> 768   (dropout before)
        fc_1  768 -> 256   == S1
        out   256 -> 4     (dropout before)

    Dimensions are read from the cached feature file, so a CARE-large cache
    (25 x 2048 + 1024) drops in by changing the config alone.
    """

    def __init__(
        self,
        num_layers: int = 13,
        layer_dim: int = 1536,
        pooled_dim: int = 768,
        proj_dim: int = 768,
        hidden_dim: int = 256,
        num_labels: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.rand(num_layers, 1))
        self.fc = nn.Linear(layer_dim + pooled_dim, proj_dim)
        self.fc_1 = nn.Linear(proj_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def get_features(self, feat: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        if feat.size(1) != self.num_layers:
            raise ValueError(
                f"cached features have {feat.size(1)} layers but the head was built "
                f"for {self.num_layers}; set model.audio.num_layers to match."
            )
        w = F.softmax(self.weights, dim=0).squeeze(-1)
        weighted = (feat * w[None, :, None]).sum(dim=1)
        fused = torch.cat([weighted, pooled], dim=-1)
        return self.fc_1(self.dropout(self.fc(fused)))

    def forward(self, feat, pooled, labels=None) -> Dict[str, Optional[torch.Tensor]]:
        features = self.get_features(feat, pooled)
        logits = self.out(self.dropout(features))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "features": features}

    @torch.no_grad()
    def get_layer_weights(self) -> torch.Tensor:
        return F.softmax(self.weights, dim=0).squeeze(-1).detach().cpu()


# ---------------------------------------------------------------------------
#  Stage II — conversation context (shared architecture for both modalities)
# ---------------------------------------------------------------------------
class DialogueStage2(nn.Module):
    """Bi-GRU + multi-head self-attention over the utterances of a conversation."""

    def __init__(
        self,
        input_dim: int,
        gru_hidden: int,
        gru_layers: int = 2,
        num_heads: int = 8,
        num_labels: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feature_dim = 2 * gru_hidden
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(feature_dim)
        self.ln_attn = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_labels)
        self.feature_dim = feature_dim

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.bigru(features)
        gru_out = self.ln_gru(gru_out)
        attn_out, _ = self.attn(
            gru_out, gru_out, gru_out, key_padding_mask=~mask, need_weights=False,
        )
        return self.ln_attn(gru_out + self.dropout(attn_out))

    def forward(self, features, mask, labels=None) -> Dict[str, Optional[torch.Tensor]]:
        x = self.encode(features, mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "features": x}


# ---------------------------------------------------------------------------
#  Stage III — co-attention fusion
# ---------------------------------------------------------------------------
class Stage3Fusion(nn.Module):
    """Co-attention fusion (paper Fig. 2), ported unchanged from merits-l-llama."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_labels: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.proj_text = nn.Linear(text_dim, hidden_dim)
        self.proj_audio = nn.Linear(audio_dim, hidden_dim)
        self.text_attends_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.audio_attends_text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.fuse_fc = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, text_features, audio_features, mask, labels=None):
        t = self.proj_text(text_features)
        a = self.proj_audio(audio_features)
        key_padding_mask = ~mask
        t_attn, _ = self.text_attends_audio(
            t, a, a, key_padding_mask=key_padding_mask, need_weights=False)
        a_attn, _ = self.audio_attends_text(
            a, t, t, key_padding_mask=key_padding_mask, need_weights=False)
        t_out = self.ln_text(t + self.dropout(t_attn))
        a_out = self.ln_audio(a + self.dropout(a_attn))
        fused = torch.cat([t_out, a_out, t, a], dim=-1)
        h = self.dropout(F.relu(self.fuse_fc(fused)))
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "fused_hidden": h}


# ---------------------------------------------------------------------------
#  The whole thing
# ---------------------------------------------------------------------------
class OneStageMERITSLlama(nn.Module):
    """All five trainable blocks under one optimizer.

    Two entry points, because the two data streams have different shapes:
      * `forward_conversation` — a batch of IEMOCAP conversations, returns all
        five IEMOCAP losses at once.
      * `forward_aux_text`     — a flat batch of MSP-PODCAST transcripts.

    The trainer weights and sums them; nothing here knows about lambdas.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        num_labels = int(cfg.num_labels)

        self.text_stage1 = LlamaTextStage1(
            model_id=str(cfg.text.model_id),
            num_labels=num_labels,
            num_labels_aux=int(cfg.text.num_labels_aux),
            use_lora=bool(cfg.text.get("use_lora", True)),
            lora_r=int(cfg.text.get("lora_r", 16)),
            lora_alpha=int(cfg.text.get("lora_alpha", 32)),
            lora_dropout=float(cfg.text.get("lora_dropout", 0.1)),
            target_modules=cfg.text.get("target_modules", ["q_proj", "v_proj"]),
            pool=str(cfg.text.get("pool", "mean")),
            dropout=float(cfg.text.dropout),
            base_dtype=str(cfg.text.get("base_dtype", "bfloat16")),
            gradient_checkpointing=bool(cfg.text.get("gradient_checkpointing", True)),
        )
        self.audio_stage1 = CAREStage1(
            num_layers=int(cfg.audio.num_layers),
            layer_dim=int(cfg.audio.layer_dim),
            pooled_dim=int(cfg.audio.pooled_dim),
            proj_dim=int(cfg.audio.get("proj_dim", 768)),
            hidden_dim=int(cfg.audio.hidden_dim),
            num_labels=num_labels,
            dropout=float(cfg.audio.dropout),
        )
        self.text_stage2 = DialogueStage2(
            input_dim=self.text_stage1.hidden_size,
            gru_hidden=int(cfg.text_stage2.gru_hidden),
            gru_layers=int(cfg.text_stage2.get("gru_layers", 2)),
            num_heads=int(cfg.text_stage2.num_heads),
            num_labels=num_labels,
            dropout=float(cfg.text_stage2.dropout),
        )
        self.audio_stage2 = DialogueStage2(
            input_dim=self.audio_stage1.hidden_dim,
            gru_hidden=int(cfg.audio_stage2.gru_hidden),
            gru_layers=int(cfg.audio_stage2.get("gru_layers", 2)),
            num_heads=int(cfg.audio_stage2.num_heads),
            num_labels=num_labels,
            dropout=float(cfg.audio_stage2.dropout),
        )
        self.fusion = Stage3Fusion(
            text_dim=self.text_stage2.feature_dim,
            audio_dim=self.audio_stage2.feature_dim,
            hidden_dim=int(cfg.fusion.hidden_dim),
            num_heads=int(cfg.fusion.num_heads),
            num_labels=num_labels,
            dropout=float(cfg.fusion.dropout),
        )
        self.num_labels = num_labels

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _scatter(flat: torch.Tensor, index: torch.Tensor, B: int, K: int) -> torch.Tensor:
        """Place per-utterance vectors back into their (B, K, D) slots.

        Padded slots stay exactly zero; they are masked out of every attention
        and every loss downstream.
        """
        out = flat.new_zeros(B * K, flat.size(-1))
        out[index] = flat
        return out.view(B, K, flat.size(-1))

    # -- IEMOCAP -----------------------------------------------------------
    def forward_conversation(
        self,
        input_ids: torch.Tensor,       # (B, K, L)
        attention_mask: torch.Tensor,  # (B, K, L)
        care_features: torch.Tensor,   # (B, K, num_layers, layer_dim)
        care_pooled: torch.Tensor,     # (B, K, pooled_dim)
        mask: torch.Tensor,            # (B, K) bool — True = real utterance
        labels: Optional[torch.Tensor] = None,   # (B, K) long, -100 on pads
        text_chunk: int = 0,
    ) -> Dict[str, torch.Tensor]:
        B, K, L = input_ids.shape
        flat_mask = mask.reshape(-1)
        index = flat_mask.nonzero(as_tuple=True)[0]

        # Only real utterances go through the encoder — a padded conversation
        # slot would otherwise cost a full Llama forward for nothing.
        ids = input_ids.reshape(B * K, L)[index]
        att = attention_mask.reshape(B * K, L)[index]
        flat_labels = labels.reshape(-1)[index] if labels is not None else None

        if text_chunk and ids.size(0) > text_chunk:
            # IEMOCAP's longest conversation is 110 utterances, which can exceed
            # what fits in one Llama pass even with checkpointing. Chunking only
            # splits the forward — the autograd graph and therefore the
            # gradients are identical.
            t1_flat = torch.cat([
                self.text_stage1.get_features(ids[i:i + text_chunk], att[i:i + text_chunk])
                for i in range(0, ids.size(0), text_chunk)
            ], dim=0)
        else:
            t1_flat = self.text_stage1.get_features(ids, att)              # (N, 4096)

        s1_flat = self.audio_stage1.get_features(
            care_features.reshape(B * K, *care_features.shape[2:])[index],
            care_pooled.reshape(B * K, care_pooled.size(-1))[index],
        )                                                                   # (N, 256)

        logits_t1 = self.text_stage1.out_proj(t1_flat)
        logits_s1 = self.audio_stage1.out(self.audio_stage1.dropout(s1_flat))

        out: Dict[str, torch.Tensor] = {
            "index": index,
            "logits_text_stage1": logits_t1,
            "logits_audio_stage1": logits_s1,
        }
        if flat_labels is not None:
            out["loss_text_stage1"] = F.cross_entropy(logits_t1, flat_labels)
            out["loss_audio_stage1"] = F.cross_entropy(logits_s1, flat_labels)

        t1 = self._scatter(t1_flat, index, B, K)
        s1 = self._scatter(s1_flat, index, B, K)

        text2 = self.text_stage2(t1, mask, labels)
        audio2 = self.audio_stage2(s1, mask, labels)
        fused = self.fusion(text2["features"], audio2["features"], mask, labels)

        if labels is not None:
            out["loss_text_stage2"] = text2["loss"]
            out["loss_audio_stage2"] = audio2["loss"]
            out["loss_fusion"] = fused["loss"]

        out["logits_fusion"] = fused["logits"]
        out["logits_text_stage2"] = text2["logits"]
        out["logits_audio_stage2"] = audio2["logits"]
        return out

    # -- MSP-PODCAST -------------------------------------------------------
    def forward_aux_text(self, input_ids, attention_mask, labels=None):
        """LLM-supervised term: 3-class silver labels on MSP-PODCAST transcripts."""
        return self.text_stage1(input_ids, attention_mask, labels, head="aux")

    # -- introspection ------------------------------------------------------
    @torch.no_grad()
    def modality_balance(self) -> float:
        """||grad(proj_text)|| / ||grad(proj_audio)|| at the fusion input.

        Read as a trend, not a level: proj_text is 2048->256 and proj_audio is
        256->256, so the ratio starts above 1 for purely mechanical reasons
        (~2.6 at initialisation with Llama). What matters is whether it climbs
        during training, which is what co-attention abandoning the audio branch
        looks like from the inside. Call it while gradients are still populated,
        i.e. after clipping and before `optimizer.zero_grad()`.
        """
        def _n(module) -> float:
            return sum(float(p.grad.detach().float().pow(2).sum())
                       for p in module.parameters() if p.grad is not None) ** 0.5
        a = _n(self.fusion.proj_audio)
        return _n(self.fusion.proj_text) / a if a > 0 else float("nan")

    def param_groups(self, encoder_lr: float, head_lr: float, weight_decay: float,
                     audio_lr: Optional[float] = None):
        """Three learning rates, because the branches have opposite needs.

        `encoder` here means the LoRA adapters (the base is frozen), which
        merits-l-llama trained at 5e-5. `audio` is the CARE Stage I head only —
        CARE's own downstream recipe is 1e-4 for that small FC, while every
        Stage II / III block in the staged pipeline trained at 5e-5. Running
        audio Stage II at 1e-4 alongside Stage I made its test wF1 swing
        0.345-0.790 across seeds in the RoBERTa version of this experiment.
        """
        audio_lr = head_lr if audio_lr is None else audio_lr
        no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight",
                    # SUPERB layer-mixing logits are a distribution, not a
                    # weight matrix; decaying them pulls the mix toward uniform.
                    "audio_stage1.weights")
        specs = [("encoder", encoder_lr), ("audio", audio_lr), ("head", head_lr)]
        groups = []
        for family, lr in specs:
            groups.append({"name": family, "params": [], "lr": lr,
                           "weight_decay": weight_decay})
            groups.append({"name": f"{family}_no_decay", "params": [], "lr": lr,
                           "weight_decay": 0.0})

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("text_stage1.base."):
                family = 0
            elif name.startswith("audio_stage1."):
                family = 1
            else:
                family = 2
            groups[2 * family + (1 if any(nd in name for nd in no_decay) else 0)]["params"].append(p)
        return [g for g in groups if g["params"]]


def build_model(cfg) -> OneStageMERITSLlama:
    return OneStageMERITSLlama(cfg)
