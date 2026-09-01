# One-stage MERITS-L (Llama)

Collapsing the six sequential training runs of
[`merits-l-llama`](https://github.com/ouoouoouoouo/merits-l-llama) into a
**single joint optimization** with deep supervision, keeping the LLM-supervised
MSP-PODCAST objective as a live auxiliary loss rather than a pre-training phase.

Sibling repo: [`one-stage-merits-l`](https://github.com/ouoouoouoouo/one-stage-merits-l-)
— the same experiment with RoBERTa-large, already complete.

## What is being collapsed

`merits-l-llama` follows MERITS-L (Dutta & Ganapathy, ICASSP 2025) and trains
each block to convergence, then freezes it:

| # | Training run | Data | Objective |
|---|---|---|---|
| 0 | RoBERTa/Llama pre-training | MSP-PODCAST transcripts | CE, 3-class GPT-3.5 silver labels |
| 1 | Text Stage I | IEMOCAP utterances | CE, 4-class |
| 2 | Audio Stage I | frozen CARE embeddings | CE, 4-class |
| 3 | Text Stage II | conversations | CE, 4-class |
| 4 | Audio Stage II | conversations | CE, 4-class |
| 5 | Stage III fusion | conversations | CE, 4-class |

The cascade has a structural weakness: **Stage I's utterance representations are
frozen before the fusion objective is ever computed**. If Llama encodes an
utterance in a way that is unhelpful for cross-modal fusion, nothing downstream
can repair it — Stage II and III can only re-weight what they are handed. And
fusion is precisely where this model's accuracy comes from.

This repo holds all five trainable blocks in one module and drives them with a
weighted sum of all six cross-entropies:

$$\mathcal{L} = \lambda_{f}\mathcal{L}_{fusion} + \lambda_{t2}\mathcal{L}^{II}_{text} + \lambda_{a2}\mathcal{L}^{II}_{audio} + \lambda_{t1}\mathcal{L}^{I}_{text} + \lambda_{a1}\mathcal{L}^{I}_{audio} + \lambda_{aux}(t)\,\mathcal{L}_{MSP}$$

## Architecture

```
MSP transcripts ──┐                                    ┌─► FC(3) ──► L_MSP
                  ├─► Llama-3.1-8B + LoRA ─► T1 (4096) ┤
IEMOCAP text ─────┘   base FROZEN, q/v adapters train  └─► FC(4) ──► L_text_I
                                        │
                                        └─► Bi-GRU 1024×2 + self-attn ──► T2 (2048) ──┐
                                                        │                              │
                                                        └──► FC(4) ──► L_text_II       │
                                                                                       ├─► co-attn ──► L_fusion
                                        ┌─► Bi-GRU 128×2 + self-attn ──► S2 (256) ────┘
                                        │               │
IEMOCAP wav ─► CARE (FROZEN, cached) ───┴─► S1 (256)    └──► FC(4) ──► L_audio_II
   (13, 1536) + (768,)                      │
                                            └─► FC(4) ──► L_audio_I
```

Every dimension matches `merits-l-llama`, so a joint-vs-staged comparison
isolates the training schedule and nothing else.

* **Text**: `meta-llama/Meta-Llama-3.1-8B`, base frozen, LoRA r=16 α=32 on
  `q_proj` / `v_proj`, mean-pool over non-pad tokens → T¹ = 4096-d. This is
  exactly what `iemocap_llama_features.pt` held in the staged run.
* **Audio**: CARE-WavLM is **not loaded at all** — its IEMOCAP embeddings are
  cached to disk, as in the paper where CARE is frozen. No MSP-PODCAST *audio*
  is touched at training time.
* ~60 M trainable parameters (LoRA adapters + every head) out of ~8.1 B.

### Three things the encoder swap forces

1. **No CLS token.** Llama is decoder-only, so utterance embeddings are the mean
   over non-pad tokens (`pool: mean`; `last` is also available). The RoBERTa
   version pooled `last_hidden_state[:, 0]`.
2. **No `dense` projection.** RoBERTa's came from `RobertaClassificationHead`
   and was the parameter the MSP and IEMOCAP heads shared. Here the shared
   parameters are the LoRA adapters themselves — more faithful to
   `merits-l-llama`, and 16.8 M parameters cheaper.
3. **fp32 at the interface.** The base runs in bf16; the pooled vector is cast
   to float32 before the Stage II Bi-GRU. Feeding half precision into that GRU
   was a documented failure in the merits-l inference release.

## What happened to MSP-PODCAST

MSP-PODCAST plays two unrelated roles in the paper, and only one survives here:

* **CARE's self-supervised pre-training** (semantic distillation + PASE+ on
  230 h of audio) — already done, `care-training/ckpts_faithful/best.pth`.
* **LLM-supervised text pre-training** (Whisper-large-v3 transcripts labelled
  positive/negative/neutral by GPT-3.5 Turbo) — kept, but **folded into the
  joint objective as `L_MSP`** instead of run as a phase. 118,505 transcripts,
  text only, no waveforms.

The 3-class silver labels and the 4-class emotion labels never meet: two output
heads sit on the shared encoder, and nothing maps between them. Transfer happens
through the LoRA adapters, which both heads' gradients reach.

Because `L_MSP` replaces a *pre-training phase*, its weight decays over training
(`loss.aux_schedule: cosine`, 1.0 → 0.05) — the continuous-time image of
"pre-train for 10 epochs, then stop". A constant weight would leave a 3-class
sentiment task on 118 K samples competing with a 4-class emotion task on 3,205
all the way to the end.

Nothing in this repo is self-supervised: `L_MSP` is supervised by LLM-generated
silver labels, and CARE's self-supervision happened in a previous project.

## Expectations

Two independent pieces of evidence say to expect **parity, not a new best**:

| Evidence | Result |
|---|---|
| One-stage vs staged, RoBERTa (sibling repo) | 0.8281 ± 0.0064 vs 0.8305 ± 0.0138 — indistinguishable |
| Staged Llama, with vs without MSP pre-training | 0.8550 ± 0.0072 vs 0.8567 ± 0.0139 — no benefit |

The mechanism that could still deliver a gain: in the staged pipeline Llama's
utterance embeddings were shaped by an utterance-level CE and then frozen, never
by the fusion objective. Llama's representations are far richer than
RoBERTa's, so there is more for co-attention to reshape. That is a hypothesis,
not a prediction.

`lambda_aux` is a config flag, so both arms cost the same to build. Run them as
two arms rather than deciding in advance.

## Baselines

| System | training runs | wF1 (best) | wF1 (mean ± std) |
|---|---:|---:|---:|
| Paper Table I (RoBERTa-base + CARE, staged) | 6 | 0.8648 | — (single seed) |
| `merits-l-text` staged reproduction | 6 | 0.8504 | 0.8305 ± 0.0138 |
| `one-stage-merits-l` (RoBERTa) | 1 | 0.8368 | 0.8281 ± 0.0064 |
| **`merits-l-llama` staged** ⭐ | 6 | **0.8746** | **0.8567 ± 0.0139** |
| `merits-l-llama` staged + MSP pre-train | 6 | 0.8640 | 0.8550 ± 0.0072 |
| **One-stage Llama (this repo)** | **1** | — | — |

## Setup (cluster)

```bash
cd /home/ouo/one-stage-merits-l-llama
ln -s /home/ouo/merits-l-text/data data      # manifests + CARE cache + MSP labels
pip install -r requirements.txt
huggingface-cli login                        # Llama-3.1 is a gated repo
```

The symlinked `data/` must contain:

| Path | Size | Source |
|---|---|---|
| `manifests/iemocap/{train,val,test}.csv` | 612 K | `merits-l-text/scripts/preprocess_iemocap.py` |
| `cache/iemocap_care_downstream.pt` | 441 M | `care-training/scripts/extract_iemocap_care_downstream_style.py` |
| `manifests/msp_podcast/pseudo_labels.csv` | 18 M | `merits-l-text/scripts/llm_pseudo_label_msp.py` |

IEMOCAP audio is **not** needed — the CARE cache already holds every
utterance's embeddings.

## Run

```bash
# wiring check first — asserts the fusion gradient reaches the LoRA adapters
python -m scripts.smoke_test --real

# single seed
python -m src.train_joint --config configs/one_stage_iemocap_llama.yaml

# the two arms, 5 seeds each
bash scripts/run_multiseed.sh msp
bash scripts/run_multiseed.sh nomsp loss.lambda_aux=0.0

python -m scripts.summarize_seeds outputs/msp outputs/nomsp --latex
```

The smoke test also prints the **modality balance** — the ratio of the fusion
gradient reaching `proj_text` versus `proj_audio`. With an 8 B text encoder the
co-attention can learn to lean on text and starve the audio branch; that ratio
is the number to watch if audio Stage I collapses.

## Protocol

Paper split (Sec. IV-A): sessions 2–4 train, session 1 val, session 5 test;
5,531 utterances, 4-way (excited merged into happy). Model selection is on the
**fusion head's val weighted-F1**; the other five losses are auxiliary and never
select a checkpoint. Test is touched once, at the end, with the best-val weights.

Checkpoints hold **trainable tensors only** (LoRA adapters + heads, ~240 MB).
Writing the frozen 8 B base on every improvement would cost 16 GB a time for
weights that are already on the Hub; the stored config is enough for a loader to
rebuild the architecture and re-attach the adapters.

## What the RoBERTa version already established

Carried over into this repo's defaults, so it does not have to be rediscovered:

1. **Deep supervision is what makes the collapse work** — dropping the four
   Stage I/II losses cost 2.21 pp (p < 0.001, zero seed overlap). Gradients from
   `L_fusion` do reach Stage I, but reaching is not the same as training.
2. **Every block needs its own learning rate.** The cascade gave each one its
   own schedule for free; a naive one-stage port silently removes that. Audio
   Stage I starved at the text branch's rate (wF1 0.24), and audio Stage II
   oscillated across seeds (std 0.168) when run at the Stage I head's 1e-4.
   Hence three groups: LoRA adapters 5e-5, CARE Stage I head 1e-4, everything
   else 5e-5.
3. **`L_MSP` acts as a regulariser, not a source of emotion knowledge** —
   removing it cost 1.1 pp (marginal, p ≈ 0.12), doubled the fusion variance and
   pulled the best epoch earlier.

## Status

- [ ] Smoke test on the cluster (shapes, gradient reachability, peak memory)
- [ ] 5-seed sweep, `lambda_aux = 1.0`
- [ ] 5-seed sweep, `lambda_aux = 0.0`
- [ ] Ablations over the remaining lambda terms

## Reference

Soumya Dutta and Sriram Ganapathy, "LLM supervised Pre-training for Multimodal
Emotion Recognition in Conversations", ICASSP 2025.

Component code is ported from `merits-l-llama`, `merits-l-text` and
`care-training`. This is an independent extension, not an official
implementation.
