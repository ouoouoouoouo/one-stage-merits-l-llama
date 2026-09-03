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
| One-stage Llama, `lambda_aux = 1.0` | 1 | 0.8362 | 0.8288 ± 0.0074 |
| **One-stage Llama, `lambda_aux = 0`** | **1** | **0.8588** | **0.8420 ± 0.0114** |

## Result: the LLM-supervised term helps RoBERTa and hurts Llama

| Text encoder | with `L_MSP` | without | Δ | |
|---|---:|---:|---:|---|
| RoBERTa-large (355 M) | **0.8281** | 0.8172 | **+1.09** | p ≈ 0.12 |
| Llama-3.1-8B | 0.8288 | **0.8420** | **−1.32** | p ≈ 0.06 |

The sign flips. Four of five `nomsp` seeds beat every `msp` seed (`msp`
0.8209–0.8362, `nomsp` 0.8291–0.8588), and this is the second independent
experiment pointing the same way: the staged pipeline in `merits-l-llama` also
came out slightly worse with MSP pre-training (0.8550 vs 0.8567).

The mechanism is consistent with what the RoBERTa ablations showed. `L_MSP`
acts as a regulariser, not as a source of emotion knowledge — with a 355 M
encoder fine-tuned on 3,205 utterances that anchor is worth 1.1 pp, but
GPT-3.5's three-way polarity on noisy ASR transcripts is a *coarser* teacher
than an 8 B model's own pretrained knowledge, so for Llama it pulls the LoRA
adapters toward a weaker task. **The paper's title contribution does not
survive scaling the text encoder.**

### Against the staged pipeline

One-stage Llama (`nomsp`) reaches 0.8420 ± 0.0114 against the staged
0.8567 ± 0.0139 — −1.47 pp, t ≈ 1.83, p ≈ 0.10. Not significant, but unlike
the RoBERTa version (−0.24 pp, p ≈ 0.73) the direction is consistent.

Two caveats before reading that as "one-stage costs more with a stronger
encoder". First, the RoBERTa configuration went through four rounds of tuning
and this one is a straight transplant of it — the tuning effort is not
comparable. Second, the gap has an obvious candidate:

| audio Stage I wF1 | |
|---|---:|
| staged CARE downstream | 0.5787 |
| one-stage RoBERTa | 0.5547 |
| one-stage Llama, `msp` | 0.5388 |
| one-stage Llama, `nomsp` | **0.4580 ± 0.0577** |

The audio branch is the worst it has been in any configuration, with the
largest variance — the modality-imbalance failure the smoke test was built to
watch for. Note that `msp` has the *better* audio branch: slowing the text side
down gave audio room to catch up, yet it still loses on fusion, so the
text-side cost outweighs the audio-side gain.

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

## Cross-corpus: MSP-PODCAST 8-class probes

How much of MSP-PODCAST's eight-way emotion structure is already linearly
available in representations learned from IEMOCAP four-way? The trained model is
frozen and a fresh 8-way linear layer is fitted at every depth of the pipeline.

```bash
# 1. run MSP through the frozen model, caching T1 / S1 / T2 / S2 / h  (~1 h, ~2.2 GB)
python -m scripts.extract_msp_representations \
    --checkpoint outputs/nomsp/seed_1/best/one_stage.pt \
    --manifest   /home/ouo/AdaLTM-reproduction/8class_DropTextNAN.csv \
    --audio-dir  /home/ouo/dataset/MSP_Podcast/Audios \
    --care-ckpt  /home/ouo/care_training/ckpts_faithful/best.pth \
    --care-repo  /home/ouo/care_training/CARE/pretraining \
    --extract-script /home/ouo/care_training/care-training/scripts/extract_iemocap_care_downstream_style.py \
    --out data/cache/msp_reps_nomsp_seed1.pt

# 2. fit the probes  (minutes)
python -m scripts.probe_msp_8class --reps data/cache/msp_reps_nomsp_seed1.pt
```

Extraction runs at ~28 utterances/s, so the full 161 K takes ~1.6 h and writes
~4.5 GB. It shards cleanly by `Split_Set` across GPUs — Train is the long pole
at ~53 min — and `--reps` takes the pieces back as a list:

```bash
# Shards go in their own directory: a glob over data/cache/ will otherwise pick
# up any earlier --limit smoke-test cache, which the probe refuses (it checks
# for repeated utterances) but only after the load.
i=0; for s in Train Test1 Development Test2; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m scripts.extract_msp_representations \
      ... --splits $s --out data/cache/msp8/${s}.pt &
  i=$((i+1)); done
python -m scripts.probe_msp_8class --reps data/cache/msp8/*.pt
```

Data comes from AdaLTM's `8class_DropTextNAN.csv` — MSP-PODCAST 1.12, 161,350
utterances with the official `Split_Set`, the 8-class `EmoClass`, and ASR text,
so no transcription step is needed. Label map, class weighting and the reported
metrics follow AdaLTM (`utils/data/podcast.py`, `tools/eval_pretrained_ser.py`):
UAR, macro precision, macro-F1 with bootstrap CIs on Test1 and Test2. Chance is
12.5 % UAR; AdaLTM quotes 35.56 % macro-F1 for vox-profile's fine-tuned
WavLM-large, which is the scale these numbers live on — **not** the 0.84 of
IEMOCAP four-way.

Two mismatches, both handled explicitly:

* **MSP has no dialogues.** Utterances are grouped into pseudo-conversations by
  podcast show (`MSP-PODCAST_<show>_<segment>`, in segment order): 4,970 shows,
  median 15 segments, the same order of magnitude as IEMOCAP's ~35. `--no-group`
  makes every utterance a length-1 conversation instead, which turns Stage II/III
  into near-identity — the honest ablation for what the conversation blocks add.
* **Class weighting is not optional.** Neutral is 34 % of Train against Fear's
  1.3 %; without the effective-number weights a probe predicts Neutral and
  reports near-zero macro-F1.

### Result (`nomsp` seed 1, Test1, n = 34,778)

| probe | dim | UAR | Macro-F1 |
|---|---:|---:|---:|
| `t1` Llama utterance | 4096 | 28.74 ± 0.71 | 24.76 ± 0.44 |
| `s1` CARE Stage I | 256 | 31.57 ± 0.71 | **28.58 ± 0.50** |
| `t2` text Stage II | 2048 | 31.97 ± 0.85 | 25.71 ± 0.42 |
| `s2` audio Stage II | 256 | 31.30 ± 0.77 | 27.17 ± 0.43 |
| **`t2‖s2`** | 2304 | **33.06 ± 0.85** | 28.23 ± 0.46 |
| `h` Stage III fused | 256 | 31.73 ± 0.67 | 28.10 ± 0.45 |
| chance | | 12.50 | |
| AdaLTM's reference: fine-tuned WavLM-large | | | 35.56 |

**Transfer is substantial.** The best probe is 2.6× chance, and its macro-F1
reaches ~80 % of a WavLM-large *fine-tuned on MSP 8-class* — from a single
linear layer on frozen features learned on a different corpus with a different
label space.

**The modality ranking inverts.** Audio beats text here (`s1` 28.58 against `t1`
24.76), the opposite of IEMOCAP, where the text branch dominates (Stage II 0.80
against 0.64). It is not capacity: `s1` is 256-d and `t1` is 4096-d, and a
linear probe on 89 K training samples is helped, not hurt, by width.

**But do not read that as "audio transfers better".** CARE was self-supervised
on MSP-PODCAST, so the audio branch has home-field advantage on this corpus's
acoustics and the text branch has none. Separating "CARE knows MSP audio" from
"the IEMOCAP-trained audio head transfers" needs a random-initialised control —
same architecture, same frozen Llama base, every trained weight reset. If the
random `s1` probes near 28.58, the advantage is entirely CARE's pre-training.

**Stage III compresses away usable information.** `h` (256-d, 31.73 UAR) does
not beat its own input `t2‖s2` (2304-d, 33.06). That 2304→256 projection was
fitted for IEMOCAP 4-way, and under a different task it is lossy.

Test2 sits 8–9 pp lower throughout, which is class balance rather than model
quality: Neutral is 58 % of Test2 against 34 % of Test1, and macro metrics are
sensitive to that. Compare each test set against chance, not against each other.

**Disclose this in any writeup:** CARE was self-supervised on MSP-PODCAST v1.11,
so part of this test set was seen (unlabelled) during its pre-training. The audio
branch is therefore not a clean cross-corpus transfer. The text branch is clean
when probing a `lambda_aux = 0` checkpoint, which is why `nomsp` is the arm to
use here.

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

- [x] Smoke test on the cluster — 18/18, peak 15.57 GiB at B=1 K=46 L=43
- [x] 5-seed sweep, `lambda_aux = 1.0` — 0.8288 ± 0.0074
- [x] 5-seed sweep, `lambda_aux = 0.0` — **0.8420 ± 0.0114**, the better arm
- [x] Smoothed checkpoint selection — **tested and rejected**, 0.8286 ± 0.0194
      against 0.8420 ± 0.0114. Default back to 1.
- [x] `lambda_aux` sensitivity sweep, 6 arms × 5 seeds — see below
- [ ] Ablations over the remaining lambda terms

## The `lambda_aux` sweep

| arm | `lambda_aux` | fusion wF1 | vs λ = 0 | |
|---|---|---:|---:|---|
| `nomsp` | **0** | **0.8420 ± 0.0114** | — | |
| `lam01` | 0.1, cosine | 0.8272 ± 0.0137 | −1.48 | p ≈ 0.10 |
| `lam03` | 0.3, cosine | 0.8380 ± 0.0090 | −0.40 | p ≈ 0.55 |
| `msp` | 1.0, cosine | 0.8288 ± 0.0074 | −1.32 | p ≈ 0.06 |
| `phase15` | 1.0, phase 15 % | **0.8394 ± 0.0076** | −0.26 | p ≈ 0.68 |
| `phase03` | 1.0, phase 30 % | 0.8273 ± 0.0143 | −1.47 | p ≈ 0.10 |

**Read the shape before the numbers: the curve is not monotone.** If the harm
grew with the strength of the auxiliary term, `lam01` (λ = 0.1) would sit
closest to λ = 0; it is the lowest of all six. `phase03` uses only slightly more
MSP than `phase15` and lands 1.2 pp below it. There is no mechanism for that, so
at five seeds most of the spread across these arms is noise — the whole sweep
spans 1.48 pp against a per-arm standard error of 0.003–0.006.

The one statement that survives: **no setting of `lambda_aux` beats 0.** The
best case is parity, and `phase15` and `lam03` reach it (both statistically
indistinguishable from λ = 0).

For a reported system that keeps the paper's LLM-supervised objective,
`phase15` is the defensible pick: parity with λ = 0, the lowest variance of any
MSP-bearing arm, and a schedule that is the literal translation of "pre-train
for 10 epochs, then stop" rather than a swept constant. Stated carefully:
**retaining the LLM-supervised objective costs nothing measurable, provided it
is applied as an early phase rather than as a term that runs the whole way.**
Against the RoBERTa repo that completes a clean picture — the same objective
wants to stay on throughout for a 355 M encoder (λ = 1.0, cosine) and to switch
off after 15 % of training for an 8 B one.

Reporting caveat: picking the maximum of six arms inflates it by roughly 1 pp at
this standard error. Report the curve, phrase comparisons as "indistinguishable
from λ = 0", and avoid point-to-point claims such as "λ = 0.1 beats λ = 1.0",
which is exactly the part driven by noise. Firming up `phase15` vs `nomsp`
would need 10+ seeds per arm.

### Two hypotheses this repo has already falsified

**"The audio branch is the gap."** Audio Stage I is worst in the arm with the
*best* fusion (`nomsp` 0.4580 / 0.8420 against `msp` 0.5388 / 0.8288), and
`nomsp` is worse than `msp` on three of four auxiliary heads while winning on
fusion. In this regime the auxiliary head metrics do not predict fusion
performance — they measure how linearly separable a representation is at that
depth, which is not what co-attention needs. Tuning `lambda_audio_stage1` or
`audio_lr` to lift audio Stage I would likely cost fusion points.

**"Val selection is too noisy, so smooth it."** Rejected above. The val curve is
a real early peak, not a noisy plateau, so a moving average systematically
selects down the far side. The ~9 pp val/test gap is more plausibly a genuine
session 1 vs session 5 difficulty difference.

The modality-balance ratio does drift upward during training (~2.6 → ~3.8 by the
end, identically in both arms), so co-attention really does shift toward text —
but that drift is not what separates the arms, and part of the level is
mechanical (`proj_text` is 2048→256 against `proj_audio`'s 256→256).

## Reference

Soumya Dutta and Sriram Ganapathy, "LLM supervised Pre-training for Multimodal
Emotion Recognition in Conversations", ICASSP 2025.

Component code is ported from `merits-l-llama`, `merits-l-text` and
`care-training`. This is an independent extension, not an official
implementation.
