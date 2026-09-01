#!/usr/bin/env bash
# 5-seed sweep — merits-l-llama is reported as a 5-seed mean, so a single joint
# run is not comparable to it no matter how good it looks.
#
#   bash scripts/run_multiseed.sh msp                       # lambda_aux = 1.0 (default)
#   bash scripts/run_multiseed.sh nomsp loss.lambda_aux=0.0
#
# Arg 1 (optional) is a tag for the output directory; everything after it is
# passed through to --override, so ablations reuse this script unchanged.
set -euo pipefail

CONFIG="${CONFIG:-configs/one_stage_iemocap_llama.yaml}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TAG="${1:-msp}"
shift || true
EXTRA=("$@")

for seed in $SEEDS; do
    out="outputs/${TAG}/seed_${seed}"
    if [ -f "${out}/result.json" ]; then
        echo "== seed ${seed}: already done (${out}/result.json), skipping"
        continue
    fi
    echo "== seed ${seed} -> ${out}"
    python -m src.train_joint --config "${CONFIG}" --override \
        seed="${seed}" \
        output_dir="${out}" \
        run_name="${TAG}_seed${seed}" \
        "${EXTRA[@]}"
done

echo
python -m scripts.summarize_seeds "outputs/${TAG}"
