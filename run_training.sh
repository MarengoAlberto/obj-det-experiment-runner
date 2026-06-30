#!/usr/bin/env bash
set -Eeuo pipefail

RUN_NAME="${1:?Usage: /app/run_training.sh <run-name>}"
OUTPUT_DIR="/app/outputs/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

terminate_self() {
  exit_code=$?

  echo "Training ended with exit code: ${exit_code}"
  echo "Requesting Pod termination..."

  # Runpod injects both values into Pods automatically.
  if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
    curl --fail --silent --show-error \
      --request DELETE \
      --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      || true
  else
    echo "RUNPOD_POD_ID or RUNPOD_API_KEY missing; Pod was not deleted."
  fi

  exit "${exit_code}"
}

trap terminate_self EXIT

NUM_GPUS="$(python -c 'import torch; print(torch.cuda.device_count())')"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No CUDA GPU detected."
  exit 1
fi

echo "Detected ${NUM_GPUS} GPU(s)."
echo "Starting run: ${RUN_NAME}"

set +e

torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  train.py \
  hydra.run.dir="${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

TRAIN_EXIT_CODE="${PIPESTATUS[0]}"

set -e

exit "${TRAIN_EXIT_CODE}"