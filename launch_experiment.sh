#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./launch_experiment.sh my_experiment_name
#
# Example:
#   ./launch_experiment.sh yolo640_ciou_seed1

RUN_NAME="${1:?Usage: ./launch_experiment.sh <run-name>}"

# Keep run names shell/path safe.
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run name may contain only letters, numbers, dots, underscores, and hyphens."
  exit 1
fi

# ---------- Customize these ----------
IMAGE="yourdockerhubuser/license-plate-trainer:0.1"
GPU="NVIDIA GeForce RTX 4090"
GPU_COUNT=1
CONTAINER_DISK_GB=30
LOCAL_CONFIG_DIR="./configs"
REMOTE_CONFIG_DIR="/app/configs"
REMOTE_OUTPUT_DIR="/app/outputs/${RUN_NAME}"
MAX_RUNTIME="12h" # Safety backup if something hangs
# -------------------------------------

for cmd in runpodctl jq rsync ssh; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}"
    exit 1
  fi
done

if [[ ! -d "${LOCAL_CONFIG_DIR}" ]]; then
  echo "Could not find local config directory: ${LOCAL_CONFIG_DIR}"
  exit 1
fi

if [[ ! -f "${LOCAL_CONFIG_DIR}/config.yaml" ]]; then
  echo "Could not find ${LOCAL_CONFIG_DIR}/config.yaml"
  exit 1
fi

POD_ID=""
TRAINING_STARTED="false"

cleanup_failed_setup() {
  status=$?

  # Before detached training starts, clean up a partially-created Pod.
  if [[ "${status}" -ne 0 ]] \
    && [[ -n "${POD_ID}" ]] \
    && [[ "${TRAINING_STARTED}" != "true" ]]; then
    echo
    echo "Setup failed. Deleting Pod ${POD_ID}..."
    runpodctl pod delete "${POD_ID}" >/dev/null 2>&1 || true
  fi

  exit "${status}"
}

trap cleanup_failed_setup EXIT

echo "Creating Runpod Pod for: ${RUN_NAME}"

CREATE_JSON="$(
  runpodctl pod create \
    --name "${RUN_NAME}" \
    --image "${IMAGE}" \
    --gpu-id "${GPU}" \
    --gpu-count "${GPU_COUNT}" \
    --container-disk-in-gb "${CONTAINER_DISK_GB}" \
    --ports "22/tcp" \
    --ssh \
    --docker-args "sleep infinity" \
    --terminate-after "${MAX_RUNTIME}"
    --env '{
      "WANDB_KEY": "{{ RUNPOD_SECRET_WANDB_KEY }}"
    }'
)"

POD_ID="$(printf '%s' "${CREATE_JSON}" | jq -r '.id // empty')"

if [[ -z "${POD_ID}" ]]; then
  echo "Could not get Pod ID from Runpod response:"
  echo "${CREATE_JSON}"
  exit 1
fi

echo "Created Pod: ${POD_ID}"
echo "Waiting for SSH..."

SSH_HOST=""
SSH_PORT=""
SSH_KEY=""

for _ in {1..90}; do
  SSH_INFO="$(runpodctl ssh info "${POD_ID}" 2>/dev/null || true)"
  SSH_COMMAND="$(printf '%s' "${SSH_INFO}" | jq -r '.sshCommand // empty' 2>/dev/null || true)"

  if [[ -n "${SSH_COMMAND}" ]]; then
    SSH_HOST="$(printf '%s\n' "${SSH_COMMAND}" | sed -nE 's/.*root@([^ ]+).*/\1/p')"
    SSH_PORT="$(printf '%s\n' "${SSH_COMMAND}" | sed -nE 's/.* -p ([0-9]+).*/\1/p')"
    SSH_KEY="$(printf '%s\n' "${SSH_COMMAND}" | sed -nE 's/.* -i ([^ ]+).*/\1/p')"

    # Expand ~/.ssh/... returned by runpodctl.
    SSH_KEY="${SSH_KEY/#\~/${HOME}}"

    if [[ -n "${SSH_HOST}" && -n "${SSH_PORT}" && -n "${SSH_KEY}" ]]; then
      if ssh \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=5 \
        -i "${SSH_KEY}" \
        -p "${SSH_PORT}" \
        "root@${SSH_HOST}" \
        "echo ready" >/dev/null 2>&1; then
        break
      fi
    fi
  fi

  sleep 5
done

if [[ -z "${SSH_HOST}" || -z "${SSH_PORT}" || -z "${SSH_KEY}" ]]; then
  echo "Pod did not become available through SSH in time."
  exit 1
fi

echo "Pod is reachable."
echo "Uploading local configs..."

# Ensure the target directory is clean before copying your current local configs.
ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -i "${SSH_KEY}" \
  -p "${SSH_PORT}" \
  "root@${SSH_HOST}" \
  "rm -rf '${REMOTE_CONFIG_DIR}' && mkdir -p '${REMOTE_CONFIG_DIR}' '${REMOTE_OUTPUT_DIR}'"

# The trailing slash matters: copy the CONTENTS of ./configs into /app/configs.
rsync -az --delete \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ${SSH_KEY} -p ${SSH_PORT}" \
  "${LOCAL_CONFIG_DIR}/" \
  "root@${SSH_HOST}:${REMOTE_CONFIG_DIR}/"

echo "Starting detached training..."

ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -i "${SSH_KEY}" \
  -p "${SSH_PORT}" \
  "root@${SSH_HOST}" \
  "bash -lc '
    mkdir -p \"${REMOTE_OUTPUT_DIR}\"
    nohup /app/run_training.sh \"${RUN_NAME}\" \
      > \"${REMOTE_OUTPUT_DIR}/launcher.log\" \
      2>&1 \
      < /dev/null &
    echo \$! > \"${REMOTE_OUTPUT_DIR}/training.pid\"
  '"

TRAINING_STARTED="true"
trap - EXIT

echo
echo "Training launched successfully."
echo "Pod ID: ${POD_ID}"
echo "Streaming logs now. Ctrl-C stops viewing only."
echo

ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -i "${SSH_KEY}" \
  -p "${SSH_PORT}" \
  "root@${SSH_HOST}" \
  "tail -n +1 -F '${REMOTE_OUTPUT_DIR}/train.log'"