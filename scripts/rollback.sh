#!/usr/bin/env bash
set -euo pipefail

# Stub rollback script. Replace the placeholders with your registry name,
# version selection logic, and deployment command.

MODEL_NAME="${MODEL_NAME:-churn-model}"
TARGET_VERSION="${TARGET_VERSION:-}"

if [[ -z "${TARGET_VERSION}" ]]; then
  echo "Set TARGET_VERSION to the model version you want to restore."
  exit 1
fi

echo "Promoting ${MODEL_NAME} version ${TARGET_VERSION} to Production..."
echo "mlflow models transition-stage --name ${MODEL_NAME} --version ${TARGET_VERSION} --stage Production"
echo "Redeploy your serving stack after the transition completes."
