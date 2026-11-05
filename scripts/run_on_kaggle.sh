#!/usr/bin/env bash
# Запуск пробного прогона (3 млн взаимодействий, 5 эпох) в ядре Kaggle.
#
# Подготовка:
#   1. pip install kaggle
#   2. Положите API-токен в ~/.kaggle/kaggle.json
#      (Kaggle -> Settings -> API -> Create New Token)
#
# Запуск:
#   bash scripts/run_on_kaggle.sh

set -euo pipefail

KERNEL="${KERNEL:-gshjis/sasrec-training-yambda-3m-test-run}"
KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/kaggle"
OUTPUT_DIR="${OUTPUT_DIR:-./kaggle_output}"

echo "=== 1/3. Push ядра $KERNEL ==="
kaggle kernels push -p "$KERNEL_DIR"

echo "=== 2/3. Статус ядра ==="
kaggle kernels status "$KERNEL"

echo "=== 3/3. Скачивание результатов в $OUTPUT_DIR ==="
mkdir -p "$OUTPUT_DIR"
kaggle kernels output "$KERNEL" -p "$OUTPUT_DIR"

echo "Готово. Логи: $OUTPUT_DIR"
