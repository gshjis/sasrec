#!/usr/bin/env bash
# Скрипт для запуска проекта на Kaggle ядре через Kaggle CLI.
#
# Предварительно:
#   1. Установите kaggle CLI: pip install kaggle
#   2. Настройте API ключ: ~/.kaggle/kaggle.json (от https://www.kaggle.com/account)
#   3. Создайте ядро на Kaggle (или используйте существующее)
#   4. Замените <USER> и <KERNEL_SLUG> в kaggle-metadata.json

set -euo pipefail

KERNEL_SLUG="${KERNEL_SLUG:-informatica-sasrec-training}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-<USER>}"

echo "=== Kaggle CLI run: $KERNEL_SLUG ==="

# 1. Пушим код на ядро Kaggle (с GPU, ядро запустится автоматически)
echo "[1/3] Push kernel to Kaggle..."
kaggle kernels push -p . -u "$KAGGLE_USERNAME" --accelerator NvidiaTeslaT4

# 2. Ждём завершения и получаем статус
echo "[2/3] Kernel pushed. Monitor at: https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_SLUG"
echo "Wait for completion, then check output."

# 3. Вытянуть результаты (логи, метрики)
echo "[3/3] Pull results..."
kaggle kernels pull "$KAGGLE_USERNAME/$KERNEL_SLUG" -p ./kaggle_output

echo "=== Done ==="
