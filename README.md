# Informatica — SASRec Next-Item Recommendation

Обучение модели **SASRec** (Self-Attentive Sequential Recommendation) на задачу next-item предсказания по данным **Yandex Music "yambda 50m listens"**. Сравнение трёх стратегий логитов: `plain` (in-batch negatives), `standard logQ` и `improved logQ` (по статье Khrylchenko et al., RecSys'25, arXiv:2507.09331).

## Структура проекта

```
informatica/
├── pyproject.toml            # Зависимости и сборка (poetry)
├── requirements.txt          # Pinned версии для Kaggle + pip
├── README.md                 # Этот файл
├── train_model.py            # Точка входа обучения (рефакторинг Этапа Б)
├── kaggle/
│   ├── kaggle_entry.py       # Тонкий оркестратор для Kaggle ядра
│   ├── kernel-metadata.json  # Конфиг ядра Kaggle
│   └── requirements.txt      # Зависимости для ядра Kaggle
├── scripts/
│   ├── prepare_3m.py         # Подготовка выборки с сессиями (train/val/test)
│   ├── run_ablations.py      # Запуск матрицы абляционных исследований
│   ├── aggregate_ablations.py # Агрегация результатов абляций
│   └── run_on_kaggle.sh      # Скрипт запуска на Kaggle
├── src/
│   ├── paths.py              # Абсолютные пути (независимость от CWD)
│   ├── config.py             # Единый конфиг (dataclass, presets smoke/full/ablation)
│   ├── run_meta.py           # Метаданные воспроизводимости (run_meta.json)
│   ├── checkpoint.py         # Чекпоинты best.pt / last.pt
│   ├── pipeline.py           # Сквозной конвейер (PREPARE → TRAIN → EVALUATE → COMPARE)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── compare_runs.py   # Агрегация metrics.jsonl → CSV/Markdown
│   ├── data_extractor/       # Загрузка данных с Hugging Face
│   ├── data_preprocessing/   # Split (global_split.py), сессии (session.py), токенизатор
│   └── models/sasrec/        # Модель, формирование батчей, метрики
├── logs/                     # Логи запусков
├── data/
│   ├── raw/                  # Сырые данные (parquet)
│   └── processed/            # Подготовленные train/val/test parquet
├── runs/                     # Результаты запусков (run_id/ с run_meta.json, metrics.jsonl)
└── tests/                    # Тесты (pytest)
```

## Быстрый старт

### Локальный запуск (smoke-прогон)

```bash
# 1. Подготовить данные
PYTHONPATH=src python scripts/prepare_3m.py \
    --interactions 100000 \
    --seed 42 \
    --train-share 0.70 --val-share 0.15 --test-share 0.15

# 2. Обучить (все три режима logQ, единая температура)
PYTHONPATH=src python train_model.py \
    --mode smoke \
    --logq-mode all \
    --temperature 1.0 \
    --epochs 2 \
    --batch-size 256 \
    --seed 42 \
    --device cpu \
    --with-test

# 3. Сравнить результаты
PYTHONPATH=src python -m src.evaluation.compare_runs \
    runs/run_improved_smoke runs/run_standard_smoke runs/run_plain_smoke
```

### Полный прогон (все данные, все эпохи, оценка на test)

```bash
PYTHONPATH=src python train_model.py \
    --mode full \
    --logq-mode improved \
    --temperature 1.0 \
    --epochs 10 \
    --batch-size 256 \
    --lr 1e-3 \
    --seed 42 \
    --device cuda \
    --with-test
```

### Абляционные исследования

```bash
# Smoke-матрица (12 комбинаций, 1 сид)
PYTHONPATH=src python scripts/run_ablations.py \
    --matrix smoke \
    --output-dir ./ablation_runs/smoke \
    --device cuda

# Полная матрица с 5 сидами
PYTHONPATH=src python scripts/run_ablations.py \
    --matrix full \
    --seeds 42,123,2024,777,2026 \
    --n-runs 5 \
    --output-dir ./ablation_runs/full \
    --device cuda

# Агрегация результатов
PYTHONPATH=src python scripts/aggregate_ablations.py \
    --input-dir ./ablation_runs/full \
    --output-dir ./ablation_runs/full/comparison
```

### Запуск на Kaggle

```bash
# Подготовка данных (отдельное ядро)
kaggle kernels push -p kaggle/  # ядро prepare-3m-dataset

# Абляции на разных ядрах (параллельно)
kaggle kernels push -p kaggle/  # ядро ablation-plain
kaggle kernels push -p kaggle/  # ядро ablation-standard
kaggle kernels push -p kaggle/  # ядро ablation-improved

# Сбор результатов
PYTHONPATH=src python scripts/aggregate_ablations.py \
    --input-dir ./kaggle_results \
    --output-dir ./kaggle_results/comparison
```

## Режимы запуска

| Режим | Взаимодействий | Эпохи | logQ-режимы | Температура | Оценка на test |
|-------|---------------|-------|-------------|-------------|----------------|
| `smoke` | 100 000 | 2 | все три | 1.0 | нет |
| `full` | все (3м/50м) | 10 | все три | 1.0 | да |
| `ablation` | из конфига | 5 | матрица вариантов | матрица | да |

## Конфигурация

Все гиперпараметры вынесены в `src/config.py` (dataclass). Поддерживаются presets:
- `ExperimentConfig.smoke()` — быстрая проверка
- `ExperimentConfig.full()` — полный прогон
- `ExperimentConfig.ablation()` — абляционные исследования

CLI-переопределения имеют приоритет над конфигурационным файлом:

```bash
python train_model.py --config configs/full.yaml --epochs 10 --lr 5e-4 --seed 123
```

## Метрики

| Метрика | Описание |
|---------|----------|
| `Recall@K` | Доля пользователей, у кого цель в топ-K |
| `NDCG@K` | Нормализованный дисконтированный кумулятивный Gewinn |
| `MRR@K` | Среднее обратное ранжирование первой релевантной позиции |
| `Coverage@K` | Доля каталога, покрытая топ-K |
| `Pareto_coverage@K` | Покрытие парето-множества (head ∪ tail) |
| `Pareto_head_coverage@K` | Покрытие парето-головы |
| `Pareto_tail_coverage@K` | Покрытие парето-хвоста |
| `Entropy` | Шанновская энтропия softmax по всем айтемам |

## Воспроизводимость

Каждый запуск создаёт `runs/<run_name>/run_meta.json` с фиксацией:
- Git-коммита и ветки
- Версий Python, torch, polars
- Сида (torch, numpy, random, PYTHONHASHSEED)
- Всех гиперпараметров и аргументов CLI
- Хэша сырых данных (SHA-256)
- Времени начала/окончания и продолжительности
- Статуса (success/failed) и ошибок

## Тесты

```bash
# Запуск всех тестов
pytest tests/

# Проверка качества кода
ruff check src/ scripts/ kaggle/ train_model.py
pyright src/ train_model.py
```

## Лицензия

Внутренний проект.
