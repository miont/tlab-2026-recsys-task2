# RL-дообучение последовательной рекомендательной модели (Yambda-50M)

Задание: [`docs/task.md`](docs/task.md). План исследования: [`docs/codeagents/research_plan.md`](docs/codeagents/research_plan.md).

## Структура

```
configs/            YAML-конфиги экспериментов (base.yaml)
docs/               задание и план исследования
notebooks/          тонкие точки входа для Google Colab
results/            итоговые таблицы (CSV) и графики (PNG) по этапам, из Colab
src/recsys_rl/
  config.py         dataclass-конфиги, загрузка/сохранение YAML
  utils.py          сиды, выбор устройства, атомарные чекпоинты
  data.py           загрузка, Global Temporal Split, словарь, окна       (этап 1)
  metrics.py        Recall/NDCG/MAP, Coverage, Gini, повторы, ILD         (этап 1)
  model.py          GPT-2 next-item рекомендатель (π_ref)                 (этап 2)
  train.py          обучение базовой модели: sampled softmax, early stop   (этап 2)
  evaluate.py       оценка на val/test, бейзлайны MostPop и Repeat        (этап 2)
  reward_model.py   модель отклика: ratio / listen+ / skip / like / continue, (этап 3)
                    ансамбль голов, обучение proxy- и gold-RM
  rm_eval.py        оценка RM: бейзлайны, срезы, калибровка, чувствительность (этап 3)
  rewards.py        функции награды R1–R6                                 (этапы 4–6)
  rl.py             RLOO/GRPO + KL, off-policy REINFORCE, DPO             (этапы 4–5)
  generate.py       top-K, reranking, multi-step роллауты                 (этап 6)
  cli/download.py   загрузка Yambda с Hugging Face
  cli/train_base.py обучение и оценка базовой модели (yambda-train)
  cli/train_rm.py   обучение и оценка reward model (yambda-train-rm)
tests/              pytest
```

## Установка (локально, [uv](https://docs.astral.sh/uv/))

```bash
uv sync                      # зависимости + dev-группа (pytest, ruff, ipykernel)
uv sync --extra tracking     # + wandb
uv run pytest
uv run ruff check .
```

Данные (по умолчанию в `./data`, путь задаётся переменной `RECSYS_DATA_ROOT`):

```bash
uv run yambda-download                # flat/50m/multi_event + artist/album mappings (~0.5 GB)
uv run yambda-download --embeddings   # + аудио-эмбеддинги (~13.8 GB)
```

## Google Colab

Откройте `notebooks/00_colab_setup.ipynb`. Он монтирует Google Drive (`RECSYS_DATA_ROOT=/content/drive/MyDrive/tlab-recsys`), клонирует репозиторий, ставит пакет через `uv pip install --system -e .` и скачивает данные. Нижние границы версий в `pyproject.toml` намеренно мягкие, чтобы не переустанавливать предустановленные в Colab torch и numpy. Точные локальные версии зафиксированы в `uv.lock`.
