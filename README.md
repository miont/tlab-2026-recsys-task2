# RL-дообучение последовательной рекомендательной модели (Yambda-50M)

Исследование: как RL-дообучение под модель отклика меняет поведение next-item рекомендателя. Базовая модель — GPT-2 на последовательностях прослушиваний Yambda-50M (Яндекс Музыка). Награду даёт reward model, обученная на логах.

- **Отчёт:** [`docs/report/report.pdf`](docs/report/report.pdf), исходник — [`docs/report/report.md`](docs/report/report.md).
- **Задание:** [`docs/task.md`](docs/task.md).
- **План исследования:** [`docs/codeagents/research_plan.md`](docs/codeagents/research_plan.md).
- **Журнал наблюдений и решений по ходу прогонов:** [`docs/codeagents/observations.md`](docs/codeagents/observations.md).

## Главное

| Метод | KL к π_ref | val NDCG@10 | test NDCG@10 | gold ratio (val) |
|---|---|---|---|---|
| Repeat (недавние разные треки пользователя) | — | 0.135 | 0.133 | — |
| GPT-2, π_ref | 0 | 0.127 | 0.122 | 0.665 |
| RLOO, R1 (E[ratio]), β = 0.1 | 0.14 | 0.129 | 0.124 | 0.678 |
| `distill` реранкера, β = 0.05 | 0.21 | 0.132 | 0.127 | 0.683 |
| Реранкер π_ref·exp(r/β), β = 0.1 | 0.08 | **0.135** | **0.130** | **0.685** |

- **RL максимизирует награду в основном через «дешёвые» пути.** Выдача становится короче, в ней меньше повторов, она концентрируется на меньшем наборе треков. NDCG@10 монотонно падает с KL (0.127 → 0.08 при KL 4). Gold RM переоптимизации не показывает: компромисс проходит между «удовлетворённостью по RM» и поведенческой релевантностью.
- **RLOO, GRPO, Best-of-N и онлайн-DPO при равном KL неразличимы.** Разница «при одном β» объясняется только масштабом регуляризации.
- **Лучший компромисс даёт переранжирование.** Аналитический оптимум той же KL-регуляризованной цели на top-100 π_ref повышает NDCG@10 на 6% (значимо на обоих днях) и поднимает gold-награду. RL — неполная амортизация этого оптимума. Дистилляция реранкера в одну модель лучше RLOO, но разрыв до реранкера не закрывает.
- **Варианты награды.** Время прослушивания взламывается длиной трека, награда за лайк переоптимизирует слабую голову RM. Нормировка ratio по длине (R4) убирает смещение к коротким трекам. Штраф за повтор артиста работает, но и здесь реранкер лучше RL.
- **Долгосрочный прокси p(continue)** почти целиком пользовательский: внутри пользователя он не связан ни с длиной сессии в минутах, ни с возвращением на следующий день. Его оптимизация идёт через короткие и популярные треки.
- **При генерации нескольких шагов** все политики, включая π_ref, зацикливаются на повторе одного трека. RM оценивают такую петлю высоко.
- **Off-policy REINFORCE на логах** (Matveev et al., 2026) учит подражать продовому рекомендателю: NDCG −19…−26%.

Ответы на пять вопросов задания — §5.1–5.5 отчёта, сопоставление с GPTRec — §5.6, off-policy ветка — §6.

## Структура

```
configs/base.yaml     конфиг эксперимента; поля и значения по умолчанию — src/recsys_rl/config.py
docs/                 задание, план, журнал наблюдений, отчёт (report.md → report.pdf)
notebooks/            тонкие точки входа для Google Colab (00–05)
results/stage*/       итоговые таблицы (CSV) и графики (PNG), выгруженные из Colab
src/recsys_rl/
  config.py           dataclass-конфиги, загрузка YAML, переопределения key=value
  utils.py            сиды, выбор устройства, атомарные чекпоинты
  data.py             загрузка, Global Temporal Split, словарь, сессии, EventStore   (этап 1)
  eda.py              статистики EDA                                                 (этап 1)
  metrics.py          Recall/NDCG/MAP, coverage, Gini, повторы, метрики статьи Yambda (этап 1)
  model.py            GPT-2 next-item рекомендатель с признаками отклика (π_ref)     (этап 2)
  train.py            обучение: sampled softmax + logQ, early stopping               (этап 2)
  evaluate.py         оценка на val/test, бейзлайны MostPop и Repeat                 (этап 2)
  reward_model.py     модель отклика: ratio / Listen+ / skip / like / dislike / continue,
                      бутстреп-ансамбль, proxy- и gold-RM                            (этап 3)
  rm_eval.py          оценка RM: бейзлайны, срезы, калибровка, чувствительность      (этап 3)
  rewards.py          награды R1–R6 и штрафы R5                                      (этапы 4–6)
  rl.py               RLOO / GRPO / BoN / DPO, distill / exact, slate, off-policy    (этапы 4–5)
  rerank.py           реранкер π_ref·exp(r/β) и жадный слейт                         (этапы 4–5)
  significance.py     парный бутстреп по пользователям, поправка Holm               (этап 5)
  ope.py              офлайн-оценки Step-IS / SNIPS для off-policy ветки             (этап 5)
  generate.py         многошаговые роллауты с симулированным откликом               (этап 6)
  longterm.py         офлайн-валидность долгосрочных прокси                          (этап 6)
  cli/                yambda-download, yambda-train, yambda-train-rm, yambda-train-rl
tests/                pytest
```

## Установка (локально, [uv](https://docs.astral.sh/uv/), Python 3.13)

```bash
uv sync                      # зависимости + dev-группа (pytest, ruff, ipykernel)
uv sync --extra tracking     # + wandb
uv run pytest                # тесты на синтетических данных, GPU не нужен
uv run ruff check .
```

Данные лежат в `./data`, путь задаётся переменной `RECSYS_DATA_ROOT`:

```bash
uv run yambda-download                # flat/50m/multi_event + артисты/альбомы (~0.5 ГБ)
uv run yambda-download --embeddings   # + аудио-эмбеддинги (~13.8 ГБ), нужны только gold-RM
```

Внутри `RECSYS_DATA_ROOT`:

- `raw/` — скачанные parquet;
- `processed/50m/` — кэш подготовленных массивов;
- `runs/<run_name>/` — чекпоинты и `log.jsonl`;
- `results/<stage>/` — таблицы и графики для отчёта.

## Воспроизведение

Все прогоны выполнялись в Google Colab (A100). Вся логика находится в `src/recsys_rl`, ноутбуки только вызывают её и сохраняют таблицы и графики. Каждый ноутбук начинается с одной и той же bootstrap-ячейки: монтирует Drive, клонирует репозиторий, ставит пакет. Чекпоинты и кэши лежат на Drive (`MyDrive/tlab-recsys`), поэтому после разрыва сессии обучение продолжается с `last.pt`, а готовые прогоны и таблицы бутстрепа не пересчитываются.

Репозиторий приватный, поэтому в секретах Colab нужен `GH_TOKEN`: fine-grained токен с доступом *Contents: Read-only* к этому репозиторию. Подробнее — [`notebooks/README.md`](notebooks/README.md).

Ноутбуки запускаются по порядку: каждый следующий читает модели предыдущих из `runs/`.

| Ноутбук | Этап | Что делает | Модели (`runs/`) | Результаты |
|---|---|---|---|---|
| `00_colab_setup` | 0 | Drive, установка, загрузка данных; опционально кэш аудио-эмбеддингов словаря | — | — |
| `01_eda` | 1 | EDA и выбор протокола по train | — | выводы в ноутбуке |
| `02_baseline` | 2 | обучение π_ref, бейзлайны, абляции, примеры | `base`, `no_feedback`, `listen_plus` | `results/stage2` |
| `03_reward_model` | 3 | proxy- и gold-RM, их оценка | `rm_proxy`, `rm_gold` | `results/stage3` |
| `04_rl` | 4–5 | RL (методы, сетка β, награды R1–R5), реранкер, бутстреп, дистилляция, разнообразие, листовой слейт, off-policy | `rl_*` | `results/stage4` |
| `05_analysis` | 6 | многошаговые роллауты, валидность долгосрочных прокси, R6 | `rl_continue_*`, `rl_distill_continue_*` | `results/stage6` |

После прогона папку `MyDrive/tlab-recsys/results/` нужно скопировать в `results/` репозитория.

Тот же пайплайн доступен из командной строки. Значения `key=value` переопределяют поля YAML, остальные поля секции сохраняются:

```bash
uv run yambda-train train.epochs=8                           # π_ref → runs/base (в отчёте 8 эпох)
uv run yambda-train-rm run_name=rm_proxy                     # proxy-RM (нужен runs/base/best.pt)
uv run yambda-train-rm run_name=rm_gold seed=7 rm.base_run=null \
    rm.audio_embeddings=true rm.user_fraction=0.8            # gold-RM (нужны аудио-эмбеддинги)
uv run yambda-train-rl run_name=rl_ratio_b0.1 rl.kl_beta=0.1 # RL (нужны base, rm_proxy, rm_gold)
uv run yambda-train-rl run_name=rl_distill_b0.05 rl.method=distill rl.kl_beta=0.05
uv run yambda-train train.max_steps=50 run_name=smoke        # короткий прогон для проверки
```

Переопределения для каждого RL-прогона отчёта (β, метод, награда, штрафы) перечислены в ячейках `04_rl` и `05_analysis`.

**Сборка отчёта:**

```bash
docs/report/build_pdf.sh
```

Скрипт собирает PDF через pandoc и XeLaTeX. Нужен TeX Live с polyglossia, fontspec и unicode-math, а также шрифты Noto Serif, Noto Sans, DejaVu Sans Mono и Latin Modern Math.

## Протокол оценки

- **Разбиение.** Global Temporal Split: train | 30 мин | val-день | 30 мин | test-день. В val и test остаются только пользователи из train.
- **Что считается только на train:** словарь, популярность и нормировки. На train обучаются и RM. Контексты для RL тоже берутся из train.
- **Выбор гиперпараметров** — по val. Test используется как второй независимый день.
- **Релевантность** — Listen+ (доля прослушивания ≥ 0.5).
- **Фильтр уже прослушанного** (`exclude_seen`) — отдельный режим оценки, а не скрытая постобработка.
- **Две RM:** proxy даёт награду для RL, gold используется только для оценки.
- **Значимость** — парный бутстреп по пользователям с поправкой Holm.
- **Совместимость с бенчмарком Yambda.** Метрики в формулах статьи Yambda (`metrics.yambda_paper_metrics`) сравнимы с опубликованным SASRec.
