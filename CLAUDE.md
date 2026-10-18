# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project: RL fine-tuning of a GPT-2 next-item recommender on Yambda-50M (Yandex Music). The task statement is `docs/task.md`; the full research plan (stages 0–7, evaluation protocol, reward variants R1–R6, risks) is `docs/codeagents/research_plan.md` — read the relevant section before implementing a stage. Docs and README are in Russian; code, comments and docstrings are in English.

Status: stage 0 (infrastructure) is done; stage 1 is in progress. `config.py`, `utils.py`, `cli/download.py` and `metrics.py` are implemented; `data.py` has loading, split, Listen+, sessions, vocabulary and eval targets (feedback bins and windows wait for the EDA). `model.py`, `reward_model.py`, `rewards.py`, `rl.py`, `generate.py` are docstring-only stubs that describe what each stage must provide.

## Commands

Uses [uv](https://docs.astral.sh/uv/), Python 3.13.

```bash
uv sync                                   # deps + dev group (pytest, ruff, ipykernel)
uv sync --extra tracking                  # + wandb
uv run pytest                             # all tests
uv run pytest tests/test_smoke.py::test_config_roundtrip   # single test
uv run ruff check .                       # lint (E, F, I, UP, B; line length 100)
uv run ruff format .
uv run yambda-download                    # flat/50m/multi_event + artist/album mappings (~0.5 GB)
uv run yambda-download --embeddings       # + audio embeddings (~13.8 GB)
```

Tests load `configs/base.yaml` by relative path, so run pytest from the repo root.

## Architecture

- **All logic lives in `src/recsys_rl`; notebooks are thin Colab entry points** (`notebooks/README.md` maps notebook → stage). Don't put real code in notebooks.
- **Config**: nested dataclasses in `config.py` (`Config` → `DataConfig`, `ModelConfig`, `TrainConfig`, `RLConfig`). `load_config(path, **overrides)` rejects unknown YAML keys, so a new option must be added to the dataclass first. YAML files in `configs/` override only the fields they list.
- **Paths**: everything derives from `Config.data_root` (`$RECSYS_DATA_ROOT`, default `./data`): `raw/` (downloaded parquet), `processed/<dataset_size>/` (cached numpy windows), `runs/<run_name>/` (checkpoints). On Colab `data_root` is a Google Drive folder so artifacts survive session restarts. `data/`, `runs/`, `*.pt`, `*.parquet`, `*.npy` are gitignored.
- **Checkpoints**: use `utils.save_checkpoint` (atomic write via tmp + rename) so interrupted Colab sessions never leave broken files. `utils.get_device` only picks CUDA if the installed torch build supports the GPU arch.
- **Colab compatibility**: dependency lower bounds in `pyproject.toml` are intentionally loose so Colab's preinstalled torch/numpy are reused (`uv pip install --system -e .`). Don't tighten them; exact local versions are in `uv.lock`.
- Data processing uses **polars** (pandas does not fit 46M rows in Colab RAM). The RL loop (RLOO/GRPO + KL to π_ref) is meant to be a custom loop, not TRL.

## Evaluation protocol (leak-prevention rules from the plan, §3)

- Global Temporal Split (as in the Yambda benchmarks): train | 30-min gap | val day | 30-min gap | test day; test is the last day before timestamp 26_000_000 (seconds). Val/test keep only train users. Val is used for hyperparameter/early stopping, including RL β and step selection.
- Vocabulary, popularity, per-track mean ratio and length normalisation are computed **on train only**. Reward models are trained on train only; RL contexts come from train, RL validation on the val day.
- Reward-model features for event t use only events < t.
- Relevance target is Listen+ (played ratio ≥ 0.5). The repeat filter (not recommending already-heard items) is a separate, explicitly reported evaluation mode, never hidden post-processing.
- Sanity check: SASRec on Yambda-50M Listen+ reaches NDCG@10 = 0.0748, Recall@10 = 0.0325, Recall@100 = 0.1026; much lower GPT-2 numbers indicate a pipeline bug. These paper numbers use buggy formulas (recall over target events, pooled NDCG), so compare them with `metrics.yambda_paper_metrics`, not `relevance_metrics`.
