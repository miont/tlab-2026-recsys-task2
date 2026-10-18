# Notebooks

Notebooks are thin entry points for Google Colab; all logic lives in `src/recsys_rl`.

| Notebook | Stage (see `docs/codeagents/research_plan.md`) |
|---|---|
| `00_colab_setup.ipynb` | Drive, install, data download |
| `01_eda.ipynb` | 1. Data, EDA, temporal split |
| `02_baseline.ipynb` | 2. GPT-2 next-item model |
| `03_reward_model.ipynb` | 3. Reward model and its evaluation |
| `04_rl.ipynb` | 4–5. RL fine-tuning, reward / KL sweeps |
| `05_analysis.ipynb` | 5–6. Trade-off, behaviour change, multi-step rollouts |

Each Colab notebook runs in its own runtime, so every notebook after `00` starts with the same bootstrap cell (mount Drive, clone, install, fix `sys.path`); locally that cell only changes the working directory to the repo root. The bootstrap installs the code from GitHub, so push changes before running a notebook in Colab.
