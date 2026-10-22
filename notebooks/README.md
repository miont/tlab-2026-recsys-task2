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

Report artifacts: notebooks 02 and later write their final tables (CSV) and plots (PNG) to `results/<stage>/` under `RECSYS_DATA_ROOT` (`MyDrive/tlab-recsys/results/` in Colab). After a run, download that folder into the repo's `results/` and commit it; checkpoints and logs stay on Drive. Notebooks are committed without outputs.

GitHub token (the repo is private): create a fine-grained personal access token on GitHub (Settings → Developer settings → Fine-grained tokens) with *Repository access: only `miont/tlab-2026-recsys-task2`* and *Permissions → Contents: Read-only*. In Colab open *Secrets* (key icon in the left panel), add `GH_TOKEN` with the token as its value and switch on *Notebook access*. The bootstrap stops with an error if the secret is empty or the clone fails.
