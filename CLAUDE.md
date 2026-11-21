# DX300 boom RL study

RL energy-vs-accuracy study for the DX300 excavator boom. Algorithm: SAC (stable-baselines3).

## Conventions

- Run all scripts from the repo root with `.venv` activated (`source .venv/bin/activate`).
- Boom length is 6.245 m, defined in `dx300sim/dx300_params.py`. Read it from there; do not hardcode it.
- Scripts are numbered `rl/0N_*.py`.
- Never start long training runs yourself. Give the user the command to run in their terminal.
- `rl/models/` (trained models) and `rl/tb/` (TensorBoard logs) are gitignored.

## New labs: reproducibility rules

- Start every new lab from `rl/00_lab_template.py` + `configs/lab_template.yaml` (Hydra). Copy both, rename to the lab number.
- No hardcoded parameters in new scripts: they go in the YAML config and are overridden on the command line.
- Seed everything with `dx300sim.runlog.set_global_seed(cfg.seed)` and pass the seed to SB3 and `env.reset(seed=...)`.
- Log every run with `dx300sim.runlog.start_run(...)` / `run.finish(metrics)`. This appends a row to `runs/runs.csv` (git commit, seed, config, metrics). Commit `runs/runs.csv` together with the results it produced.
- Commit code before a real run: a run started with uncommitted changes prints a warning and is marked `git_dirty` in `runs.csv`.
- Existing labs (`rl/01`–`rl/14`) are left as they are. Do not refactor them to Hydra unless asked: their reported results must stay reproducible exactly as written.
- Optional W&B mirror: `DX300_WANDB=1 python rl/NN_x.py ...` (offline by default, `wandb sync` uploads later).

## Workflow

- The Superpowers plugin is installed: plan before coding (brainstorming, writing-plans), write tests first (test-driven-development), and verify before saying a task is done (verification-before-completion).
- One task per session. Show test output as proof; "it should work" is not enough.

## Deadline

- 15 Nov: operating-point write-up due.
