# DX300 boom RL study

RL energy-vs-accuracy study for the DX300 excavator boom. Algorithm: SAC (stable-baselines3).

## Conventions

- Run all scripts from the repo root with `.venv` activated (`source .venv/bin/activate`).
- Boom length is 6.245 m, defined in `dx300sim/dx300_params.py`. Read it from there; do not hardcode it.
- Scripts are numbered `rl/0N_*.py`.
- Never start long training runs yourself. Give the user the command to run in their terminal.
- `rl/models/` (trained models) and `rl/tb/` (TensorBoard logs) are gitignored.

## Deadline

- 15 Nov: operating-point write-up due.
