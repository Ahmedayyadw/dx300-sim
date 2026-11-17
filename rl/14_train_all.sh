#!/usr/bin/env bash
# Lab 14b-14d: all 19 training runs, 1 thread each, at most 14 at a time (longest first).
# Prerequisite: rl/models/14_dataset_1M.npz (python rl/14_relabel.py). Resumable: finished models are skipped.
# Run from the repo root:  bash rl/14_train_all.sh 2>&1 | tee rl/models/14_train_all.log
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ON=".venv/bin/python rl/14_online_train.py"
OFF=".venv-d3rlpy/bin/python rl/14_offline_train.py"
{
  for s in 0 1 2; do echo "$OFF --algo cql --seed $s"; done          # 14b offline, ~3.5-4.5 h each
  for s in 0 1 2; do echo "$OFF --algo iql --seed $s"; done          # 14b offline, ~0.6 h
  for s in 0 1 2; do echo "$ON --algo sac --seed $s"; done           # 14b online, ~0.5 h
  for s in 0 1 2; do echo "$ON --algo td3 --seed $s"; done           # 14b online, ~0.5 h
  for a in sac td3; do for x in 0.3 3; do echo "$ON --algo $a --seed 0 --lr $x"; done; done  # 14d
  for s in 0 1 2; do echo "$OFF --algo bc --seed $s"; done           # 14c, ~0.15 h
} | xargs -P 14 -I{} bash -c '{} 2>&1 | grep -E "^14_|Error|error" ; true'
echo "done: $(date)"
