#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${B3_DATA_ROOT:?Set B3_DATA_ROOT to the directory containing PASCALContext}"
export PYTHONPATH="$PWD/models/ops_dcnv3${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py --config_exp "${1:-configs/pascal/internimage_l.yaml}" --dataset_log PASCAL
