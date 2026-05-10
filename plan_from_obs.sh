#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OBS_DIR="${OBS_DIR:-stretch_example/stretch_obs}"

python run_path_planner.py \
  --config configs/stretch_path_planner.yaml \
  --image "$OBS_DIR/rgb.png" \
  --depth "$OBS_DIR/depth_m.npy" \
  --goal-x 0.0 \
  --goal-y 10.0 \
  --mode rgbd \
  --output-dir "$OBS_DIR/planner_output"
