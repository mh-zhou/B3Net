#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../run_nyud.sh" configs/nyud/variants/internimage_l_20k.yaml
