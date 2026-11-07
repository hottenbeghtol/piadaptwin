#!/usr/bin/env bash
set -euo pipefail
python -m piadaptwin.console gauge --config_path "${1:-configs/experiment/main.yaml}"
