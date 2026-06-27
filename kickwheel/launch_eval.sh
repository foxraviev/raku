#!/usr/bin/env bash
set -euo pipefail

# Score a trained checkpoint on the test split and print the metric block.

CONFIG="${CONFIG:-benchtops/kiln/main.toml}"
WEIGHTS="${WEIGHTS:-firings/odir5k_main/final.pt}"

python -m raku.wheelhead.console grade "${CONFIG}" "${WEIGHTS}"
