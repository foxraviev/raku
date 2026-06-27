#!/usr/bin/env bash
set -euo pipefail

# Four-GPU run reproducing the ODIR-5K headline numbers (Table 1).
# Override CONFIG to point at any sheet under benchtops/kiln/.

CONFIG="${CONFIG:-benchtops/kiln/main.toml}"
NPROC="${NPROC:-4}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  -m raku.wheelhead.console throw "${CONFIG}"
