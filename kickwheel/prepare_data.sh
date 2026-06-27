#!/usr/bin/env bash
set -euo pipefail

# Lay out the three fundus collections under data/. Access points and licences
# are listed in clay-beds.txt; downloads require the respective accounts.
#
# Each split is read from data/<dataset>/<split>.csv with rows
#   relative/image/path.jpg,01001000
# where the second field is either a 0/1 string of length num_classes or a
# space-separated list of positive class indices. Images sit under
# data/<dataset>/ at the recorded relative path.

ROOT="${ROOT:-data}"

for ds in odir5k rfmid jsiec; do
  mkdir -p "${ROOT}/${ds}"
done

cat <<'NOTE'
Place the downloaded collections as follows:

  data/odir5k/{train,val,test}.csv   + image folders   (8 classes)
  data/rfmid/{train,val,test}.csv    + image folders   (5 mapped classes)
  data/jsiec/{train,val,test}.csv    + image folders   (3 mapped classes)

When a split CSV is absent the loader falls back to a deterministic stand-in so
the pipeline and tests still run offline.
NOTE
