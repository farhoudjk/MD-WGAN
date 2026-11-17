#!/usr/bin/env bash
# Package the paper, figures, tables, code and results into one zip.
#
# Run:  bash make_package.sh
# Out:  mdwgan_package_<date>.zip  in this directory
set -euo pipefail
cd "$(dirname "$0")"

STAMP=$(date +%Y%m%d)
OUT="mdwgan_package_${STAMP}.zip"
rm -f "$OUT"

# Results are ~55 MB of JSON; keep them, they are the evidence for every number.
# The .npz dataset cache is excluded: it is regenerable from data.csv.
zip -r -q "$OUT" \
  mdwgan/paper \
  mdwgan/paper_assets \
  mdwgan/docs \
  mdwgan/core \
  mdwgan/experiments \
  mdwgan/analysis \
  mdwgan/runs \
  mdwgan/ns3_runs \
  mdwgan/README.md \
  -x 'mdwgan/**/__pycache__/*' \
  -x 'mdwgan/cache/*' \
  -x 'mdwgan/tmp/*' \
  -x 'mdwgan/runs/_progress/*'

# The ns-3 simulation lives outside this tree.
zip -q "$OUT" ../../scratch/mdwgan-comm.cc 2>/dev/null || true

echo "wrote $OUT"
unzip -l "$OUT" | tail -1
