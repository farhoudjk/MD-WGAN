#!/usr/bin/env bash
# Full sweep on corrected (segment) non-IID partitions.
PY=/home/ubuntu/ns-allinone-3.43/ns-3.43/myenv/bin/python3
cd /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN
LOG=mdwgan/sweep2.log
R="--seeds 5 --rounds 60 --generators 2 --disc-per-gen 5 --max-tracks 400 \
   --noniid segment --workers 7 --threads 2 --out mdwgan/runs"
echo "=== start $(date) ===" >> $LOG
for s in baseline sota noniid ablation candidates diversity sensitivity; do
  echo "--- $s $(date) ---" >> $LOG
  $PY mdwgan/experiments/runner.py --suites $s $R >> mdwgan/sweep2_jobs.log 2>&1
  echo "$s done: $(ls mdwgan/runs/*/*.json 2>/dev/null|wc -l) total" >> $LOG
  $PY mdwgan/analysis/pertable.py mdwgan/runs/$s > mdwgan/docs/TABLE_$s.txt 2>/dev/null
done
$PY mdwgan/analysis/allmetrics.py mdwgan/runs/sota > mdwgan/docs/RANK_sota.txt 2>/dev/null
$PY mdwgan/analysis/final_report.py > mdwgan/docs/RESULTS.md 2>/dev/null
$PY mdwgan/analysis/figures.py >> $LOG 2>&1
$PY mdwgan/analysis/paper_tables.py >> $LOG 2>&1
echo "=== done $(date) ===" >> $LOG
