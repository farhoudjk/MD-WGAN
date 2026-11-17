#!/usr/bin/env bash
# Unattended overnight pipeline. Ordered by decision value: the suites that
# determine the recommended method run first, so a partial result is still
# actionable.
PY=/home/ubuntu/ns-allinone-3.43/ns-3.43/myenv/bin/python3
cd /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN
LOG=mdwgan/overnight.log
R="--seeds 5 --rounds 60 --generators 4 --disc-per-gen 5 --max-tracks 400 --workers 6 --threads 2 --out mdwgan/runs"
echo "=== overnight start $(date) ===" >> $LOG

for suite in sota candidates diversity ablation baseline noniid sensitivity; do
  echo "--- $suite $(date) ---" >> $LOG
  $PY mdwgan/experiments/runner.py --suites $suite $R >> mdwgan/sweep.log 2>&1
  echo "$suite done: $(ls mdwgan/runs/*/*.json 2>/dev/null|wc -l) total results" >> $LOG
  # Refresh analysis after every suite so partial results are always usable.
  $PY mdwgan/analysis/summary.py > mdwgan/docs/ATTRIBUTION.txt 2>/dev/null
  $PY mdwgan/analysis/final_report.py > mdwgan/docs/RESULTS.md 2>>$LOG
done

echo "--- ns3 $(date) ---" >> $LOG
ROUNDS=40 SEEDS=5 bash mdwgan/experiments/ns3_sweep.sh \
  /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/mdwgan/ns3_runs >> $LOG 2>&1

echo "--- final analysis $(date) ---" >> $LOG
$PY mdwgan/analysis/report.py >> $LOG 2>&1
$PY mdwgan/analysis/final_report.py > mdwgan/docs/RESULTS.md 2>>$LOG
echo "=== overnight done $(date) ===" >> $LOG
