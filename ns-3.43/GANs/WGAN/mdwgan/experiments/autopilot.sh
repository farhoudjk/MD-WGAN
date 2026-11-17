#!/usr/bin/env bash
# Runs unattended: waits for the sweep, then produces every paper asset.
# Survives disconnects; safe to re-run (everything downstream is idempotent).
PY=/home/ubuntu/ns-allinone-3.43/ns-3.43/myenv/bin/python3
cd /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN
LOG=mdwgan/autopilot.log
echo "autopilot start $(date)" >> $LOG

# 1. wait for the python sweep to finish (restart it if it died early)
while true; do
  n=$(ls mdwgan/runs/*/*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 230 ]; then echo "sweep done: $n" >> $LOG; break; fi
  if ! ps aux | grep -q "[r]unner.py --suites"; then
    echo "$(date) runner gone at $n/230, resuming" >> $LOG
    $PY mdwgan/experiments/runner.py --suites all --seeds 5 --rounds 60 \
      --generators 4 --disc-per-gen 5 --max-tracks 400 --workers 6 --threads 2 \
      --out mdwgan/runs >> mdwgan/sweep.log 2>&1
  fi
  sleep 120
done

# 2. ns-3 communication sweep (needs the cores the training sweep just freed)
echo "$(date) ns3 sweep" >> $LOG
ROUNDS=40 SEEDS=5 bash mdwgan/experiments/ns3_sweep.sh \
  /home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/mdwgan/ns3_runs >> $LOG 2>&1

# 3. analysis: attribution summary, LaTeX tables, figures
echo "$(date) analysis" >> $LOG
$PY mdwgan/analysis/summary.py > mdwgan/docs/ATTRIBUTION.txt 2>> $LOG
$PY mdwgan/analysis/report.py >> $LOG 2>&1

echo "autopilot done $(date)" >> $LOG
