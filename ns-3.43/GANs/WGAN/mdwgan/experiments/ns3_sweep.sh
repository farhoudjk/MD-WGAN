#!/usr/bin/env bash
# Communication-scalability sweep over the ns-3 802.11p model.
#
# Produces the measured counterpart to the analytical O(S_sync * N_G * K_S * d)
# scalability claim: latency and byte overhead per federated round as the
# vehicle count, selected-subset size and payload vary.
set -euo pipefail

NS3_DIR=/home/ubuntu/ns-allinone-3.43/ns-3.43
OUT_DIR=${1:-/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/mdwgan/ns3_runs}
ROUNDS=${ROUNDS:-40}
SEEDS=${SEEDS:-5}

mkdir -p "$OUT_DIR"
cd "$NS3_DIR"

run() {  # run <tag> <extra args...>
  local tag=$1; shift
  for s in $(seq 1 "$SEEDS"); do
    local out="$OUT_DIR/${tag}_seed${s}.csv"
    [ -f "$out" ] && { echo "skip $tag seed$s"; continue; }
    echo "run $tag seed$s"
    ./ns3 run "mdwgan-comm --rounds=$ROUNDS --seed=$s --simArea=${AREA:-600} --out=$out $*" >/dev/null 2>&1 \
      || echo "FAILED $tag seed$s"
  done
}

# (a) scale the vehicle population (K)
for n in 8 16 24 32 40; do
  run "nveh${n}" --nVehicles=$n --nRsu=4 --kSelected=3
done

# (b) scale the selected subset (K_S) -- the knob the RL agent controls
for k in 1 2 3 4 5; do
  run "k${k}" --nVehicles=20 --nRsu=4 --kSelected=$k
done

# (c) scale the RSU count (N_G)
for g in 2 4 6 8; do
  run "nrsu${g}" --nVehicles=24 --nRsu=$g --kSelected=3
done

# (d) scale the per-message payload (d)
for p in 512 1024 2048 4096; do
  run "payload${p}" --nVehicles=20 --nRsu=4 --kSelected=3 --payloadBytes=$p
done

echo "ns-3 sweep complete -> $OUT_DIR"
