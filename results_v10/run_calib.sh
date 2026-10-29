#!/bin/bash
# Protocol Section 9, step 3: calibration on C for seeds 0, 1, 2 (sequential; features cached per modality)
cd /home/claude/v3
R=/home/claude/runs/v10
W=(/home/claude/v3/models/aifnet_v4.pt /home/claude/runs/multi/seed1/aifnet_v4.pt /home/claude/runs/multi/seed2/aifnet_v4.pt)
for s in 0 1 2; do
  [ -f $R/calib_selection_seed$s.json ] && continue
  echo "seed $s start $(date -u +%FT%TZ)"
  python3 experiments/calib_v10.py $s ${W[$s]} $R/manifest_v10.json /home/claude/data/v10_C $R > $R/logs/calib_seed$s.log 2>&1 || { echo "seed $s FAILED"; exit 1; }
  echo "seed $s done $(date -u +%FT%TZ)"
done
echo ALL_DONE
