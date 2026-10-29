#!/bin/bash
# Protocol Section 9, step 7: guarded final evaluation of U for one seed (frozen code, frozen weights)
cd /home/claude/v3; R=/home/claude/runs/v10
W=(/home/claude/v3/models/aifnet_v4.pt /home/claude/runs/multi/seed1/aifnet_v4.pt /home/claude/runs/multi/seed2/aifnet_v4.pt)
s=$1
echo "seed $s start $(date -u +%FT%TZ)"
python3 experiments/eval_final_v10.py $s ${W[$s]} /home/claude/data/v10_U $R > $R/logs/eval_seed$s.log 2>&1 && echo "seed $s DONE $(date -u +%FT%TZ)" || echo "seed $s FAILED rc=$? $(date -u +%FT%TZ)"
