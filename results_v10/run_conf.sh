#!/bin/bash
# Protocol Section 9, step 4: final conformance on every selected configuration (C data), sharded by seed
cd /home/claude/v3; R=/home/claude/runs/v10; mkdir -p $R/conf
SEEDS=$1 python3 experiments/conformance_v10.py final $R $R/conf/conformance_final_seeds$(echo $1|tr -d ,).json > $R/logs/conf_final_$(echo $1|tr -d ,).log 2>&1 && echo "shard $1 DONE" || echo "shard $1 FAILED"
