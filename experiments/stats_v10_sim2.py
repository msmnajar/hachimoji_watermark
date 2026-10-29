# sparse-event and cohort-size coverage of M1 / M2 (revision 10 protocol review; synthetic only)
import json, sys, numpy as np
sys.path.insert(0, "/home/claude/v3/experiments")
from stats_v10 import coverage_sim
res = []
res += coverage_sim(N=1000, p0s=(1e-6, 3e-6), kappas=(0, 0.2, 0.05), R=300, B=300, seed=1)
res += coverage_sim(N=842, p0s=(1e-5, 1e-4), kappas=(0, 0.2, 0.05), R=300, B=300, seed=2)
json.dump(res, open(sys.argv[1], "w"), indent=1)
