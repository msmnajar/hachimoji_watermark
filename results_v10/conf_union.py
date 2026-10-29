import json, glob
R='/home/claude/runs/v10'
need=set()
for s in (0,1,2):
    S=json.load(open(f'{R}/calib_selection_seed{s}.json'))
    for m,sel in S['selection'].items():
        need|={json.dumps([s,m,v['cfg'],v['T']],sort_keys=True) for k,v in sel.items() if isinstance(v,dict) and 'cfg' in v}
have={}; tot={k:0 for k in ('n_attempts','T1_mismatch','T2_mismatch','T3_mismatch','helper_mismatch')}
for f in glob.glob(f'{R}/conf/conformance_final_*.json'):
    for c in json.load(open(f))['configs']:
        k=json.dumps([c['seed'],c['modality'],c['cfg'],c['T']],sort_keys=True)
        if k not in have:
            have[k]=c
            for t in tot: tot[t]+=c[t]
print(len(need), len(set(have)&need), tot)
