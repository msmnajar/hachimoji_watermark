# Post-run verification of one seed's U results (read-only; not part of the frozen pipeline).
import json, sys, numpy as np
s = int(sys.argv[1]); R = '/home/claude/runs/v10'
res = json.load(open(f'{R}/U_results_seed{s}.json')); FZ = json.load(open(f'{R}/frozen_v10.json'))[str(s)]
M = json.load(open(f'{R}/manifest_v10.json'))
fail = []; summary = {}
print('freeze record in output:', res['freeze_record_sha256'][:12])
for mod in ('brats', 'nih', 'isic'):
    if mod not in res['modalities']:
        fail.append(f'{mod}: missing'); continue
    out = res['modalities'][mod]; N = len(M['modalities'][mod]['untouched']); Mp = N * (N - 1)
    pats = [x['patient'] for x in M['modalities'][mod]['untouched']]
    _, cnt = np.unique(pats, return_counts=True); same = int((cnt * (cnt - 1)).sum())
    for lab, item in FZ[mod]['codes'].items():
        r = out.get(lab)
        if item == 'none':
            if r != 'none': fail.append(f'{mod} {lab}: expected none'); continue
            continue
        if not isinstance(r, dict): fail.append(f'{mod} {lab}: missing'); continue
        if r['cfg'] != item['cfg'] or r['T'] != item['T']: fail.append(f'{mod} {lab}: config differs from frozen')
        if r['N'] != N or r['M'] != Mp or sum(r['counts'].values()) != Mp: fail.append(f'{mod} {lab}: pair count')
        st = r['stats']
        if st['same_patient_pairs'] != same: fail.append(f'{mod} {lab}: same-patient pair count {st["same_patient_pairs"]} != {same}')
        acc = r['counts'].get('accept_genuine', 0) + r['counts'].get('accept_other', 0)
        if r['accepted_draw_pairs_incl_same_patient'] != acc: fail.append(f'{mod} {lab}: accepted draw pairs inconsistent')
        if r['FA'] != st['FA'] or st['FA'] + st['same_patient_accepted'] != acc: fail.append(f'{mod} {lab}: FA inconsistent')
        if abs(st['p_hat'] - st['FA'] / Mp) > 1e-15: fail.append(f'{mod} {lab}: p_hat')
        if r['FA_a'] + r['FA_b'] != acc: fail.append(f'{mod} {lab}: mechanisms')
        gs = r['genuine_status']; ng = sum(gs.values())
        if ng != 200 * 24: fail.append(f'{mod} {lab}: genuine attempts {ng}')
        t = r['test_set_rev9']
        if t['M'] != 9900 or sum(t['counts'].values()) != 9900: fail.append(f'{mod} {lab}: test-set pair count')
    for op, v in out['NC'].items():
        if v['M'] != Mp: fail.append(f'{mod} NC {op}: M')
    summary[mod] = dict(N=N, M=Mp, same_patient_pairs=same, configs=sum(1 for v in FZ[mod]['codes'].values() if v != 'none'))
print(json.dumps(summary)); print('FAILURES:', fail if fail else 'none')
sys.exit(1 if fail else 0)
