# experiments/summarize.py  (v3): prints the numbers used in the paper from a results file
import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
cols = ['JPEG_90','JPEG_50','JPEG_30','Gaussian_10','Gaussian_20','SaltPepper_1%','Median_3','Rotation_5','Rotation_10','Crop_10%','HistEq','Downsample_x2']
print('collection', ' '.join(cols))
for k in ['E1_sipi','E2_brats','E5_nih','E5_isic']:
    print(k, len(R[k]), ' '.join('%.4f' % np.mean([r['attacks'][c] for r in R[k]]) for c in cols))
b = R['E2_brats']
print('--- BraTS mean/min')
for c in b[0]['attacks']:
    v = [r['attacks'][c] for r in b]; print('%-22s %.4f %.4f' % (c, np.mean(v), np.min(v)))
print('--- unseen, all sets')
for k in ['E1_sipi','E2_brats','E5_nih','E5_isic']:
    print(k, {c: round(float(np.mean([r['attacks'][c] for r in R[k]])), 4) for c in R[k][0]['attacks'] if c.startswith('Unseen')})
print('--- ROI/timing BraTS')
for f in ['roi_pixel_fraction','roi_block_fraction','ops_saving_pct','saving_postwarm_pct','t_dual_postwarm_ms','t_uniform_postwarm_ms','t_total_dual_ms','t_total_uniform_ms','n_full','saving_pct']:
    v = [r[f] for r in b]; print(f, '%.3f +- %.3f' % (np.mean(v), np.std(v)))
for k in ['E1_sipi','E5_nih','E5_isic']:
    d = R[k]; print(k, 'ops', round(np.mean([r['ops_saving_pct'] for r in d]), 2), 'blk', round(np.mean([r['roi_block_fraction'] for r in d]), 3), 't', round(np.mean([r['t_total_dual_ms'] for r in d]), 1))
for r in R['E1_sipi']:
    a = r['attacks']; print(r['name'], a['JPEG_50'], a['Gaussian_20'], a['Median_3'], a['Rotation_10'], r['t_total_dual_ms'])
print('--- TCIA', len(R['E3_tcia']), 'excluded', R.get('E3_excluded_blank'))
for c in R['E3_tcia'][0]['attacks']:
    v = [t['attacks'][c] for t in R['E3_tcia']]; print('%-16s %.4f [%.4f, %.4f] <0.6: %d' % (c, np.mean(v), min(v), max(v), sum(x < 0.6 for x in v)))
K = R['E3_keytests']
for c in K[0]:
    v = [t[c][0] for t in K]; print('%-26s %.4f [%.4f, %.4f] <0.6: %d/%d' % (c, np.mean(v), min(v), max(v), sum(x < 0.6 for x in v), len(v)))
print('clean', set(t['nc_clean'] for t in R['E3_tcia']))
d = np.array(R['E6_distinguish_brats']); print('--- distinguish', len(d), '%.4f +- %.4f [%.4f, %.4f]' % (d.mean(), d.std(), d.min(), d.max()), 'n>=0.95:', int((d >= 0.95).sum()), 'n>=0.90:', int((d >= 0.90).sum()))
print('JPEG50 min', min(r['attacks']['JPEG_50'] for r in b))
V = R['E4_volumes']; print('--- volumes', len(V), 'reg ms', np.mean([v['reg_ms_per_slice'] for v in V]))
for s in V[0]['scenarios']:
    row = []
    for t in (0.90, 0.95):
        al = sum(sum(1 for x in v['scenarios'][s]['nc'] if x < t) for v in V)
        vols = sum(1 for v in V if any(x < t for x in v['scenarios'][s]['nc']))
        s10 = sum(1 for v in V if len(v['scenarios'][s]['nc']) > 10 and v['scenarios'][s]['nc'][10] < t)
        row.append('t=%.2f alarms %d vols %d slice10 %d' % (t, al, vols, s10))
    print('%-22s' % s, ' | '.join(row), 'fragile', sorted(set(v['scenarios'][s]['fragile_first'] for v in V), key=str))
for s in ['replace_other_patient','duplicate_neighbour','copy_move_48px','local_smooth_48px','replace_grey']:
    print(s, 'slice10 NC mean', round(np.mean([v['scenarios'][s]['nc'][10] for v in V]), 4), 'max', max(v['scenarios'][s]['nc'][10] for v in V))
sw = [v['scenarios']['swap_5_12']['nc'] for v in V]; print('swap flagged positions @0.95', sorted(set(i for n in sw for i, x in enumerate(n) if x < 0.95)))
print('config', R.get('config'))
