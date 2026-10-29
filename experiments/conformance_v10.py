# experiments/conformance_v10.py  (revision 10)
"""Conformance of the evaluated pipeline (v10_decision.Coder) with
   (T1) the theoretical bounded-distance condition, and
   (T2) an INDEPENDENT reference implementation of the complete pipeline (ref_v10: Berlekamp-Welch RS,
        galois BCH, separate symbols/erasures/reconstruction/content extraction/HMAC check), and
   (T3) exhaustive nearest-code-word search for GF(4)/GF(8) blocks (subset of attempts).
Also checks: registration (helper data) identical between pipeline and reference; GRS model of the
reedsolo codes (encodings of random messages).
Modes:
  grid  : stratified sample of grid configurations on REVISION-9 TEST data (seed-0 cache) - pre-freeze evidence
  final : every configuration selected in calib_selection_seed{0,1,2}.json, on CALIBRATION data - run after
          calibration and before freezing (protocol step 5)
Usage: python experiments/conformance_v10.py grid <out.json> [n_registrants]
       python experiments/conformance_v10.py final <out_dir> <out.json>
"""
import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import v10_decision as VD                                              # noqa: E402
import ref_v10 as RF                                                   # noqa: E402
from core.key_generator import PatientKey                              # noqa: E402
from eval_cached import meta_2d                                        # noqa: E402

BCH63 = [(36, 5), (30, 6), (24, 7), (18, 10), (16, 11)]
BCH255 = [(131, 18), (115, 21), (99, 23), (87, 26), (71, 29)]
HRS64_K = (16, 20, 24, 28, 32, 36, 40, 44)


def grid_configs():
    out = []
    for q in (0.6, 0.7, 0.8, 0.9, 1.0):
        L = int(round(q * 512))
        for il in (False, True):
            ns = min(63, L // 6)
            for k in HRS64_K:
                if ns - k >= 2:
                    for s in range(0, ns - k + 1, 4):
                        out.append(dict(fam="HRS64", q=q, il=il, m=6, nblk=1, nc=ns, k=k, s=s))
            nb = (L // 3) // 7
            for k in (1, 2, 3, 4):
                for s in range(0, 7 - k):
                    out.append(dict(fam="HRS8", q=q, il=il, m=3, nblk=nb, nc=7, k=k, s=s))
            nb = (L // 2) // 3
            for k in (1, 2):
                for s in range(0, 3 - k):
                    out.append(dict(fam="DNA4", q=q, il=il, m=2, nblk=nb, nc=3, k=k, s=s))
            for k, t in BCH63:
                for s in (0, 2, 4, 6):
                    if s <= 2 * t:
                        out.append(dict(fam="BCH", q=q, il=il, m=1, nblk=L // 63, nc=63, k=k, s=s, t=t))
            for k, t in BCH255:
                nb = 1 if L <= 255 else 2; Lb = min(255, L // nb)
                for s in (0, 4, 8, 12, 16):
                    out.append(dict(fam="BCH", q=q, il=il, m=1, nblk=nb, nc=Lb, k=k - (255 - Lb), s=s, t=t))
    for c in out:
        c["msg_bits"] = c["nblk"] * c["k"] * c["m"]
    return [c for c in out if c["k"] > 0]


def feasible(c, T):
    return c["msg_bits"] - T >= 64


def stratified(seed=0, per_cell=3):
    rng = random.Random(seed); G = grid_configs(); pick = []
    for T in (32, 64, 128):
        for fam in ("HRS64", "HRS8", "DNA4", "BCH"):
            for ss in (False, True):                                    # with / without erasures
                cand = [c for c in G if c["fam"] == fam and feasible(c, T) and (c["s"] > 0) == ss]
                if fam == "BCH":                                        # both BCH lengths
                    for long in (False, True):
                        cc = [c for c in cand if (c["nc"] > 63) == long]
                        pick += [(dict(c), T) for c in rng.sample(cc, min(per_cell, len(cc)))]
                else:
                    pick += [(dict(c), T) for c in rng.sample(cand, min(per_cell, len(cand)))]
    return pick


def grs_selftest():
    import creedsolo
    res = {}
    for m, n, K in [(6, 63, 20), (6, 59, 28), (6, 51, 16), (6, 63, 44), (3, 7, 1), (3, 7, 4), (2, 3, 1), (2, 3, 2)]:
        g = RF.GRS(m, n, K); C = creedsolo.RSCodec(nsym=n - K, nsize=2 ** m - 1, c_exp=m, prim=RF.PRIM[m], generator=2)
        ok = 0
        for _ in range(30):
            msg = list(np.random.randint(0, 2 ** m, K)); cw = list(C.encode(bytearray(msg)))
            # corrupt within radius and decode with Berlekamp-Welch
            r = list(cw); e = (n - K) // 2
            for p in np.random.choice(n, e, replace=False):
                r[p] ^= int(np.random.randint(1, 2 ** m))
            ok += g.decode(r, set()) == cw
        res[f"m{m}_n{n}_K{K}"] = ok / 30
    return res


def interleave_test(n=400, seed=0):
    """Regression test for the creedsolo global-table hazard: codecs of GF(4), GF(8), GF(64) used in random
    interleaved order; every decode compared with the pure-Python reedsolo result."""
    import reedsolo as pyrs
    rng = np.random.default_rng(seed); codes = [(2, 3, 1), (3, 7, 3), (6, 63, 20), (6, 59, 28), (3, 7, 1)]
    bad = 0
    for _ in range(n):
        m, nc, K = codes[rng.integers(len(codes))]
        VD._rs(m, nc - K); cd = VD._CODECS[("rs", m, nc - K)]
        VD.activate_field(m)
        msg = bytearray(rng.integers(0, 2 ** m, K).tolist()); cw = bytearray(cd.encode(msg))
        r = bytearray(cw)
        for p in rng.choice(nc, (nc - K) // 2, replace=False):
            r[p] ^= int(rng.integers(1, 2 ** m))
        py = pyrs.RSCodec(nsym=nc - K, nsize=2 ** m - 1, c_exp=m, prim=VD.PRIM[m], generator=2)
        try:
            if bytes(py.encode(msg)) != bytes(cw):
                bad += 1; continue
        except Exception:
            bad += 1; continue
        try:
            a = bytes(cd.decode(bytearray(r))[1])
        except Exception:
            a = None
        try:
            b = bytes(py.decode(bytearray(r))[1])
        except Exception:
            b = None
        bad += a != b
    return dict(n=n, mismatches=bad)


def run_config(cfg, T, Z0, Za, names, n_reg, brute_regs, rs):
    st = dict(n_attempts=0, T1_mismatch=0, T2_mismatch=0, T3_mismatch=0, T3_n=0, helper_mismatch=0,
              status={}, ref_seconds=0.0, pipe_seconds=0.0)
    for i in range(n_reg):
        key = PatientKey(meta_2d(names[i])).sub("hrs")
        cd = VD.Coder(key, cfg, T)
        content = list(rs.randint(0, 2, cd.content_bits))
        rec = cd.register(Z0[i], content)
        ref = RF.RefBCH(key, cfg, T) if cfg["m"] == 1 else RF.RefRS(key, cfg, T)
        rrec = ref.register(Z0[i], content)
        st["helper_mismatch"] += int(not np.array_equal(np.asarray(rrec["helper"], np.int64), rec["helper"]))
        Zall = np.concatenate([Za[i], np.delete(Z0, i, 0)])
        a = time.time(); cond, status = cd.verify_batch(Zall, rec); st["pipe_seconds"] += time.time() - a
        a = time.time(); refres = ref.verify_many(Zall, rrec) if cfg["m"] == 1 else None; st["ref_seconds"] += time.time() - a
        for p in range(len(Zall)):
            a = time.time(); ok, got = refres[p] if refres is not None else ref.verify(Zall[p], rrec); st["ref_seconds"] += time.time() - a
            acc = status[p] in ("accept_genuine", "accept_other")
            st["T1_mismatch"] += int(bool(cond[p]) != (status[p] == "accept_genuine"))
            ref_kind = "reject" if not ok else ("accept_genuine" if got == content else "accept_other")
            pipe_kind = status[p] if acc else "reject"
            st["T2_mismatch"] += int(ref_kind != pipe_kind)
            st["status"][status[p]] = st["status"].get(status[p], 0) + 1
            if cfg["m"] in (2, 3) and i < brute_regs and p % 7 == 0:
                okb, gotb = ref.verify(Zall[p], rrec, brute=True)
                st["T3_mismatch"] += int((okb, gotb) != (ok, got)); st["T3_n"] += 1
        st["n_attempts"] += len(Zall)
    return st


N_RANDOM = {"RS": 10000, "BCH": 2000}       # random impostor pairs per configuration (BCH: galois reference cost)


def final_config(F, cfg, T):
    z, names, gi, za = F["z"], F["names"], F["gen_idx"], F["z_att"]
    N = len(z); rs = np.random.RandomState(1); gpos = {int(g): k for k, g in enumerate(gi)}
    nr = N_RANDOM["BCH" if cfg["m"] == 1 else "RS"]
    pick = sorted(set(map(tuple, np.random.RandomState(0).randint(0, N, (3 * nr, 2)).tolist())))
    pick = [p for p in pick if p[0] != p[1]]
    pick = [pick[i] for i in sorted(np.random.RandomState(0).permutation(len(pick))[:nr])]
    want = {}
    for i, j in pick:
        want.setdefault(i, set()).add(j)
    st = dict(n_attempts=0, T1_mismatch=0, T2_mismatch=0, T3_mismatch=0, T3_n=0, helper_mismatch=0, status={},
              n_pipeline_pairs=0)
    pend = []
    for i in range(N):
        key = PatientKey(meta_2d(str(names[i]))).sub("hrs")
        cd = VD.Coder(key, cfg, T); content = list(rs.randint(0, 2, cd.content_bits)); rec = cd.register(z[i], content)
        ref = RF.RefBCH(key, cfg, T) if cfg["m"] == 1 else RF.RefRS(key, cfg, T); rrec = ref.register(z[i], content)
        st["helper_mismatch"] += int(not np.array_equal(np.asarray(rrec["helper"], np.int64), rec["helper"]))
        others = [j for j in range(N) if j != i]
        cond, status = cd.verify_batch(z[others], rec); st["n_pipeline_pairs"] += len(others)
        tests = [(z[j], cond[k], status[k]) for k, j in enumerate(others)
                 if j in want.get(i, ()) or status[k] != "reject_fail"]
        if i in gpos:
            c2, s2 = cd.verify_batch(za[gpos[i]], rec); tests += list(zip(za[gpos[i]], c2, s2))
        if cfg["m"] == 1:                                   # BCH: decode later in one global galois batch
            for zz, cnd, stt in tests:
                r, er = ref.prepare(zz, rrec); pend.append((ref, rrec, content, r, er, cnd, stt))
            continue
        for ti, (zz, cnd, stt) in enumerate(tests):
            ok, got = ref.verify(zz, rrec)
            st["T1_mismatch"] += int(bool(cnd) != (stt == "accept_genuine"))
            rk = "reject" if not ok else ("accept_genuine" if got == content else "accept_other")
            pk = stt if stt in ("accept_genuine", "accept_other") else "reject"
            st["T2_mismatch"] += int(rk != pk); st["status"][stt] = st["status"].get(stt, 0) + 1
            if cfg["m"] in (2, 3) and st["T3_n"] < 300:
                st["T3_mismatch"] += int(ref.verify(zz, rrec, brute=True) != (ok, got)); st["T3_n"] += 1
            st["n_attempts"] += 1
    if pend:
        r0 = pend[0][0]
        dec = RF.bch_decode_global(r0.code, r0.t, r0.s, r0.nblk, r0.nc, r0.K,
                                   np.stack([p[3] for p in pend]), np.stack([p[4] for p in pend]))
        for (ref, rrec, content, r, er, cnd, stt), msg in zip(pend, dec):
            ok, got = ref.finish(msg, rrec)
            st["T1_mismatch"] += int(bool(cnd) != (stt == "accept_genuine"))
            rk = "reject" if not ok else ("accept_genuine" if got == content else "accept_other")
            pk = stt if stt in ("accept_genuine", "accept_other") else "reject"
            st["T2_mismatch"] += int(rk != pk); st["status"][stt] = st["status"].get(stt, 0) + 1
            st["n_attempts"] += 1
    return st


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "grid":
        out = sys.argv[2]; n_reg = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        from eval_test_decoders import z_of, STD, CLIN
        C = np.load("/home/claude/runs/cache/seed0_test.npy", allow_pickle=True).item()
        if os.path.exists(out):                         # resume: keep finished configurations
            res = json.load(open(out))
        else:
            res = dict(grs_selftest=grs_selftest(), interleave=interleave_test(), configs=[])
        print("GRS self-test", res["grs_selftest"], "interleave", res["interleave"], flush=True)
        done = {json.dumps([c["set"], c["cfg"], c["T"]], sort_keys=True) for c in res["configs"]}
        sets = ["brats100", "nih100", "isic100"]
        for n, (cfg, T) in enumerate(stratified()):
            s = sets[n % 3]; E = C["sets"][s]
            if json.dumps([s, cfg, T], sort_keys=True) in done:
                continue
            cl = C["conditions"][1:]; idx = [cl.index(x) for x in STD + CLIN]
            Z0 = np.stack([z_of(e)[0] for e in E]); Za = np.stack([z_of(e)[1:] for e in E])[:, idx]
            st = run_config(cfg, T, Z0, Za, [e["name"] for e in E], n_reg, 2, np.random.RandomState(n))
            res["configs"].append(dict(set=s, cfg=cfg, T=T, **st))
            print(s, cfg["fam"], {k: cfg[k] for k in ("q", "il", "nblk", "nc", "k", "s")}, "T", T,
                  {k: st[k] for k in ("n_attempts", "T1_mismatch", "T2_mismatch", "T3_n", "T3_mismatch", "helper_mismatch")},
                  "ref %.1fs" % st["ref_seconds"], flush=True)
            json.dump(res, open(out, "w"), indent=1)
    elif mode == "final":
        # every selected configuration of the three calibration selections, on CALIBRATION data:
        #   all genuine attempts; 10,000 random impostor pairs (numpy RandomState(0)); and every impostor pair
        #   whose pipeline status is not reject_fail (all pairs are run through the pipeline first).
        out_dir, out = sys.argv[2], sys.argv[3]
        seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]     # shards (same tests)
        if os.path.exists(out):                                                 # resume after interruption
            res = json.load(open(out))
        else:
            res = dict(grs_selftest=grs_selftest(), interleave=interleave_test(), configs=[])
        seen = {json.dumps([c["seed"], c["modality"], c["cfg"], c["T"]], sort_keys=True) for c in res["configs"]}
        for sd in seeds:
            S = json.load(open(os.path.join(out_dir, f"calib_selection_seed{sd}.json")))
            rev = os.environ.get("REVERSE") == "1"          # second worker walks the same list from the end
            mods = list(S["selection"].items())
            for mod, sel in (reversed(mods) if rev else mods):
                F = dict(np.load(os.path.join(out_dir, f"calib_features_seed{sd}_{mod}.npz")))
                items = list(sel.items())
                for lab, item in (reversed(items) if rev else items):
                    if not isinstance(item, dict) or "cfg" not in item:
                        continue
                    key = json.dumps([sd, mod, item["cfg"], item["T"]], sort_keys=True)
                    for other in glob.glob(os.path.join(os.path.dirname(out), "conformance_final_*.json")):
                        if other != out:                     # skip configurations another worker finished
                            try:
                                seen |= {json.dumps([c["seed"], c["modality"], c["cfg"], c["T"]], sort_keys=True)
                                         for c in json.load(open(other))["configs"]}
                            except (ValueError, OSError):
                                pass
                    if key in seen:
                        continue
                    seen.add(key)
                    st = final_config(F, item["cfg"], item["T"])
                    res["configs"].append(dict(seed=sd, modality=mod, label=lab, cfg=item["cfg"], T=item["T"], **st))
                    print(sd, mod, lab, {k: st[k] for k in ("n_attempts", "T1_mismatch", "T2_mismatch", "T3_mismatch",
                                                            "helper_mismatch")}, flush=True)
                    json.dump(res, open(out, "w"), indent=1)
