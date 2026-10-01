# experiments/run_all_v2.py
"""Re-runs every experiment of the paper with the v2 implementation.
Usage: python experiments/run_all_v2.py <data_root> <out_dir>"""
import sys, os, glob, json, time, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from PIL import Image
from core.key_generator import PatientKey, DEFAULT_K_INST
from core.chaotic_system import seed_from_digest
from watermarking.zero_watermark import ZeroWatermarkV2, otsu_roi, nc, ber
from watermarking.volumetric import VolumeChain
from evaluation.attacks import STANDARD_ATTACKS, jpeg_compression, pacs_png_conversion, \
    telemedicine_transcoding, gaussian_noise

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
WM = np.random.RandomState(42).randint(0, 2, (64, 64)).astype(np.uint8)


def to_u8(a):
    a = a.astype(np.float64)
    return ((a - a.min()) / (a.max() - a.min() + 1e-8) * 255).astype(np.uint8)


def uid(s):  # deterministic (v1 used Python hash(), which is salted per run)
    return "2.25." + str(int(hashlib.sha256(s.encode()).hexdigest()[:30], 16))


def zw_for(meta, k_inst=DEFAULT_K_INST, domain="pixel"):
    k = PatientKey(meta, k_inst)
    return ZeroWatermarkV2(k.sub(domain), k.arnold_params()), k


def attack_battery(img, zw, rec, extra=None):
    out = {}
    rs = np.random.RandomState(7)
    atk = dict(STANDARD_ATTACKS)
    if extra:
        atk.update(extra)
    for name, fn in atk.items():
        np.random.seed(rs.randint(1 << 30))
        try:
            out[name] = round(nc(WM, zw.extract(fn(img), rec)), 4)
        except Exception as e:
            out[name] = None
    return out


MED2D = {"PACS_PNG": pacs_png_conversion,
         "PACS_JPEG95": lambda im: jpeg_compression(im, 95),
         "Downsample_x2": lambda im: telemedicine_transcoding(im, 2)}


# ------------------------------------------------------------ 2D cases
def run_2d(args):
    name, img, meta = args
    zw, k = zw_for(meta)
    roi = otsu_roi(img)
    rec = zw.register(img, WM, roi)
    # timing: dual vs uniform full strength (median of 5)
    rd = [zw.register(img, WM, roi) for _ in range(5)]
    ru = [zw.register(img, WM, roi, uniform_full=True) for _ in range(5)]
    td = np.median([r["t_keystream_ms"] for r in rd]); tu = np.median([r["t_keystream_ms"] for r in ru])
    tdp = np.median([r["t_keystream_ms"] - r["t_warm_ms"] for r in rd])
    tup = np.median([r["t_keystream_ms"] - r["t_warm_ms"] for r in ru])
    ttd = np.median([r["t_total_ms"] for r in rd]); ttu = np.median([r["t_total_ms"] for r in ru])
    ext = zw.extract(img, rec)
    return dict(name=name, shape=list(img.shape), roi_pixel_fraction=round(float(roi.mean()), 4),
                roi_block_fraction=round(rec["roi_block_fraction"], 4),
                nc_clean=nc(WM, ext), ber_clean=ber(WM, ext),
                t_dual_ms=round(float(td), 2), t_uniform_ms=round(float(tu), 2),
                saving_pct=round(100 * (1 - td / tu), 2),
                t_dual_postwarm_ms=round(float(tdp), 2), t_uniform_postwarm_ms=round(float(tup), 2),
                saving_postwarm_pct=round(100 * (1 - tdp / tup), 2),
                t_total_dual_ms=round(float(ttd), 1), t_total_uniform_ms=round(float(ttu), 1),
                n_full=rec["n_full"], ops_saving_pct=round(100 * (1 - (4 * rec["n_full"] + (4096 - rec["n_full"])) / (4 * 4096)), 2),
                attacks=attack_battery(img, zw, rec, MED2D)), (name, img, rec, meta)


def load_brats(n=20):
    fs = sorted(glob.glob(f"{ROOT}/brats2023/*/*/*t1c.nii.gz"))[:n]
    import nibabel as nib
    out = []
    for f in fs:
        d = nib.load(f).get_fdata()
        case = os.path.basename(os.path.dirname(f))
        out.append((case, to_u8(d[:, :, d.shape[2] // 2]),
                    dict(PatientID=case, StudyInstanceUID=uid(case), Modality="MR", SeriesDate="UNKNOWN")))
    return out, fs


def load_folder(pattern, n, prefix, resize=None):
    out = []
    for f in sorted(glob.glob(pattern))[:n]:
        im = Image.open(f).convert("L")
        if resize:
            im = im.resize((resize, resize), Image.LANCZOS)
        nm = os.path.basename(f)
        out.append((nm, np.array(im, np.uint8),
                    dict(PatientID=prefix + nm, StudyInstanceUID=uid(nm), Modality="OT", SeriesDate="UNKNOWN")))
    return out


# ------------------------------------------------------------ DICOM
def run_dicom(f):
    import pydicom
    ds = pydicom.dcmread(f)
    raw = ds.pixel_array.astype(np.float64)
    img = to_u8(raw)
    meta = {k: str(getattr(ds, k, "UNKNOWN")) for k in ["PatientID", "StudyInstanceUID", "Modality", "SeriesDate"]}
    zw, k = zw_for(meta)
    rec = zw.register(img, WM, otsu_roi(img))
    res = dict(file=os.path.relpath(f, ROOT), patient=meta["PatientID"], study=meta["StudyInstanceUID"][-12:],
               nc_clean=nc(WM, zw.extract(img, rec)), ber_clean=ber(WM, zw.extract(img, rec)))
    # display-window changes applied to the raw DICOM values
    lo, hi = np.percentile(raw, [1, 99]); c, w = (lo + hi) / 2, max(hi - lo, 1.0)
    def win(center, width):
        width = max(width, 1.0)
        return (np.clip((raw - (center - width / 2)) / width, 0, 1) * 255).astype(np.uint8)
    wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
    atks = {"RLE_lossless": lambda: img.copy(), "PACS_PNG": lambda: pacs_png_conversion(img),
            "PACS_JPEG95": lambda: jpeg_compression(img, 95),
            "Downsample_x2": lambda: telemedicine_transcoding(img, 2),
            "Window_p1-p99": lambda: win(c, w), "Window_narrow50": lambda: win(c, 0.5 * w)}
    if wc is not None and ww is not None:
        wc0 = float(wc[0] if hasattr(wc, "__len__") else wc); ww0 = float(ww[0] if hasattr(ww, "__len__") else ww)
        atks["Window_header"] = lambda: win(wc0, ww0)
    res["attacks"] = {n: round(nc(WM, zw.extract(fn(), rec)), 4) for n, fn in atks.items()}
    return res, (meta, img, rec)


def key_tests(meta, img, rec, other_meta):
    zw, _ = zw_for(meta)
    out = {}
    def t(m, kinst=DEFAULT_K_INST):
        z, _ = zw_for(m, kinst)
        e = z.extract(img, rec); return round(nc(WM, e), 4), round(ber(WM, e), 4)
    out["other_patient"] = t(other_meta)
    out["same_patient_other_study"] = t(dict(meta, StudyInstanceUID=meta["StudyInstanceUID"] + ".1"))
    out["random_patient_id"] = t(dict(meta, PatientID="WRONG_PATIENT"))
    out["wrong_institution_key"] = t(meta, hashlib.sha256(b"another-hospital").digest())
    flipped = bytearray(DEFAULT_K_INST); flipped[0] ^= 1
    out["k_inst_1bit_flip"] = t(meta, bytes(flipped))
    # 1e-15 perturbation of the chaotic initial value x0
    k = PatientKey(meta)
    z = ZeroWatermarkV2(k.sub("pixel"), k.arnold_params())
    import core.chaotic_system as cs
    orig = cs.seed_from_digest
    def pert(d):
        v = orig(d); v["x0"] += 1e-15; return v
    import watermarking.zero_watermark as zwm
    zwm.seed_from_digest = pert
    e = z.extract(img, rec)
    zwm.seed_from_digest = orig
    out["x0_plus_1e-15"] = (round(nc(WM, e), 4), round(ber(WM, e), 4))
    return out


# ------------------------------------------------------------ volumes
def run_volume(args):
    f, n_slices, tau_list = args
    import nibabel as nib
    d = nib.load(f).get_fdata()
    idx = np.linspace(d.shape[2] // 4, 3 * d.shape[2] // 4, n_slices, dtype=int)
    vol = np.stack([to_u8(d[:, :, i]) for i in idx], 2)
    case = os.path.basename(os.path.dirname(f))
    k = PatientKey(dict(PatientID=case, StudyInstanceUID=uid(case), Modality="MR", SeriesDate="UNKNOWN"))
    ch = VolumeChain(k.master, k.arnold_params())
    t0 = time.perf_counter(); recs, frag = ch.register(vol, WM); t_reg = (time.perf_counter() - t0) * 1000
    scen = {}
    def run(name, v):
        r = ch.verify(v, WM, recs, frag, tau=0.0)
        scen[name] = dict(nc=[round(x, 4) for x in r["nc"]], fragile_first=r["fragile_first"])
    run("clean", vol)
    rs = np.random.RandomState(3)
    run("benign_png", np.stack([pacs_png_conversion(vol[:, :, i]) for i in range(n_slices)], 2))
    run("benign_jpeg90", np.stack([jpeg_compression(vol[:, :, i], 90) for i in range(n_slices)], 2))
    run("benign_jpeg50", np.stack([jpeg_compression(vol[:, :, i], 50) for i in range(n_slices)], 2))
    np.random.seed(5)
    run("benign_noise5", np.stack([gaussian_noise(vol[:, :, i], 5) for i in range(n_slices)], 2))
    t = 10
    v = vol.copy(); v[:, :, t] = 128; run("replace_grey", v)
    other = sorted(glob.glob(f"{ROOT}/brats2023/*/*/*t1c.nii.gz"))[-1]
    od = nib.load(other).get_fdata(); v = vol.copy(); v[:, :, t] = to_u8(od[:, :, idx[t]]); run("replace_other_patient", v)
    v = vol.copy(); v[:, :, t] = vol[:, :, t + 1]; run("duplicate_neighbour", v)
    v = vol.copy(); s = v[:, :, t].copy(); s[130:178, 130:178] = s[60:108, 60:108]; v[:, :, t] = s; run("copy_move_48px", v)
    v = vol.copy(); s = v[:, :, t].astype(float); from scipy.ndimage import uniform_filter
    s[96:144, 96:144] = uniform_filter(s, 15)[96:144, 96:144]; v[:, :, t] = s.astype(np.uint8); run("local_smooth_48px", v)
    v = vol.copy(); v[:, :, [5, 12]] = v[:, :, [12, 5]]; run("swap_5_12", v)
    v = np.delete(vol, t, axis=2); run("delete_slice", v)
    return dict(case=case, n_slices=n_slices, reg_ms_per_slice=round(t_reg / n_slices, 1), scenarios=scen)


if __name__ == "__main__":
    t_all = time.time()
    res = {}
    with Pool(2) as pool:
        # E1 USC-SIPI
        sipi = load_folder(f"{ROOT}/USC_SIPI/*.tif", 4, "SIPI_", 512)
        r = pool.map(run_2d, sipi); res["E1_sipi"] = [x[0] for x in r]; print("E1 done", time.time() - t_all, flush=True)
        # E2 BraTS
        brats, brats_files = load_brats(20)
        r = pool.map(run_2d, brats); res["E2_brats"] = [x[0] for x in r]; brats_recs = [x[1] for x in r]
        print("E2 done", time.time() - t_all, flush=True)
        # E5 NIH / ISIC
        nih = load_folder(f"{ROOT}/NIH_ChestXray14/images/*.png", 20, "NIH_", 512)
        r = pool.map(run_2d, nih); res["E5_nih"] = [x[0] for x in r]
        isic = load_folder(f"{ROOT}/ISIC2020/Training_JPEG/train/*.jpg", 20, "ISIC_", 512)
        r = pool.map(run_2d, isic); res["E5_isic"] = [x[0] for x in r]; print("E5 done", time.time() - t_all, flush=True)
        # E3 TCIA
        dfiles = sorted(glob.glob(f"{ROOT}/TCIA/RIDER_Breast_MRI/*/*.dcm"))
        r = pool.map(run_dicom, dfiles)
        res["E3_tcia"] = [x[0] for x in r]
        metas = [x[1] for x in r]
        keyres = []
        for i, (meta, img, rec) in enumerate(metas):
            other = next(m for (m, _, _) in metas if m["PatientID"] != meta["PatientID"])
            keyres.append(key_tests(meta, img, rec, other))
        res["E3_keytests"] = keyres; print("E3 done", time.time() - t_all, flush=True)
        # E6 distinguishability (BraTS): record of image i verified with image j (same key)
        dist = []
        for i, (ni, _, reci, meta) in enumerate(brats_recs):
            zw, _ = zw_for(meta)
            for j, (nj, imgj, _, _) in enumerate(brats_recs):
                if i != j:
                    dist.append(round(nc(WM, zw.extract(imgj, reci)), 4))
        res["E6_distinguish_brats"] = dist; print("E6 done", time.time() - t_all, flush=True)
        # E4 volumes (10 BraTS volumes x 20 slices)
        r = pool.map(run_volume, [(f, 20, None) for f in brats_files[:10]])
        res["E4_volumes"] = r; print("E4 done", time.time() - t_all, flush=True)
    json.dump(res, open(f"{OUT}/results_v2.json", "w"), indent=1)
    print("ALL DONE", time.time() - t_all)
