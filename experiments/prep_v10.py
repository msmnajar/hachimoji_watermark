"""Revision 10 cohort preparation on the laptop (numpy + PIL only).
BraTS: middle axial T1c slice (index shape[2]//2, on-disk order as nibabel get_fdata) -> float32 .npy
NIH / ISIC: gray level, LANCZOS resize to 512x512 -> PNG (as cache_features.load_set)
Usage: python3 prep_v10.py <datasets_root> <out_dir> <list> [<list> ...]"""
import gzip, os, sys
import numpy as np

DT = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32}


def nifti_mid(path):
    b = gzip.open(path, "rb").read()
    h = b[:348]
    end = "<" if int.from_bytes(h[:4], "little") == 348 else ">"
    dim = np.frombuffer(h[40:56], end + "i2")
    dtype = np.dtype(DT[int(np.frombuffer(h[70:72], end + "i2")[0])]).newbyteorder(end)
    vox = int(np.frombuffer(h[108:112], end + "f4")[0])
    slope, inter = np.frombuffer(h[112:120], end + "f4")
    nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
    a = np.frombuffer(b, dtype, count=nx * ny * nz, offset=vox).reshape((nx, ny, nz), order="F")
    s = a[:, :, nz // 2].astype(np.float64)
    if slope not in (0, 1) or inter != 0:
        if slope != 0:
            s = s * slope + inter
    return s.astype(np.float32)


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    from PIL import Image
    for lst in sys.argv[3:]:
        for rel in [l.strip() for l in open(lst) if l.strip()]:
            name = os.path.basename(rel)
            if rel.endswith(".nii.gz"):
                o = os.path.join(out, "brats", name.replace("-t1c.nii.gz", ".npy"))
                if not os.path.exists(o):
                    os.makedirs(os.path.dirname(o), exist_ok=True); np.save(o, nifti_mid(os.path.join(root, rel)))
            else:
                sub = "nih" if "NIH" in rel else "isic"
                o = os.path.join(out, sub, name.rsplit(".", 1)[0] + ".png")
                if not os.path.exists(o):
                    os.makedirs(os.path.dirname(o), exist_ok=True)
                    Image.open(os.path.join(root, rel)).convert("L").resize((512, 512), Image.LANCZOS).save(o)
        print("done", lst, flush=True)
