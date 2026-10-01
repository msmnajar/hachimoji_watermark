# watermarking/volumetric.py  (v2)
"""
Hash-chained volumetric verification (v2).

Robust layer: slice n is registered with a seed
    s_n = SHA-256(MWK_{n-1} || K_patient || n),   MWK_0 := K_patient,
so each slice record is bound to its position and to the previous record.
A slice passes if the extracted watermark reaches NC >= tau.

Fragile layer (new in v2): H_n = SHA-256(H_{n-1} || SHA-256(pixels_n)),
H_0 = K_patient. Any pixel change, including benign re-compression,
breaks H from that slice on; the robust layer then tells whether the
content is still the registered content.

v1 used only the robust layer with tau = 0.999, which also flags benign
JPEG re-compression (NC 0.9988 at Q50) as tampering.
"""
import hashlib
import numpy as np
from watermarking.zero_watermark import ZeroWatermarkV2, nc, otsu_roi


def slice_seed(prev_mwk: bytes, k_patient: bytes, n: int) -> bytes:
    return hashlib.sha256(prev_mwk + k_patient + n.to_bytes(4, "big")).digest()


class VolumeChain:
    def __init__(self, k_patient: bytes, arnold_params):
        self.k = k_patient
        self.ap = arnold_params

    def register(self, vol, wm):
        recs, prev, Hc = [], self.k, self.k
        fragile = []
        for n in range(vol.shape[2]):
            sl = vol[:, :, n]
            zw = ZeroWatermarkV2(slice_seed(prev, self.k, n), self.ap)
            r = zw.register(sl, wm, otsu_roi(sl))
            recs.append(r)
            prev = r["mwk"].tobytes()
            Hc = hashlib.sha256(Hc + hashlib.sha256(np.ascontiguousarray(sl).tobytes()).digest()).digest()
            fragile.append(Hc)
        return recs, fragile

    def verify(self, vol, wm, recs, fragile, tau):
        prev, Hc = self.k, self.k
        ncs, frag_first = [], None
        n_slices = min(vol.shape[2], len(recs))
        for n in range(n_slices):
            sl = vol[:, :, n]
            zw = ZeroWatermarkV2(slice_seed(prev, self.k, n), self.ap)
            ncs.append(nc(wm, zw.extract(sl, recs[n])))
            prev = recs[n]["mwk"].tobytes()
            Hc = hashlib.sha256(Hc + hashlib.sha256(np.ascontiguousarray(sl).tobytes()).digest()).digest()
            if frag_first is None and Hc != fragile[n]:
                frag_first = n
        flagged = [i for i, v in enumerate(ncs) if v < tau]
        if vol.shape[2] != len(recs):
            frag_first = frag_first if frag_first is not None else n_slices
        return dict(nc=ncs, robust_flagged=flagged, fragile_first=frag_first,
                    count_mismatch=vol.shape[2] != len(recs))
