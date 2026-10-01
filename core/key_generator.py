# core/key_generator.py  (v2)
"""
Patient-bound key derivation.

v1:  master = SHA-256(PatientID|StudyUID|Modality|SeriesDate)  -> no secret;
     anyone holding the DICOM header could recompute every key.
v2:  master = HMAC-SHA256(K_inst, PatientID|StudyUID|Modality|SeriesDate)
     where K_inst is a 256-bit institutional secret held by the trusted
     authority that also stores the MWKs. Sub-keys are HMAC(master, domain).
"""
import hashlib
import hmac

FIELDS = ["PatientID", "StudyInstanceUID", "Modality", "SeriesDate"]

# Demonstration institutional key for the experiments (fixed for
# reproducibility). In deployment this is generated with a CSPRNG and
# never leaves the trusted authority.
DEFAULT_K_INST = hashlib.sha256(b"AASTMT-thesis-demo-institutional-key-v2").digest()


class PatientKey:
    def __init__(self, metadata: dict, k_inst: bytes = DEFAULT_K_INST):
        self.metadata = metadata
        payload = "|".join(str(metadata.get(f, "UNKNOWN")) for f in FIELDS).encode("utf-8")
        self.master = hmac.new(k_inst, payload, hashlib.sha256).digest()

    def sub(self, domain: str) -> bytes:
        return hmac.new(self.master, domain.encode(), hashlib.sha256).digest()

    def arnold_params(self):
        k = self.sub("arnold")
        a = int.from_bytes(k[0:4], "big") % 97 + 1
        b = int.from_bytes(k[4:8], "big") % 97 + 1
        n = int.from_bytes(k[8:12], "big") % 20 + 5
        return a, b, n
