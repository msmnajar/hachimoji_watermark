# core/hachimoji_dna.py  (v2)
"""
Hachimoji eight-base coding (Hoshika et al., Science 2019).

Bases and pairing: A-T, G-C, B-S (isoguanine : isocytosine/1-methylcytosine),
P-Z. (v1 used B-Z and P-S and described P/S as NaM/TPT1, which belong to a
different unnatural base pair.)

Tribits t in 0..7 are mapped to bases; complementary tribits (t, 7-t) must map
to complementary bases, giving 4! * 2^4 = 384 admissible rules.
Base indices: A0 T1 G2 C3 B4 Z5 P6 S7. Every complementary pair has opposite
index parity (A0-T1, G2-C3, B4-S7, P6-Z5), a property the zero-watermark uses.
"""
from itertools import permutations
import numpy as np

BASES = ["A", "T", "G", "C", "B", "Z", "P", "S"]
IDX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G", "B": "S", "S": "B", "P": "Z", "Z": "P"}
BASE_PAIRS = [("A", "T"), ("G", "C"), ("B", "S"), ("P", "Z")]
TRIBIT_PAIRS = [(0, 7), (1, 6), (2, 5), (3, 4)]


def build_rules():
    rules = []
    for order in permutations(BASE_PAIRS):
        for flip in range(16):
            r = np.zeros(8, dtype=np.int64)
            for k, (lo, hi) in enumerate(TRIBIT_PAIRS):
                b1, b2 = order[k]
                if (flip >> k) & 1:
                    b1, b2 = b2, b1
                r[lo], r[hi] = IDX[b1], IDX[b2]
            for lo, hi in TRIBIT_PAIRS:
                assert COMPLEMENT[BASES[r[lo]]] == BASES[r[hi]]
            rules.append(r)
    return np.stack(rules)


RULES = build_rules()          # (384, 8)
N_RULES = RULES.shape[0]
assert N_RULES == 384
assert all((i % 2) != (IDX[COMPLEMENT[b]] % 2) for b, i in IDX.items())


def encode(tribits: np.ndarray, rule_idx: np.ndarray) -> np.ndarray:
    """Vectorised: base index for each tribit under its own rule."""
    return RULES[rule_idx, tribits]


# Hachimoji DNA operations on base indices (mod 8), after Wu et al. 2024
def op_add(x, k): return (x + k) % 8
def op_sub(x, k): return (x - k) % 8
def op_xor(x, k): return np.bitwise_xor(x, k) % 8
def op_inv(x):    return 7 - x          # bitwise inversion of the tribit
