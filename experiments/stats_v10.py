# experiments/stats_v10.py  (revision 10, draft for protocol review)
"""False-acceptance-rate (FAR) inference under patient-level dependence.

Data: N patients (i.i.d. draws from the population), every ordered pair (i, j), i != j, verified
once: h_ij = 1 if the image of j is accepted under the record of i.  FA = sum h_ij,
M = N (N - 1),  p_hat = FA / M  (an unbiased U-statistic of order 2 for the population FAR
p = P(h_AB = 1), A != B random patients).

  M1  distribution-free bound (primary).  Patients X_1..X_N i.i.d.; directed kernel h(X_i, X_j) in {0,1};
      symmetrized kernel g(x, y) = (h(x, y) + h(y, x)) / 2 in [0, 1], E g = p.  Then
          p_hat = sum_{i != j} h_ij / (N(N-1)) = sum_{i < j} g_ij / C(N, 2)   (a U-statistic of order 2).
      Permutation-averaging device (Hoeffding 1963; no section/equation number is cited because the full
      text could not be checked): U = (1/N!) sum_pi W_pi, W_pi = (1/k) sum_{r=1..k} g(X_pi(2r-1), X_pi(2r)),
      k = floor(N/2); each W_pi is a mean of k i.i.d. [0,1] variables (disjoint pairs of i.i.d. patients),
      so by Jensen E exp(-sU) <= E exp(-s W) <= (1 - p + p e^{-s/k})^k (independence and convexity of e^{-lx},
      l = s/k, on [0,1]), and Markov's inequality, optimized over l = s/k > 0, gives
          P(U <= p - t) <= exp(-k KL(p - t || p)),  0 < t < p.
      One-sided (1 - alpha) upper bound for the observed u = p_hat:
          p_U(u) = max{ p >= u : k KL(u || p) <= ln(1/alpha) }.
      Coverage: p_U(U) < p  <=>  U < x*(p) with k KL(x* || p) = ln(1/alpha), and P(U <= x*) <= alpha.
      Dependence between decisions sharing a patient is irrelevant: only the i.i.d. patients enter W_pi.
      With FA = 0: p_U = 1 - alpha^(1/k).
  M2  patient-cluster bootstrap (secondary, approximate): patients are resampled with replacement
      (weights w ~ Multinomial(N, 1/N)); every ordered pair of draws of different patients is kept, so
      FA* = sum_{(i,j) in FA} w_i w_j,  M* = (sum w)^2 - sum w^2,  p* = FA*/M*.  Percentile 95% interval
      and one-sided 95% upper bound; degenerate (identically 0) when FA = 0.
  M3  delete-one-patient jackknife variance of p_hat and the implied effective sample size
      n_eff = p_hat (1 - p_hat) / Var_jack  (undefined when FA = 0).
  Coverage of M1 and M2 is checked by simulation (coverage_sim) under a two-way multiplicative
  patient random-effects model, P(h_ij = 1) = p0 u_i u_j, u ~ Gamma(kappa, 1/kappa).
"""
import math

import numpy as np


def kl(a, b):
    a = min(max(a, 0.0), 1.0)
    out = 0.0
    if a > 0:
        out += a * math.log(a / b)
    if a < 1:
        out += (1 - a) * math.log((1 - a) / (1 - b))
    return out


def m1_upper(fa, N, alpha=0.05):
    n2 = N // 2; ph = fa / (N * (N - 1)); target = math.log(1 / alpha) / n2
    lo, hi = ph, 1 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2
        if kl(ph, mid) > target:
            hi = mid
        else:
            lo = mid
    return hi


def m2_bootstrap(pairs, N, B=2000, seed=0, alpha=0.05):
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    P = np.empty(B)
    for b in range(B):
        w = rng.multinomial(N, np.full(N, 1.0 / N)).astype(float)
        M = w.sum() ** 2 - (w ** 2).sum()
        P[b] = (w[pairs[:, 0]] * w[pairs[:, 1]]).sum() / M if len(pairs) else 0.0
    return dict(ci=[float(np.quantile(P, alpha / 2)), float(np.quantile(P, 1 - alpha / 2))],
                upper=float(np.quantile(P, 1 - alpha)), degenerate=bool(len(pairs) == 0))


def m3_jackknife(pairs, N):
    pairs = np.asarray(pairs, int).reshape(-1, 2); fa = len(pairs)
    if fa == 0:
        return dict(var=0.0, n_eff=None)
    inv = np.bincount(pairs.ravel(), minlength=N)              # pairs involving each patient
    pl = (fa - inv) / ((N - 1) * (N - 2))
    var = (N - 1) / N * ((pl - pl.mean()) ** 2).sum()
    ph = fa / (N * (N - 1))
    return dict(var=float(var), n_eff=float(ph * (1 - ph) / var) if var > 0 else None)


def disjoint_check(pairs, N, seed=0):
    """One fixed random perfect matching (seed 0): floor(N/2) patient-disjoint ordered pairs (a -> b);
    exact Clopper-Pearson one-sided 95% upper bound (independent pairs)."""
    from scipy.stats import beta
    perm = np.random.default_rng(seed).permutation(N); k = N // 2
    S = set(map(tuple, np.asarray(pairs, int).reshape(-1, 2).tolist()))
    x = sum((int(perm[2 * r]), int(perm[2 * r + 1])) in S for r in range(k))
    up = float(beta.ppf(0.95, x + 1, k - x)) if x < k else 1.0
    return dict(pairs=k, FA=x, cp_upper95=up)


def far_report(pairs, N, alpha=0.05, B=2000, report_m2=True, patients=None, design=None):
    """pairs: accepted ordered pairs (i, j) of DRAWS i != j.
    patients: patient id of every draw (needed when the same patient can be drawn twice);
    design: dict(kind="iid draws with replacement" | "census of all remaining eligible patients", population_size=P).

    Kernel h'(x, y) = h(x, y) if the two draws are different patients, else 0 (pairs of the same patient are
    never impostor attempts; they are counted separately and reported).
      * i.i.d. draws with replacement from a finite population of P patients (NIH U): the draws are exactly
        i.i.d., so M1 is a rigorous one-sided 95% bound on p' = E h' = (1 - 1/P) p, and
        p <= M1 / (1 - 1/P) is a rigorous bound on the impostor FAR p of that finite population.
      * census (BraTS U, ISIC U): FA / M is the EXACT false-acceptance rate among all ordered pairs of the
        remaining eligible patients of the release (no sampling error).  M1 is reported only as a
        bound for a hypothetical superpopulation from which the release would be an i.i.d. sample; that
        model is an assumption, not a property of the design.
    M2 is reported only when FA >= 3 (pre-registered)."""
    P = np.asarray(pairs, int).reshape(-1, 2)
    same = np.zeros(len(P), bool)
    if patients is not None and len(P):
        pa = np.asarray(patients)
        same = pa[P[:, 0]] == pa[P[:, 1]]
    Pd = P[~same]; fa = len(Pd); M = N * (N - 1)
    n_same_pairs = 0
    if patients is not None:
        _, cnt = np.unique(np.asarray(patients), return_counts=True)
        n_same_pairs = int((cnt * (cnt - 1)).sum())
    out = dict(FA=fa, M=M, N=N, same_patient_pairs=n_same_pairs, same_patient_accepted=int(same.sum()),
               p_hat=fa / M, FA_rate_among_distinct_patient_pairs=fa / (M - n_same_pairs),
               M1_upper=m1_upper(fa, N, alpha),
               M2=m2_bootstrap(Pd, N, B) if (report_m2 and fa >= 3) else "not reported (FA < 3)",
               M3=m3_jackknife(Pd, N), disjoint=disjoint_check(Pd, N),
               patients_involved_descriptive=int(len(set(Pd.ravel().tolist()))) if fa else 0)
    if design:
        out["design"] = design
        Ppop = design.get("population_size")
        if design["kind"].startswith("iid"):
            out["M1_upper_finite_population"] = out["M1_upper"] / (1 - 1 / Ppop)
            out["interpretation"] = ("rigorous one-sided 95% bound on the impostor FAR of the finite eligible "
                                     "population (i.i.d. draws with replacement)")
        else:
            out["exact_finite_population_FAR"] = out["FA_rate_among_distinct_patient_pairs"]
            out["interpretation"] = ("census: exact FAR of the remaining eligible patients of the release; "
                                     "M1 holds only under the superpopulation (i.i.d.) model assumption")
    return out


def simulate(N, p0, kappa, rng):
    u = rng.gamma(kappa, 1.0 / kappa, N) if kappa else np.ones(N)
    lam = p0 * (u.sum() ** 2 - (u ** 2).sum())                # expected FA (Poisson approximation)
    k = rng.poisson(lam)
    if k == 0:
        return np.zeros((0, 2), int), p0
    pr = u / u.sum(); out = []
    while len(out) < k:
        i, j = rng.choice(N, 2, p=pr)
        if i != j:
            out.append((i, j))
    return np.array(out), p0                                    # population FAR = p0 (E u = 1)


def coverage_sim(N=1000, p0s=(1e-5, 1e-4, 3e-4), kappas=(0, 1.0, 0.2, 0.05), R=300, B=300, seed=0):
    rng = np.random.default_rng(seed); res = []
    for p0 in p0s:
        for ka in kappas:
            c1 = c2 = 0
            for _ in range(R):
                pairs, p = simulate(N, p0, ka, rng)
                c1 += m1_upper(len(pairs), N) >= p
                c2 += m2_bootstrap(pairs, N, B, seed=int(rng.integers(1 << 30)))["upper"] >= p
            res.append(dict(N=N, p0=p0, kappa=ka, R=R, cover_M1=c1 / R, cover_M2_upper95=c2 / R))
            print(res[-1], flush=True)
    return res


if __name__ == "__main__":
    import json, sys, time
    t = time.time()
    rng = np.random.default_rng(1)
    demo = far_report(simulate(1000, 1e-4, 0.2, rng)[0], 1000)
    print("demo", demo, "%.1f s" % (time.time() - t))
    print("M1 bound with FA = 0: N=1000 %.2e, N=842 %.2e, N=300 %.2e" % (m1_upper(0, 1000), m1_upper(0, 842), m1_upper(0, 300)))
    t = time.time(); cov = coverage_sim()
    json.dump(dict(demo=demo, coverage=cov, seconds=time.time() - t), open(sys.argv[1], "w"), indent=1)
