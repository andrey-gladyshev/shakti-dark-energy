"""Fits of cosmological models to DESI DR2 BAO + Pantheon+ SNe.

Section 4 of the paper. Step 1: Lambda-CDM joint fit.

Free parameter: Omega_m.
Analytically marginalized:
  alpha = rs*H0/c   (in BAO, enters as amplitude 1/alpha)
  M                 (SNe absolute magnitude, additive offset)
"""
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
C_KMS = 299792.458
C_OVER_H0 = C_KMS / 100.0  # Mpc, with h = 1

Z_CUT_SN = 0.01


# ---------------- BAO ----------------

def load_bao():
    mean_path = DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_path  = DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_cov.txt"

    z_list, val_list, qty_list = [], [], []
    with open(mean_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            z_list.append(float(parts[0]))
            val_list.append(float(parts[1]))
            qty_list.append(parts[2])

    return (np.array(z_list), np.array(val_list),
            np.array(qty_list), np.loadtxt(cov_path))


def E(z, om):
    return np.sqrt(om * (1.0 + z) ** 3 + (1.0 - om))


def comoving(z, om):
    return quad(lambda zp: 1.0 / E(zp, om), 0, z, epsabs=1e-10, epsrel=1e-10)[0]


def D_bao_single(zi, qi, om):
    """The alpha-independent part of the BAO model at one redshift."""
    Ei = E(zi, om)
    if qi == "DV_over_rs":
        Ii = comoving(zi, om)
        return (zi * Ii * Ii / Ei) ** (1.0 / 3.0)
    if qi == "DM_over_rs":
        return comoving(zi, om)
    if qi == "DH_over_rs":
        return 1.0 / Ei
    raise ValueError(f"Unknown quantity: {qi}")


def chi2_bao_profiled(om, z, qty, obs, cov_inv):
    """chi^2_BAO after analytic marginalization over alpha.

    model = D / alpha.  Optimal alpha solves dchi2/d(1/alpha) = 0.
    """
    D = np.array([D_bao_single(zi, qi, om) for zi, qi in zip(z, qty)])
    CD = cov_inv @ D
    Co = cov_inv @ obs
    a = float(D @ Co)
    b = float(D @ CD)
    if b <= 0.0 or a <= 0.0:
        return 1e12, np.nan
    alpha = b / a
    r = D / alpha - obs
    return float(r @ cov_inv @ r), alpha


# ---------------- Pantheon+ SNe ----------------

_PANTHEON_CACHE = {}


def load_pantheon():
    if "data" in _PANTHEON_CACHE:
        return _PANTHEON_CACHE["data"]

    dat = np.genfromtxt(DATA_DIR / "Pantheon+SH0ES.dat",
                        names=True, dtype=None, encoding="utf-8")
    z_all = dat["zHD"].astype(float)
    m_all = dat["m_b_corr"].astype(float)

    C_all = np.loadtxt(DATA_DIR / "Pantheon+SH0ES_STAT+SYS.cov")[1:]
    C_all = C_all.reshape(1701, 1701)

    mask = z_all > Z_CUT_SN
    z   = z_all[mask]
    m_b = m_all[mask]
    C   = C_all[np.ix_(mask, mask)]

    _PANTHEON_CACHE["data"] = (z, m_b, C)
    return z, m_b, C


def mu_lcdm(z, om):
    """Distance modulus for flat LCDM with h = 1 (Mpc)."""
    Dc = np.array([comoving(zi, om) for zi in z])
    dL = C_OVER_H0 * (1.0 + z) * Dc
    return 5.0 * np.log10(dL) + 25.0


def chi2_sn_profiled(om, z, m_b, C_inv, ones):
    """chi^2_SNe after analytic marginalization over M.

    Delta = m_b - mu_model
    chi^2 = Delta^T C^-1 Delta - (1^T C^-1 Delta)^2 / (1^T C^-1 1)
    """
    mu = mu_lcdm(z, om)
    Delta = m_b - mu
    CD = C_inv @ Delta
    S_C1 = float(ones @ (C_inv @ ones))
    S_CD = float(ones @ CD)
    return float(Delta @ CD - S_CD ** 2 / S_C1)


# ---------------- Joint fit ----------------

def fit_lcdm():
    z_bao, obs_bao, qty_bao, cov_bao = load_bao()
    cov_bao_inv = np.linalg.inv(cov_bao)

    print("Loading Pantheon+ ...")
    z_sn, m_sn, C_sn = load_pantheon()
    print(f"  SNe (z > {Z_CUT_SN}): {len(z_sn)}")

    print("Inverting SNe covariance (1590 x 1590) ...")
    C_sn_inv = np.linalg.inv(C_sn)
    ones_sn = np.ones(len(z_sn))

    def chi2_total(om):
        chi2_b, _ = chi2_bao_profiled(om, z_bao, qty_bao, obs_bao, cov_bao_inv)
        chi2_s = chi2_sn_profiled(om, z_sn, m_sn, C_sn_inv, ones_sn)
        return chi2_b + chi2_s

    res = minimize_scalar(chi2_total, bounds=(0.15, 0.55), method="bounded",
                          options={"xatol": 1e-7})
    om = res.x
    chi2_val = res.fun
    chi2_b, alpha = chi2_bao_profiled(om, z_bao, qty_bao, obs_bao, cov_bao_inv)
    chi2_s = chi2_sn_profiled(om, z_sn, m_sn, C_sn_inv, ones_sn)

    n = len(z_bao) + len(z_sn)
    k = 1
    aic = chi2_val + 2 * k
    bic = chi2_val + k * np.log(n)

    print("=" * 60)
    print("LCDM joint fit: DESI DR2 BAO + Pantheon+ SNe")
    print("=" * 60)
    print(f"  Omega_m           = {om:.6f}")
    print(f"  alpha = rs*H0/c   = {alpha:.6f}")
    print(f"  rs*H0 (km/s)      = {alpha * C_KMS:.2f}")
    print(f"  N (data points)   = {n}   (BAO {len(z_bao)} + SNe {len(z_sn)})")
    print(f"  chi^2 (BAO)       = {chi2_b:.4f}")
    print(f"  chi^2 (SNe)       = {chi2_s:.4f}")
    print(f"  chi^2 (total)     = {chi2_val:.4f}")
    print(f"  k                 = {k}")
    print(f"  AIC               = {aic:.4f}")
    print(f"  BIC               = {bic:.4f}")
    print()
    print("  Reference (Table 4):")
    print("    LCDM: k=1, chi^2 = 1424.79, AIC = 1426.79, BIC = 1432.17")
    print("=" * 60)


if __name__ == "__main__":
    run_table4()


# ---------------- Generalized chi^2 for any model ----------------

def _E_wrapper(model, om, extra=()):
    """Return callable E(z) for a given model."""
    from . import models as M
    if model == "lcdm":
        return lambda z: M.E_lcdm(z, om)
    if model == "shakti_pure":
        return lambda z: M.E_shakti(z, om, alpha0=M.ALPHA0_PURE)
    if model == "shakti_free":
        alpha0 = extra[0]
        return lambda z: M.E_shakti(z, om, alpha0=alpha0)
    if model == "cpl":
        w0, wa = extra
        return lambda z: M.E_cpl(z, om, w0, wa)
    raise ValueError(f"Unknown model: {model}")


def _comoving_from_E(E_func, z):
    """Comoving distance int_0^z dz'/E(z') for scalar z."""
    return quad(lambda zp: 1.0 / E_func(zp), 0, z,
                epsabs=1e-10, epsrel=1e-10)[0]


def chi2_bao_for_E(E_func, z_bao, obs_bao, qty_bao, cov_bao_inv):
    """chi^2_BAO after analytic marginalization over alpha."""
    D = np.zeros(len(z_bao))
    for i, (zi, qi) in enumerate(zip(z_bao, qty_bao)):
        Ei = E_func(zi)
        if qi == "DV_over_rs":
            Ii = _comoving_from_E(E_func, zi)
            D[i] = (zi * Ii * Ii / Ei) ** (1.0 / 3.0)
        elif qi == "DM_over_rs":
            D[i] = _comoving_from_E(E_func, zi)
        elif qi == "DH_over_rs":
            D[i] = 1.0 / Ei
    CD = cov_bao_inv @ D
    Co = cov_bao_inv @ obs_bao
    a = float(D @ Co)
    b = float(D @ CD)
    if b <= 0 or a <= 0:
        return 1e12
    alpha = b / a
    r = D / alpha - obs_bao
    return float(r @ cov_bao_inv @ r)


def chi2_sn_for_E(E_func, z_sn, m_sn, C_sn_inv, ones_sn):
    """chi^2_SNe after analytic marginalization over M."""
    Dc = np.array([_comoving_from_E(E_func, zi) for zi in z_sn])
    mu = 5.0 * np.log10(C_OVER_H0 * (1.0 + z_sn) * Dc) + 25.0
    Delta = m_sn - mu
    CD = C_sn_inv @ Delta
    S1 = float(ones_sn @ (C_sn_inv @ ones_sn))
    SD = float(ones_sn @ CD)
    return float(Delta @ CD - SD ** 2 / S1)


def chi2_total_for_E(E_func, z_bao, obs_bao, qty_bao, cov_bao_inv,
                     z_sn, m_sn, C_sn_inv, ones_sn):
    cb = chi2_bao_for_E(E_func, z_bao, obs_bao, qty_bao, cov_bao_inv)
    cs = chi2_sn_for_E(E_func, z_sn, m_sn, C_sn_inv, ones_sn)
    return cb + cs


# ---------------- Table 4: all four models ----------------

def _load_all_data():
    z_bao, obs_bao, qty_bao, cov_bao = load_bao()
    cov_bao_inv = np.linalg.inv(cov_bao)
    z_sn, m_sn, C_sn = load_pantheon()
    C_sn_inv = np.linalg.inv(C_sn)
    ones_sn = np.ones(len(z_sn))
    return (z_bao, obs_bao, qty_bao, cov_bao_inv,
            z_sn, m_sn, C_sn_inv, ones_sn)


def _chi2_model(params, model, data):
    (z_bao, obs_bao, qty_bao, cov_bao_inv,
     z_sn, m_sn, C_sn_inv, ones_sn) = data
    om = params[0]
    extra = params[1:]
    try:
        E_func = _E_wrapper(model, om, extra)
        return chi2_total_for_E(E_func, z_bao, obs_bao, qty_bao, cov_bao_inv,
                                z_sn, m_sn, C_sn_inv, ones_sn)
    except Exception:
        return 1e12


def run_table4():
    """Reproduce Table 4 of the paper."""
    from scipy.optimize import minimize

    print("Loading data ...")
    data = _load_all_data()
    n = len(data[0]) + len(data[4])
    ln_n = np.log(n)
    print(f"  N = {n}   ln(N) = {ln_n:.4f}")
    print()

    # Format: (name, k, initial, bounds, model_id)
    configs = [
        ("Lambda-CDM",    1, [0.30],                [(0.15, 0.55)],                  "lcdm"),
        ("SHAKTI pure",   1, [0.30],                [(0.15, 0.55)],                  "shakti_pure"),
        ("SHAKTI free",   2, [0.30, 1.0],           [(0.15, 0.55), (0.1, 3.0)],      "shakti_free"),
        ("CPL",           3, [0.30, -0.95, -0.10],  [(0.15, 0.55), (-2.0, -0.3), (-3.0, 3.0)], "cpl"),
    ]

    results = []
    for name, k, x0, bounds, model_id in configs:
        print(f"Fitting {name} (k={k}) ...")
        res = minimize(_chi2_model, x0, args=(model_id, data),
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 3000})
        chi2_val = res.fun
        aic = chi2_val + 2 * k
        bic = chi2_val + k * ln_n
        results.append((name, k, chi2_val, aic, bic, res.x))
        print(f"  chi^2 = {chi2_val:.4f}  AIC = {aic:.4f}  BIC = {bic:.4f}")
        print(f"  params = {res.x}")

    print()
    print("=" * 78)
    print(f"{'Model':<14} {'k':>3} {'chi^2':>12} {'AIC':>12} {'BIC':>12}")
    print("-" * 78)
    for name, k, c2, aic, bic, _ in results:
        print(f"{name:<14} {k:>3} {c2:>12.4f} {aic:>12.4f} {bic:>12.4f}")
    print("=" * 78)
    print()
    print("Reference (Table 4 of the paper):")
    print("  LCDM         k=1  chi^2=1424.79  AIC=1426.79  BIC=1432.17")
    print("  SHAKTI pure  k=1  chi^2=1412.77  AIC=1414.77  BIC=1420.15")
    print("  SHAKTI free  k=2  chi^2=1412.27  AIC=1416.27  BIC=1427.03")
    print("  CPL          k=3  chi^2=1412.72  AIC=1418.72  BIC=1434.86")


