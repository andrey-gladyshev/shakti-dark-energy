"""Cosmological models for Section 4 of the paper.

Models:
  - LCDM: E^2 = Om*(1+z)^3 + Or*(1+z)^4 + (1-Om-Or)
  - SHAKTI (pure): alpha_0 = 1, x_0 = 0.1510, one free param (Omega_m)
  - SHAKTI (free): alpha_0 free, x_0 = 0.1510, two free params
  - CPL: w(z) = w0 + wa z/(1+z), three free params

For SHAKTI, E(N) is obtained by solving
    x'' + 3x' + alpha0^2 x / E^2 = 0,    N = ln a
with x(0) = x_0. The ODE solution is cached per (Omega_m, alpha_0).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d

OM_R = 9e-5
X0_FIXED = 0.1510
ALPHA0_PURE = 1.0
N_RANGE = (-14.0, 0.0)

_ODE_CACHE = {}


def E_lcdm(z, om):
    """E(z) for flat LCDM with radiation."""
    z = np.asarray(z, dtype=float)
    return np.sqrt(om * (1.0 + z) ** 3 + OM_R * (1.0 + z) ** 4 + (1.0 - om - OM_R))


def _solve_shakti(om, alpha0, x0=X0_FIXED):
    """Solve the SHAKTI ODE once and cache the E(N) interpolator."""
    key = (round(om, 8), round(alpha0, 8), round(x0, 8))
    if key in _ODE_CACHE:
        return _ODE_CACHE[key]

    lam = (1.0 - om - OM_R) / x0  # lambda = Omega_DE(0) / x0

    def rhs(N, y):
        x, xp = y
        E2 = om * np.exp(-3 * N) + OM_R * np.exp(-4 * N) + lam * x
        E2 = max(E2, 1e-12)
        return [xp, -3.0 * xp - (alpha0 ** 2 / E2) * x]

    def integrate(xi):
        return solve_ivp(rhs, N_RANGE, [xi, 0.0], method='DOP853',
                         rtol=1e-8, atol=1e-10, dense_output=True)

    def root(xi):
        return integrate(xi).y[0, -1] - x0

    try:
        xi = brentq(root, 0.03, 0.40, xtol=1e-10, maxiter=200)
    except ValueError:
        _ODE_CACHE[key] = None
        return None

    sol = integrate(xi)
    N_grid = np.linspace(N_RANGE[0], N_RANGE[1], 4000)
    x_vals = sol.sol(N_grid)[0]
    E2_vals = (om * np.exp(-3 * N_grid)
               + OM_R * np.exp(-4 * N_grid)
               + lam * x_vals)
    E2_interp = interp1d(N_grid, E2_vals, kind='cubic',
                         fill_value='extrapolate')
    _ODE_CACHE[key] = (E2_interp, lam, xi)
    return _ODE_CACHE[key]


def E_shakti(z, om, alpha0=ALPHA0_PURE):
    """E(z) for SHAKTI model. z can be scalar or array."""
    z_in = np.asarray(z, dtype=float)
    scalar = z_in.ndim == 0
    N_arr = -np.log(1.0 + np.atleast_1d(z_in))
    cached = _solve_shakti(om, alpha0)
    if cached is None:
        raise RuntimeError(f"SHAKTI ODE not solvable at om={om}, alpha0={alpha0}")
    E2_interp, _, _ = cached
    E2 = E2_interp(N_arr)
    E2 = np.clip(E2, 1e-12, None)
    E = np.sqrt(E2)
    return float(E[0]) if scalar else E


def E_cpl(z, om, w0, wa):
    """E(z) for CPL: w(a) = w0 + wa (1-a). Radiation included."""
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    # rho_DE(a) / rho_DE(0) = a^{-3(1+w0+wa)} * exp(-3 wa (1-a))
    de = a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    om_de0 = 1.0 - om - OM_R
    E2 = om * (1.0 + z) ** 3 + OM_R * (1.0 + z) ** 4 + om_de0 * de
    return np.sqrt(np.clip(E2, 1e-12, None))


if __name__ == "__main__":
    print("Sanity checks:")
    print(f"  E_lcdm(0, 0.30)   = {E_lcdm(0.0, 0.30):.6f}  (expect 1.0)")
    print(f"  E_lcdm(1, 0.30)   = {E_lcdm(1.0, 0.30):.6f}  (expect ~1.6)")
    print(f"  E_cpl(0, 0.30, -1, 0) = {E_cpl(0.0, 0.30, -1.0, 0.0):.6f}  (expect 1.0)")

    print("  Computing SHAKTI pure (om=0.31, alpha0=1) ...")
    E0 = E_shakti(0.0, 0.31, 1.0)[0]
    E1 = E_shakti(1.0, 0.31, 1.0)[0]
    print(f"  E_shakti(0, 0.31, 1) = {E0:.6f}  (expect 1.0)")
    print(f"  E_shakti(1, 0.31, 1) = {E1:.6f}  (expect ~1.6)")
    cached = _solve_shakti(0.31, 1.0)
    if cached:
        _, lam, xi = cached
        print(f"  lam = {lam:.6f}, xi = {xi:.6f}")
