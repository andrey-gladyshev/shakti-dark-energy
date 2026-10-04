"""Effective Omega_g diagnostic for the SHAKTI model.

Compares SHAKTI best-fit E(z) with Lambda-CDM at the same Omega_m
and computes the effective 'cosmic glitch' parameter Omega_g_eff(z).

Main finding (for Section 4 Discussion):
  - CGG (Hergt et al. 2026): Omega_g = -0.0081 (NEGATIVE, 3.3 sigma)
  - SHAKTI: E^2_SHAKTI > E^2_LCDM at z ~ 0.3-2, corresponding to
    POSITIVE Omega_g_eff(z) in CGG convention.

Physical interpretation:
  CGG    -> weaker cosmological gravity, LESS effective DE at high z
  SHAKTI -> frozen Sigma field, MORE DE at high z

The two models predict opposite-sign deviations from LCDM in E^2(z),
distinguishable by future BAO/SNe measurements at 0.3 < z < 2.

Key result from BAO+SNe fit:
  LCDM:         chi^2 = 1416.81  k=1
  CGG:          chi^2 = 1416.79  k=2   (degenerate with LCDM)
  SHAKTI pure:  chi^2 = 1412.49  k=1   (Delta = -4.3)

CGG is degenerate with LCDM on BAO+SNe alone (requires CMB to
break the degeneracy). SHAKTI gives a distinguishable improvement.
"""
import numpy as np
from .models import E_shakti, E_lcdm, _solve_shakti

OM = 0.30280
OM_R = 9e-5


def E_shakti_scalar(z):
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    N = -np.log(1.0 + z_arr)
    E2_interp, _, _ = _solve_shakti(OM, 1.0)
    E2 = np.clip(E2_interp(N), 1e-12, None)
    return float(np.sqrt(E2)[0])


def E_cgg(z, om, omega_g):
    """CGG: Omega_g = 1 - G_N / G_cosmo (Hergt et al. 2026)."""
    z = np.asarray(z, dtype=float)
    om_de = 1.0 - omega_g - om - OM_R
    num = om * (1 + z) ** 3 + OM_R * (1 + z) ** 4 + om_de
    return np.sqrt(np.clip(num / (1.0 - omega_g), 1e-12, None))


def omega_g_eff(z):
    """Effective Omega_g in the CGG convention:
       Omega_g_eff(z) = (E^2_SHAKTI - E^2_LCDM) /
                        (E^2_SHAKTI - Omega_m (1+z)^3 - Omega_r (1+z)^4)
    """
    E_s2 = E_shakti_scalar(z) ** 2
    E_l2 = float(E_lcdm(np.array([z]), OM)[0]) ** 2
    denom = E_s2 - OM * (1 + z) ** 3 - OM_R * (1 + z) ** 4
    if abs(denom) < 1e-12:
        return 0.0
    return (E_s2 - E_l2) / denom


def main():
    print("SHAKTI vs CGG - effective Omega_g diagnostic")
    print("=" * 60)
    print(f"  SHAKTI best-fit Omega_m = {OM}")
    print()
    print(f"{'z':>8} {'E^2_SHA/E^2_LCDM':>18} {'Omega_g_eff':>15}")
    print("-" * 43)
    for z in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        E_s2 = E_shakti_scalar(z) ** 2
        E_l2 = float(E_lcdm(np.array([z]), OM)[0]) ** 2
        ratio = E_s2 / E_l2
        og = omega_g_eff(z)
        print(f"{z:>8.2f} {ratio:>18.5f} {og:>15.6f}")
    print()
    print("Interpretation:")
    print("  ratio > 1        => SHAKTI has MORE expansion than LCDM")
    print("  omega_g_eff > 0  => OPPOSITE SIGN to CGG (-0.0081)")
    print()
    print("Key BAO+SNe fit result:")
    print("  LCDM:        chi^2 = 1416.81  k=1")
    print("  CGG:         chi^2 = 1416.79  k=2  (degenerate with LCDM)")
    print("  SHAKTI pure: chi^2 = 1412.49  k=1  (Delta = -4.3)")


if __name__ == "__main__":
    main()
