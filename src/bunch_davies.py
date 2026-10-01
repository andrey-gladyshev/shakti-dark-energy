"""Bunch–Davies amplitude δφ/H for a scalar field in de Sitter space."""

import numpy as np
from scipy.special import gamma


def nu_of_m(m, H=1.0):
    """Spectral parameter ν = sqrt(9/4 - m²/H²)."""
    val = 9 / 4 - (m / H) ** 2
    return np.sqrt(val) if val >= 0 else None


def delta_phi_over_H(m_phi, H=1.0):
    """Dimensionless Bunch–Davies amplitude δφ/H."""
    nu = nu_of_m(m_phi, H)
    if nu is None:
        raise ValueError("m > 3H/2: no valid ν")
    factor = (
        np.abs(gamma(nu)) ** 2
        / (2 ** (2 * nu - 3) * gamma(1.5) ** 2)
    )
    return np.sqrt(factor) / (2 * np.pi)


def x_i_analytic(m_phi, H=1.0):
    """Analytic approximation x_i ≈ 3ν₀/(8π)."""
    nu = nu_of_m(m_phi, H)
    return 3 * nu / (8 * np.pi)


if __name__ == "__main__":
    dphi = delta_phi_over_H(0.5)
    print(f"δφ/H (m_φ = H/2) = {dphi:.6f}")
    print(f"x_i (analytic)    = {x_i_analytic(0.5):.6f}")
