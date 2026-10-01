"""Mass hierarchy analysis for the Shakti dark energy model.

Section 8 of the paper: status of the mass chain
    m_Sigma = H,   m_phi = H/2,   m_Sigma^2 + m_phi^2 = 5/4 H^2

The chain is a phenomenological input. Twelve derivation
mechanisms from first principles were tested; none produced the
exact result. This script reproduces Eq. (18) and the summary
table (Table 7 of the paper).
"""
import numpy as np


M_SIGMA_OVER_H = 1.0    # m_Sigma = H
M_PHI_OVER_H   = 0.5    # m_phi = H/2


def nu(m_over_H):
    """nu = sqrt(9/4 - (m/H)^2) for a scalar of mass m in de Sitter."""
    val = 9.0/4.0 - m_over_H**2
    return np.sqrt(val) if val >= 0 else np.nan


def ratio_G_phi_G_Sigma():
    """Eq. (18): G_phi/G_Sigma = (3 - 2 nu_Sigma) / (3 - 2 nu_phi)."""
    nu_s = nu(M_SIGMA_OVER_H)    # sqrt(5)/2 ≈ 1.1180
    nu_p = nu(M_PHI_OVER_H)      # sqrt(2)   ≈ 1.4142
    return (3 - 2*nu_s) / (3 - 2*nu_p), nu_s, nu_p


MECHANISMS = [
    ("Modular invariance",        "x", "gives m_phi = m_Sigma, not H/2"),
    ("Vacuum energy",             "x", "3 orders of magnitude too small"),
    ("Plancherel measure",        "x", "<nu^2> = 0.32 != 13/4"),
    ("SO(4,1) tensor product",    "x", "conditions not satisfied"),
    ("Entanglement entropy",      "x", "independent of mass"),
    ("Symmetry nu -> 3/2 - nu",   "x", "does not preserve the chain"),
    ("KMS thermal mass",          "x", "does not give required masses"),
    ("Coupling types g^2 S^2 p^2","x", "G_phi/G_Sigma = 4.45 != 4"),
    ("alpha-vacuum",              "x", "increases the ratio"),
    ("One-loop potential",        "x", "ln 2 artefact"),
    ("RG flow in dS",             "x", "IR attractor xi = 1/6, not 1/48"),
    ("Trace anomaly",             "x", "does not generate the chain"),
]


def main():
    nu_s = nu(M_SIGMA_OVER_H)
    nu_p = nu(M_PHI_OVER_H)
    ratio, _, _ = ratio_G_phi_G_Sigma()
    deviation = 100.0 * abs(ratio - 4.0) / 4.0

    print("=" * 68)
    print("MASS CHAIN ANALYSIS (Section 8)")
    print("=" * 68)
    print(f"  m_Sigma / H = {M_SIGMA_OVER_H:.4f}")
    print(f"  m_phi   / H = {M_PHI_OVER_H:.4f}")
    print(f"  nu_Sigma    = {nu_s:.6f}   (= sqrt(5)/2)")
    print(f"  nu_phi      = {nu_p:.6f}   (= sqrt(2))")
    print()
    print(f"  Sum m_Sigma^2 + m_phi^2 = "
          f"{M_SIGMA_OVER_H**2 + M_PHI_OVER_H**2:.4f} H^2  (target 1.25 H^2)")
    print(f"  Sum nu_Sigma^2 + nu_phi^2 = {nu_s**2 + nu_p**2:.4f}  (= 13/4 = 3.25)")
    print()
    print(f"  Eq. (18): G_phi/G_Sigma = (3 - 2 nu_Sigma)/(3 - 2 nu_phi)")
    print(f"           = {ratio:.4f}")
    print(f"  Target    = 4.0000")
    print(f"  Deviation = {deviation:.1f}%")
    print()
    print("-" * 68)
    print("Twelve tested mechanisms (all closed):")
    print("-" * 68)
    for i, (name, status, note) in enumerate(MECHANISMS, 1):
        print(f"  {i:2d}. {name:30s} [{status}]  {note}")
    print()
    n_closed = sum(1 for _, st, _ in MECHANISMS if st == "x")
    print(f"  Closed: {n_closed}/12")
    print("=" * 68)


if __name__ == "__main__":
    main()
