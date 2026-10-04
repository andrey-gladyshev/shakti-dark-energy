"""Generate the six figures of the paper (Figure 1-6).

Figures 1: x_i and c as function of m_phi/H (the mass-chain postulate test)
Figures 2-4: background evolution w(z), H(z)/H0, Omega_DE(z)
Figure 5: Pantheon+ residuals
Figure 6: sky map (Mollweide) colored by residual
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import E_shakti, E_lcdm, E_cpl, ALPHA0_PURE

OUT_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Best-fit parameters from Table 4 reproduction
OM_PURE = 0.30280
OM_FREE = 0.30272
ALPHA0_FREE = 1.14259
OM_CPL = 0.30386
W0_CPL = -0.89581
WA_CPL = -0.18445
OM_LCDM = 0.30374

C_KMS = 299792.458

# Bunch-Davies amplitude: delta_phi/H for a scalar of mass m in dS
def delta_phi_over_H(m_over_H):
    nu2 = 9.0/4.0 - m_over_H**2
    if nu2 <= 0:
        return np.nan
    nu = np.sqrt(nu2)
    from scipy.special import gamma
    val = (abs(gamma(nu))**2) / (2.0 ** (2*nu - 3) * gamma(1.5)**2)
    return np.sqrt(val) / (2.0 * np.pi)


def figure1_mass_dependence():
    """Figure 1: delta_phi/H vs m_phi/H, and the fit x_i vs m_phi/H."""
    m_grid = np.linspace(0.30, 0.80, 200)
    delta = np.array([delta_phi_over_H(m) for m in m_grid])

    # ODE-fit x_i as function of m_phi/H: we use published Table 3 values
    # (2 sigma resolution: sampled at 6 points, cubic interpolation).
    m_tab = np.array([0.400, 0.450, 0.500, 0.527, 0.581, 0.650])
    xi_tab = np.array([0.168721, 0.168721, 0.168721, 0.168721, 0.168721, 0.168721])
    # In the paper Table 3, xi is fixed (fit to data), delta_phi/H varies.

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: x_i vs m_phi/H
    axes[0].plot(m_grid, delta, "b-", lw=2, label="Точная (Банч-Дэвис)")
    axes[0].axhline(0.168721, color="k", ls=":", lw=1.2,
                    label="фит $x_i = 0.168721$")
    axes[0].axvline(0.5, color="g", ls="--", lw=1.2, label="$m_\\varphi = H/2$")
    axes[0].axvline(0.581, color="orange", ls="--", lw=1.2,
                    label="$m_\\varphi = 0.58H$")
    axes[0].set_xlabel("$m_\\varphi / H$")
    axes[0].set_ylabel("$x_i$")
    axes[0].set_title("$x_i$ от $m_\\varphi/H$")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right: deviation of c = xi/(delta_phi/H) from 1
    c_vals = 0.168721 / delta
    dev = 100.0 * np.abs(c_vals - 1.0)
    axes[1].plot(m_grid, dev, "r--", lw=2, label="Отклонение $c$ от 1")
    axes[1].axvline(0.5, color="g", ls="--", lw=1.2, label="$m_\\varphi = H/2$")
    axes[1].axvline(0.581, color="orange", ls="--", lw=1.2,
                    label="$m_\\varphi = 0.58H$")
    axes[1].set_xlabel("$m_\\varphi / H$")
    axes[1].set_ylabel("Отклонение (%)")
    axes[1].set_title("Точность совпадения")
    axes[1].set_ylim(0, 10)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig1_mass_dependence.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")




# ---------------- Background functions ----------------

def _rho_de(z, E_func, om, om_r=9e-5):
    return E_func(z)**2 - om*(1+z)**3 - om_r*(1+z)**4


def _w_de(z, E_func, om):
    """Effective w_DE(z) extracted from E(z)."""
    h = 1e-4
    rho0 = _rho_de(z, E_func, om)
    rho_p = _rho_de(z+h, E_func, om)
    rho_m = _rho_de(max(z-h, 0.0), E_func, om)
    drho = (rho_p - rho_m) / (2*h)
    return -1.0 - (1.0 + z) * drho / (3.0 * rho0)


def _omega_de(z, E_func, om):
    return _rho_de(z, E_func, om) / E_func(z)**2


def figure2_wz():
    """Figure 2: w(z) for pure SHAKTI, CPL, LCDM."""
    z_grid = np.linspace(0.0, 2.0, 400)
    E_pure = lambda z: E_shakti(z, OM_PURE, alpha0=ALPHA0_PURE)
    E_cpl_f = lambda z: E_cpl(z, OM_CPL, W0_CPL, WA_CPL)

    w_pure = np.array([_w_de(z, E_pure, OM_PURE) for z in z_grid])
    # CPL parametrization for the plot: use paper's values w0=-0.9287, wa=-0.1065
    w0_paper, wa_paper = -0.9287, -0.1065
    w_cpl = w0_paper + wa_paper * z_grid / (1.0 + z_grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_grid, w_pure, "b-", lw=2, label="Чистая модель, $\\alpha_0 = 1$")
    ax.plot(z_grid, w_cpl, "r--", lw=2, label="CPL")
    ax.axhline(-1.0, color="k", ls=":", lw=1.2, label="$\\Lambda$CDM")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$w(z)$")
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.15, -0.85)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig2_wz.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


def figure3_Hz():
    """Figure 3: H(z)/H0 for pure SHAKTI vs LCDM."""
    z_grid = np.linspace(0.0, 2.0, 400)
    E_pure = np.array([E_shakti(z, OM_PURE, alpha0=ALPHA0_PURE) for z in z_grid])
    E_lcdm_v = E_lcdm(z_grid, OM_LCDM)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_grid, E_pure, "b-", lw=2, label="Чистая модель")
    ax.plot(z_grid, E_lcdm_v, "k:", lw=2, label="$\\Lambda$CDM")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$H(z)/H_0$")
    ax.set_xlim(0, 2)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig3_Hz.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


def figure4_omega_de():
    """Figure 4: Omega_DE(z) for pure SHAKTI vs LCDM."""
    z_grid = np.linspace(0.0, 20.0, 500)
    E_pure = lambda z: E_shakti(z, OM_PURE, alpha0=ALPHA0_PURE)
    om_pure = np.array([_omega_de(z, E_pure, OM_PURE) for z in z_grid])
    om_lcdm_val = 1.0 - OM_LCDM - 9e-5

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_grid, om_pure, "b-", lw=2, label="$\\Omega_{DE}(z)$")
    ax.axhline(om_lcdm_val, color="k", ls=":", lw=2, label="$\\Lambda$CDM")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\\Omega_{DE}(z)$")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig4_omega_de.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")



# ---------------- Figure 5: Pantheon+ residuals ----------------

def figure5_residuals():
    """Figure 5: mu_obs - mu_model for SHAKTI pure."""
    from .fits import load_pantheon
    from pathlib import Path as _P
    data_dir = _P(__file__).resolve().parent.parent / "data"
    dat = np.genfromtxt(data_dir / "Pantheon+SH0ES.dat",
                        names=True, dtype=None, encoding="utf-8")
    z_all = dat["zHD"].astype(float)
    m_all = dat["m_b_corr"].astype(float)
    err_all = dat["m_b_corr_err_DIAG"].astype(float)
    mask_z = z_all > 0.01
    z = z_all[mask_z]; m_b = m_all[mask_z]; err = err_all[mask_z]
    E_func = lambda zz: E_shakti(zz, OM_PURE, alpha0=ALPHA0_PURE)
    from scipy.integrate import quad
    Dc = np.array([quad(lambda zp: 1.0/E_func(zp), 0, zi,
                        epsabs=1e-10, epsrel=1e-10)[0] for zi in z])
    mu_model = 5.0*np.log10(C_KMS/100.0 * (1.0+z) * Dc) + 25.0
    res = m_b - mu_model
    # analytic offset
    w = 1.0/err**2
    offset = (w*res).sum()/w.sum()
    res_centered = res - offset
    # error bar: raw diagonal err (which is offset by ~M)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(z, res, yerr=err, fmt="o", ms=3, color="gray",
                ecolor="lightgray", elinewidth=0.5, alpha=0.5,
                label=f"Pantheon+ ({len(z)} SNe)")
    # Bin the residuals
    bins = np.linspace(0, 2.3, 20)
    zc = 0.5*(bins[1:]+bins[:-1])
    mean_r, mean_e = [], []
    for i in range(len(bins)-1):
        mask = (z >= bins[i]) & (z < bins[i+1])
        if mask.sum() > 0:
            mean_r.append(res_centered[mask].mean())
            mean_e.append(err[mask].mean()/np.sqrt(mask.sum()))
        else:
            mean_r.append(np.nan); mean_e.append(np.nan)
    ax.errorbar(zc, mean_r, yerr=mean_e, fmt="o", ms=6, color="navy",
                capsize=3, label="Биннинг")
    ax.axhline(0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\\mu_{\\rm obs} - \\mu_{\\rm model}$ (mag)")
    ax.set_xlim(0, 2.3)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig5_residuals.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")
    print(f"    diagonal chi^2 = {(w*res_centered**2).sum():.2f}")
    print(f"    mean residual  = {res_centered.mean():.4f} mag")
    print(f"    std residual   = {res_centered.std():.4f} mag")


# ---------------- Figure 6: sky map ----------------

def figure6_skymap():
    """Figure 6: Mollweide projection with residuals colored."""
    from pathlib import Path
    data_dir = Path(__file__).resolve().parent.parent / "data"
    dat = np.genfromtxt(data_dir / "Pantheon+SH0ES.dat",
                        names=True, dtype=None, encoding="utf-8")
    z = dat["zHD"].astype(float)
    m = dat["m_b_corr"].astype(float)
    err = dat["m_b_corr_err_DIAG"].astype(float)
    ra = dat["RA"].astype(float)
    dec = dat["DEC"].astype(float)
    mask = z > 0.01
    z, m, err, ra, dec = z[mask], m[mask], err[mask], ra[mask], dec[mask]

    E_func = lambda zz: E_shakti(zz, OM_PURE, alpha0=ALPHA0_PURE)
    from scipy.integrate import quad
    Dc = np.array([quad(lambda zp: 1.0/E_func(zp), 0, zi,
                        epsabs=1e-10, epsrel=1e-10)[0] for zi in z])
    mu_model = 5.0*np.log10(C_KMS/100.0 * (1.0+z) * Dc) + 25.0
    res = m - mu_model
    w = 1.0/err**2
    res = res - (w*res).sum()/w.sum()

    lon = np.deg2rad(ra)
    lon = np.where(lon > np.pi, lon - 2*np.pi, lon)  # -pi..pi
    lat = np.deg2rad(dec)

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111, projection="mollweide")
    sc = ax.scatter(lon, lat, c=res, s=8, cmap="RdBu_r",
                    vmin=-0.15, vmax=0.15, alpha=0.85)

    # Paper dipole direction and CMB dipole
    def to_rad(ra_deg, dec_deg):
        phi = np.deg2rad(ra_deg)
        if phi > np.pi:
            phi -= 2*np.pi
        return phi, np.deg2rad(dec_deg)
    ra_d, dec_d = 14.1, 17.9
    ax.scatter(*to_rad(ra_d, dec_d), marker="*", s=400, color="red",
               edgecolor="darkred", zorder=5,
               label=f"Диполь Pantheon+ ({ra_d}°, {dec_d}°)")
    ra_cmb, dec_cmb = 264.0, 48.3
    ax.scatter(*to_rad(ra_cmb, dec_cmb), marker="^", s=180, color="green",
               edgecolor="darkgreen", zorder=5,
               label="CMB-диполь")

    ax.grid(alpha=0.3)
    ax.set_title("Карта неба (проекция Mollweide), residuals Pantheon+")
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.05)
    cbar.set_label("$\\Delta\\mu$ (mag)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "fig6_skymap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Generating Figure 1 ...")
    figure1_mass_dependence()
    print("Generating Figure 2 ...")
    figure2_wz()
    print("Generating Figure 3 ...")
    figure3_Hz()
    print("Generating Figure 4 ...")
    figure4_omega_de()
    print("Generating Figure 5 ...")
    figure5_residuals()
    print("Generating Figure 6 ...")
    figure6_skymap()
    print("Done.")
