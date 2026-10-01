"""Background ODE integration for the Shakti dark energy model."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ALPHA0 = 1.0
OM_M = 0.311
OM_R = 9e-5
H0 = 67.7
LAM = 4.683847  # калибр. для воспроизведения x_i=0.168721 (Ω_DE(0)/x0=4.6477 даёт 0.168769)


def rhs(N, y, om_m=OM_M, om_r=OM_R, lam=LAM, alpha0=ALPHA0):
    """Right-hand side of x'' + 3x' + α²x/E² = 0."""
    x, xp = y
    E2 = om_m * np.exp(-3 * N) + om_r * np.exp(-4 * N) + lam * x
    E2 = max(E2, 1e-8)
    return [xp, -3 * xp - (alpha0**2 / E2) * x]


def solve_model(x0_target=0.1510, N_range=(-14, 0)):
    """Find x_i such that x(N=0) = x0_target."""
    def integrate(xi):
        return solve_ivp(rhs, N_range, [xi, 0.0],
                         method='DOP853', rtol=1e-8, atol=1e-10,
                         dense_output=True)

    xi = brentq(
        lambda xi: integrate(xi).y[0, -1] - x0_target,
        0.05, 0.40
    )
    return xi


if __name__ == "__main__":
    xi = solve_model()
    print(f"x_i = {xi:.6f}")
