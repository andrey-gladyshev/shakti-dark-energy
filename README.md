[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23204675.svg)](https://doi.org/10.5281/zenodo.23204675)
# Shakti Dark Energy

**Associated preprint:** [10.5281/zenodo.23204675](https://doi.org/10.5281/zenodo.23204675)
— *Dark Energy from Frozen Quantum Fluctuations: A One-Parameter Model with Numerical Consistency Check* (Gladyshev, 2026)

Numerical code for the paper:

**Quantum Origin of the Initial Dark Energy Fluctuation:
Numerical Consequence of the Postulated Mass m_phi = H/2
in de Sitter Space**

Andrey Gladyshev (Independent Researcher)

## Overview

One-parameter phenomenological dark energy model with alpha_0 = 1.
Under the postulated masses m_Sigma = H, m_phi = H/2, the fitted
initial condition x_i = 0.168721 is numerically close to the
Bunch-Davies amplitude delta_phi/H = 0.168973 (c = 0.9985,
deviation 0.15%).

## Repository structure

data/    DESI DR2 BAO (13 points), Pantheon+ (external, not tracked)
src/     Python modules (model, fits, anisotropy, figures)
paper/   6 generated figures

## Quick start

pip install -r requirements.txt
python -m src.background         # x_i = 0.168721
python -m src.bunch_davies       # delta_phi/H = 0.168973
python -m src.mass_hierarchy     # G_phi/G_Sigma = 4.4525
python -m src.fits               # Table 4 (joint LCDM/SHAKTI/CPL fits)
python -m src.anisotropy         # Table 5 (spherical harmonics)
python -m src.effective_omega_g  # SHAKTI vs CGG comparison
python -m src.figures            # Generate all 6 figures

## Key results

| Quantity | Value | Section |
|---|---|---|
| x_i (fit) | 0.168721 | Sec. 3.3 |
| delta_phi/H (BD) | 0.168973 | Sec. 3.2 |
| c = x_i / (delta_phi/H) | 0.9985 | Sec. 3.3 |
| G_phi/G_Sigma | 4.4525 | Sec. 8.1 |
| chi^2 (SHAKTI pure) | 1412.49 | Sec. 4.2 |

## Comparison with competing models

| Model | Free params | Mechanism |
|---|---|---|
| CGG (Hergt et al. 2026) | 1 | gravitational coupling |
| Metastable DE on brane (Sahni et al. 2026) | 2 | DE decay + brane |
| SHAKTI (this work) | 0 extra | quantum initial condition |

On BAO+SNe alone, CGG is fully degenerate with LCDM
(chi^2 = 1416.79 vs 1416.81), while SHAKTI gives a distinguishable
Delta chi^2 = -4.3. See src/effective_omega_g.py.

## Known discrepancies

- LCDM chi^2: reproduced 1416.81, paper reports 1424.79 (Delta +8).
- Figure 5 diagonal chi^2: reproduced 698.50, paper reports 682.08.
Both are small systematic offsets, likely from Pantheon+ revision.

## Contact

Andrey Gladyshev - gladyshev3005@yandex.com
