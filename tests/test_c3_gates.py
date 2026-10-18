"""C3-G gates with exact answers for the saddle tools (PLAN C3-G; report
docs/reports/Supersolid nucleation and solver literature.md, gates G2, G3).

G2: the Rayleigh quotient of a free particle on a periodic box. Every
eigenvector is a critical point on the sphere and the n-th (counting
multiplicity) has Morse index n - 1; the degenerate cos/sin partner is a zero
mode (translation), which deflation removes.

G3: the repulsive GP equation on a ring, -psi''/2 + g psi^3 = mu psi, real
field at fixed norm. psi = A sn(k x | m) with A^2 = m k^2 / g, mu = (1 + m) k^2 / 2,
k = 4 j K(m) / L (2j nodes) and N = 16 j^2 K (K - E) / (g L) (Carr, Clark &
Reinhardt, PRA 62, 063610 (2000), derived here for this normalization). Newton
converges to round-off; the Morse index is 2j - 1 (by continuation from g = 0:
the constant and j - 1 cos/sin pairs lie below), plus the translation zero mode.
The same states at fixed mu (grand canonical, no norm constraint) give their
norm N and, with dN/dmu > 0, the same index.
"""

import math

import numpy as np
import pytest
import torch

import dipgpe
from dipgpe.bdg import constrained_hessian
from dipgpe.mep import refine_saddle
from util import C128, DEVICE


def _agm(m):
    """a_n, c_n of the arithmetic-geometric mean for parameter m (Abramowitz & Stegun 16.4, 17.6)."""
    a, b, c = [1.0], [math.sqrt(1 - m)], [math.sqrt(m)]
    while abs(c[-1]) > 1e-16:
        a.append((a[-1] + b[-1]) / 2)
        c.append((a[-2] - b[-1]) / 2)
        b.append(math.sqrt(a[-2] * b[-1]))
    return a, c


def ellip_K_E(m):
    a, c = _agm(m)
    K = math.pi / (2 * a[-1])
    return K, K * (1 - 0.5 * sum(2 ** n * cn * cn for n, cn in enumerate(c)))


def sn(u, m):
    """Jacobi sn(u | m) for a float64 tensor u (descending AGM)."""
    a, c = _agm(m)
    N = len(a) - 1
    phi = 2 ** N * a[N] * u
    for n in range(N, 0, -1):
        phi = 0.5 * (phi + torch.asin(c[n] / a[n] * torch.sin(phi)))
    return torch.sin(phi)


def test_elliptic_helpers():
    K, E = ellip_K_E(0.9)
    assert abs(K - 2.5780921133481733) < 1e-14 and abs(E - 1.1047747327040733) < 1e-14   # A&S Table 17.1
    # sn by inversion: F(asin(sn(u)) | m) = u, F by Gauss-Legendre quadrature
    m, u = 0.9, torch.tensor([0.3, 0.7, 1.2, 2.0], dtype=torch.float64)
    phi = torch.asin(sn(u, m))
    t, w = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(64))
    theta = 0.5 * phi[:, None] * (t[None, :] + 1)
    F = 0.5 * phi * (w[None, :] / torch.sqrt(1 - m * torch.sin(theta) ** 2)).sum(1)
    assert float((F - u).abs().max()) < 1e-13, (F, u)


def _free_box(n=64):
    grid = dipgpe.Grid(n, 2 * math.pi, dtype=C128, device=DEVICE)
    return grid, dipgpe.GPE(grid, dipgpe.Contact(0.0)), grid.x64[0]


@pytest.mark.parametrize("j", [1, 2])
def test_g2_index_counting_with_a_zero_mode(j):
    """At cos(j x): eigenvalues (n^2 - j^2) / 2 of the other eigenvectors; sin(j x) is a
    zero mode; deflating it leaves index 2j - 1."""
    grid, model, x = _free_box()
    psi = (torch.cos(j * x) / math.sqrt(math.pi)).to(C128)
    others = sorted([-j * j / 2] + [(n * n - j * j) / 2 for n in range(1, j + 3) if n != j for _ in (0, 1)])
    lam, _, _ = constrained_hessian(model, psi, n_modes=2 * j + 2)
    exact = sorted(others[:2 * j + 1] + [0.0])
    assert max(abs(a - b) for a, b in zip(lam.tolist(), exact)) < 1e-9, (lam, exact)
    lam, _, _ = constrained_hessian(model, psi, n_modes=2 * j + 1, deflate=[torch.sin(j * x).to(C128)])
    assert max(abs(a - b) for a, b in zip(lam.tolist(), others[:2 * j + 1])) < 1e-9, (lam, others)
    assert sum(v < 0 for v in lam.tolist()) == 2 * j - 1


def test_g2_newton_on_a_degenerate_eigenspace():
    """From cos x + 0.1 (cos 2x + 1) + 0.05 sin x, Newton (translation deflated) reaches the
    eigenspace {cos x, sin x}: E = mu = 1/2, residual at round-off."""
    grid, model, x = _free_box()
    start = (torch.cos(x) + 0.1 * torch.cos(2 * x) + 0.1 + 0.05 * torch.sin(x)).to(C128)
    psi, info = refine_saddle(model, start / torch.sqrt(dipgpe.norm(grid, start)), tol=1e-12,
                              deflate=[torch.sin(x).to(C128)])
    assert info["converged"] and info["iterations"] <= 8, info
    assert abs(info["energy"] - 0.5) < 1e-12 and abs(info["mu"] - 0.5) < 1e-12, info
    assert abs(info["multipliers"][0]) < 1e-11 and info["full_residual"] < 1e-11, info  # sin x: translation


def test_g2_newton_in_the_even_sector():
    """With the even sector imposed, Newton from cos x + 0.1 (cos 2x + 1) + 0.05 sin x needs
    no deflation and lands exactly on cos x / sqrt(pi): the odd part is removed."""
    grid, model, x = _free_box()
    mirror = lambda f: torch.roll(torch.flip(f, [-1]), 1, dims=-1)          # noqa: E731
    start = (torch.cos(x) + 0.1 * torch.cos(2 * x) + 0.1 + 0.05 * torch.sin(x)).to(C128)
    psi, info = refine_saddle(model, start / torch.sqrt(dipgpe.norm(grid, start)), tol=1e-12,
                              sector=lambda f: 0.5 * (f + mirror(f)))
    assert info["converged"], info
    exact = (torch.cos(x) / math.sqrt(math.pi)).to(C128)
    err = min(float(dipgpe.norm(grid, psi - exact)), float(dipgpe.norm(grid, psi + exact))) ** 0.5
    assert err < 1e-12, err


@pytest.mark.parametrize("j", [1, 2])
def test_g3_ring_elliptic_states(j):
    """g = 1, L = 2 pi, m = 0.9 (strongly nonlinear): the analytic state is stationary on
    the grid; Newton from a perturbed state returns to it at round-off; Morse index 2j - 1
    with the translation mode deflated, and that mode is a zero eigenvalue otherwise."""
    g, L, m = 1.0, 2 * math.pi, 0.9
    grid = dipgpe.Grid(256, L, dtype=C128, device=DEVICE)
    x = grid.x64[0]
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    K, E = ellip_K_E(m)
    k = 4 * j * K / L
    N = 16 * j * j * K * (K - E) / (g * L)
    mu = 0.5 * (1 + m) * k * k
    psi = (k * math.sqrt(m / g) * sn(k * x, m)).to(C128)
    assert abs(float(dipgpe.norm(grid, psi)) - N) < 1e-11 * N
    assert float(dipgpe.residual(model, psi)) < 1e-10
    start = psi * (1 + 0.05 * torch.cos(2 * j * x)) + 0.05 * math.sqrt(N / L) * torch.sin(3 * j * x)
    dpsi = grid.ifft(1j * grid.k[0] * grid.fft(psi))
    out, info = refine_saddle(model, dipgpe.normalize(grid, start.to(C128), N), tol=1e-11, deflate=[dpsi])
    assert info["converged"] and info["iterations"] <= 10, info
    assert abs(info["multipliers"][0]) < 1e-10 and info["full_residual"] < 1e-10, info
    err = min(float(dipgpe.norm(grid, out - psi)), float(dipgpe.norm(grid, out + psi))) ** 0.5 / math.sqrt(N)
    assert err < 1e-12 and abs(info["mu"] - mu) < 1e-11 * mu, (err, info["mu"], mu)
    lam, _, _ = constrained_hessian(model, psi, n_modes=2 * j + 1, deflate=[dpsi])
    assert sum(v < 0 for v in lam.tolist()) == 2 * j - 1 and lam[2 * j - 1] > 1.0, lam
    lam, _, _ = constrained_hessian(model, psi, n_modes=2 * j)
    assert abs(float(lam[2 * j - 1])) < 1e-8, lam                       # translation zero mode


@pytest.mark.parametrize("j", [1, 2])
def test_g3_grand_canonical(j):
    """At the analytic mu and without the norm constraint, Newton from a start 5% too
    large finds the elliptic state and its norm N; the grand-canonical index is the
    canonical 2j - 1, since N grows with mu on this branch (dN/dm > 0 and dmu/dm > 0)."""
    g, L, m = 1.0, 2 * math.pi, 0.9
    grid = dipgpe.Grid(256, L, dtype=C128, device=DEVICE)
    x = grid.x64[0]
    model = dipgpe.GPE(grid, dipgpe.Contact(g))
    K, E = ellip_K_E(m)
    k = 4 * j * K / L
    N = 16 * j * j * K * (K - E) / (g * L)
    mu = 0.5 * (1 + m) * k * k
    psi = (k * math.sqrt(m / g) * sn(k * x, m)).to(C128)
    start = 1.05 * psi * (1 + 0.05 * torch.cos(2 * j * x)) + 0.05 * math.sqrt(N / L) * torch.sin(3 * j * x)
    dpsi = grid.ifft(1j * grid.k[0] * grid.fft(psi))
    out, info = refine_saddle(model, start.to(C128), mu=mu, tol=1e-11, deflate=[dpsi])
    assert info["converged"] and info["iterations"] <= 10, info
    assert abs(info["multipliers"][0]) < 1e-10 and info["full_residual"] < 1e-10, info
    err = min(float(dipgpe.norm(grid, out - psi)), float(dipgpe.norm(grid, out + psi))) ** 0.5 / math.sqrt(N)
    assert err < 1e-12 and abs(info["N"] - N) < 1e-11 * N, (err, info["N"], N)
    lam, _, _ = constrained_hessian(model, psi, n_modes=2 * j + 1, deflate=[dpsi], mu=mu)
    assert sum(v < 0 for v in lam.tolist()) == 2 * j - 1 and lam[2 * j - 1] > 0.1, lam


def test_grand_canonical_hessian_of_the_uniform_ring():
    """rho0 = 1, g = 1, L = 2 pi, mu = g rho0: the Hessian of E - mu N has eigenvalues
    k^2/2 + 2 g rho0 for k = 0, +-1, ...; on the sphere the k = 0 mode (psi0 itself)
    is removed."""
    grid = dipgpe.Grid(64, 2 * math.pi, dtype=C128, device=DEVICE)
    model = dipgpe.GPE(grid, dipgpe.Contact(1.0))
    psi = torch.ones(grid.shape, dtype=C128, device=DEVICE)
    lam, _, _ = constrained_hessian(model, psi, n_modes=3, mu=1.0)
    assert max(abs(a - b) for a, b in zip(lam.tolist(), [2.0, 2.5, 2.5])) < 1e-9, lam
    lam, _, _ = constrained_hessian(model, psi, n_modes=2)
    assert max(abs(a - b) for a, b in zip(lam.tolist(), [2.5, 2.5])) < 1e-9, lam
